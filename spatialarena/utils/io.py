# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import io
import os
import sys
import re

import numpy as np


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    # greyscale
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)
    file.close()


def read_pfm_bytes(data: bytes):
    stream = io.BytesIO(data)
    header = stream.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s*$', stream.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(stream.readline().rstrip())
    endian = '<' if scale < 0 else '>'
    scale = abs(scale)

    count = width * height * (3 if color else 1)
    pixels = np.frombuffer(stream.read(count * 4), dtype=endian + 'f4', count=count)
    shape = (height, width, 3) if color else (height, width)
    return np.flipud(pixels.reshape(shape)).copy(), scale


def read_exr_bytes(data: bytes, channel: int = None):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2  # pylint: disable=C0415
        pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:  # pylint: disable=W0703
        pixels = None
    if pixels is None:
        import imageio.v3 as iio  # pylint: disable=C0415
        pixels = np.asarray(iio.imread(data, extension=".exr"))
    pixels = np.asarray(pixels, dtype=np.float32)
    if channel is not None and pixels.ndim == 3:
        pixels = pixels[..., channel]
    return pixels


_PLY_DTYPES = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
}


def read_ply_bytes(data: bytes):
    """Parse a binary or ascii PLY into {element: {property: array}}.

    List properties (e.g. face vertex_indices) are returned as (N, K) arrays when every
    entry has the same length, otherwise as a Python list of arrays.
    """
    end = data.find(b'end_header')
    if end < 0:
        raise Exception('Not a PLY file.')
    end = data.find(b'\n', end) + 1
    header_lines = data[:end].decode('ascii', errors='replace').splitlines()
    if not header_lines or header_lines[0].strip() != 'ply':
        raise Exception('Not a PLY file.')

    fmt = None
    elements = []
    for line in header_lines[1:]:
        parts = line.strip().split()
        if not parts or parts[0] in ('comment', 'obj_info', 'end_header'):
            continue
        if parts[0] == 'format':
            fmt = parts[1]
        elif parts[0] == 'element':
            elements.append({'name': parts[1], 'count': int(parts[2]), 'properties': []})
        elif parts[0] == 'property':
            if parts[1] == 'list':
                elements[-1]['properties'].append(
                    {'name': parts[4], 'list': True, 'count_type': parts[2], 'item_type': parts[3]})
            else:
                elements[-1]['properties'].append({'name': parts[2], 'list': False, 'type': parts[1]})

    if fmt is None:
        raise Exception('Malformed PLY header.')
    endian = '<' if fmt == 'binary_little_endian' else '>'
    if fmt == 'ascii':
        return _read_ply_ascii(data[end:], elements)

    body = memoryview(data)[end:]
    offset = 0
    result = {}
    for element in elements:
        props = element['properties']
        count = element['count']
        if not any(p['list'] for p in props):
            dtype = np.dtype([(p['name'], endian + _PLY_DTYPES[p['type']]) for p in props])
            values = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
            offset += count * dtype.itemsize
            result[element['name']] = {name: np.array(values[name]) for name in dtype.names}
            continue

        # Fast path: every list property has the same length on every row (quad/tri meshes),
        # so the whole element can be viewed as one structured array.
        fields, probe, list_names = [], offset, []
        for prop in props:
            if prop['list']:
                count_dtype = np.dtype(endian + _PLY_DTYPES[prop['count_type']])
                item_dtype = np.dtype(endian + _PLY_DTYPES[prop['item_type']])
                first = int(np.frombuffer(body, dtype=count_dtype, count=1, offset=probe)[0])
                fields.append((prop['name'] + '__n', count_dtype))
                fields.append((prop['name'], item_dtype, (first,)))
                list_names.append((prop['name'], first))
                probe += count_dtype.itemsize + first * item_dtype.itemsize
            else:
                dtype = np.dtype(endian + _PLY_DTYPES[prop['type']])
                fields.append((prop['name'], dtype))
                probe += dtype.itemsize
        row = np.dtype(fields)
        if offset + row.itemsize * count <= len(body):
            rows = np.frombuffer(body, dtype=row, count=count, offset=offset)
            if all(np.all(rows[name + '__n'] == first) for name, first in list_names):
                result[element['name']] = {
                    p['name']: np.array(rows[p['name']]) for p in props
                }
                offset += row.itemsize * count
                continue

        values = {p['name']: [] for p in props}
        for _ in range(count):
            for prop in props:
                if prop['list']:
                    count_dtype = np.dtype(endian + _PLY_DTYPES[prop['count_type']])
                    item_dtype = np.dtype(endian + _PLY_DTYPES[prop['item_type']])
                    n = int(np.frombuffer(body, dtype=count_dtype, count=1, offset=offset)[0])
                    offset += count_dtype.itemsize
                    values[prop['name']].append(
                        np.frombuffer(body, dtype=item_dtype, count=n, offset=offset).copy())
                    offset += n * item_dtype.itemsize
                else:
                    dtype = np.dtype(endian + _PLY_DTYPES[prop['type']])
                    values[prop['name']].append(np.frombuffer(body, dtype=dtype, count=1, offset=offset)[0])
                    offset += dtype.itemsize
        for prop in props:
            if not prop['list']:
                values[prop['name']] = np.asarray(values[prop['name']])
            else:
                lengths = {len(v) for v in values[prop['name']]}
                if len(lengths) == 1:
                    values[prop['name']] = np.stack(values[prop['name']]) if values[prop['name']] else np.zeros((0, 0))
        result[element['name']] = values
    return result


def _read_ply_ascii(body: bytes, elements):
    tokens = body.split()
    position = 0
    result = {}
    for element in elements:
        values = {p['name']: [] for p in element['properties']}
        for _ in range(element['count']):
            for prop in element['properties']:
                if prop['list']:
                    n = int(tokens[position]); position += 1
                    values[prop['name']].append(np.array(tokens[position:position + n], dtype=np.float64))
                    position += n
                else:
                    values[prop['name']].append(float(tokens[position])); position += 1
        for prop in element['properties']:
            if not prop['list']:
                values[prop['name']] = np.asarray(values[prop['name']])
            elif len({len(v) for v in values[prop['name']]}) == 1 and values[prop['name']]:
                values[prop['name']] = np.stack(values[prop['name']])
        result[element['name']] = values
    return result

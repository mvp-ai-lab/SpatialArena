# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Readers and writers of the depth, image and point-cloud file formats.

PFM (files and bytes), EXR (bytes, by OpenCV with an imageio fallback) and
PLY (binary or ascii bytes, parsed without third-party PLY libraries).
"""

import io
import os
import re
import sys

import numpy as np


def read_pfm(filename):
    """Reads a PFM file.

    Args:
        filename: The .pfm path.

    Returns:
        The (H, W) or (H, W, 3) image, top row first, and the absolute scale.

    Raises:
        Exception: If the file is not a well-formed PFM file.
    """
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
    if scale < 0:  # Little-endian.
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # Big-endian.

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def write_pfm(file, image, scale=1):
    """Writes a float32 image as a PFM file.

    Args:
        file: The .pfm path.
        image: (H, W), (H, W, 1) or (H, W, 3) float32 image, top row first.
        scale: The PFM scale; its sign records the byte order.

    Raises:
        Exception: If the image is not float32 or has another shape.
    """
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # Color image.
        color = True
    # Greyscale.
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
    """Reads PFM bytes, as `read_pfm` reads a file.

    Raises:
        Exception: If the data is not a well-formed PFM file.
    """
    stream = io.BytesIO(data)
    header = stream.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s*$',
                         stream.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(stream.readline().rstrip())
    endian = '<' if scale < 0 else '>'
    scale = abs(scale)

    count = width * height * (3 if color else 1)
    pixels = np.frombuffer(stream.read(count * 4),
                           dtype=endian + 'f4',
                           count=count)
    shape = (height, width, 3) if color else (height, width)
    return np.flipud(pixels.reshape(shape)).copy(), scale


def read_exr_bytes(data: bytes, channel: int = None):
    """Decodes EXR bytes to a float32 array.

    Uses OpenCV (with OPENCV_IO_ENABLE_OPENEXR defaulted to 1) and falls back
    to imageio when OpenCV fails.

    Args:
        data: The EXR file contents.
        channel: The channel to return of a multi-channel image; None
            returns all channels.
    """
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2  # pylint: disable=C0415
        pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                              cv2.IMREAD_UNCHANGED)
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
    'char': 'i1',
    'int8': 'i1',
    'uchar': 'u1',
    'uint8': 'u1',
    'short': 'i2',
    'int16': 'i2',
    'ushort': 'u2',
    'uint16': 'u2',
    'int': 'i4',
    'int32': 'i4',
    'uint': 'u4',
    'uint32': 'u4',
    'float': 'f4',
    'float32': 'f4',
    'double': 'f8',
    'float64': 'f8',
}


def _parse_ply_header(header_lines):
    """Returns the format and the element descriptions of a PLY header."""
    fmt = None
    elements = []
    for line in header_lines[1:]:
        parts = line.strip().split()
        if not parts or parts[0] in ('comment', 'obj_info', 'end_header'):
            continue
        if parts[0] == 'format':
            fmt = parts[1]
        elif parts[0] == 'element':
            elements.append({
                'name': parts[1],
                'count': int(parts[2]),
                'properties': []
            })
        elif parts[0] == 'property':
            if parts[1] == 'list':
                elements[-1]['properties'].append({
                    'name': parts[4],
                    'list': True,
                    'count_type': parts[2],
                    'item_type': parts[3]
                })
            else:
                elements[-1]['properties'].append({
                    'name': parts[2],
                    'list': False,
                    'type': parts[1]
                })
    return fmt, elements


def read_ply_bytes(data: bytes):
    """Parses a binary or ascii PLY into {element: {property: array}}.

    List properties (e.g. face vertex_indices) are returned as (N, K) arrays
    when every entry has the same length, otherwise as a Python list of
    arrays.

    Raises:
        Exception: If the data is not a well-formed PLY file.
    """
    end = data.find(b'end_header')
    if end < 0:
        raise Exception('Not a PLY file.')
    end = data.find(b'\n', end) + 1
    header_lines = data[:end].decode('ascii', errors='replace').splitlines()
    if not header_lines or header_lines[0].strip() != 'ply':
        raise Exception('Not a PLY file.')

    fmt, elements = _parse_ply_header(header_lines)
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
            values, offset = _read_ply_scalar_element(body, offset, count,
                                                      props, endian)
            result[element['name']] = values
            continue

        values, new_offset = _read_ply_uniform_list_element(
            body, offset, count, props, endian)
        if values is None:
            values, new_offset = _read_ply_element_rows(body, offset, count,
                                                        props, endian)
        result[element['name']] = values
        offset = new_offset
    return result


def _read_ply_scalar_element(body, offset, count, props, endian):
    """Reads a binary PLY element without list properties.

    Returns:
        The property arrays and the offset after the element.
    """
    dtype = np.dtype([
        (p['name'], endian + _PLY_DTYPES[p['type']]) for p in props
    ])
    values = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
    offset += count * dtype.itemsize
    return {name: np.array(values[name]) for name in dtype.names}, offset


def _read_ply_uniform_list_element(body, offset, count, props, endian):
    """Reads a binary PLY element whose list lengths never vary.

    This is the fast path of quad and triangle meshes: every list property
    has the same length on every row, so the whole element can be viewed as
    one structured array. The lengths are probed on the first row.

    Returns:
        The property arrays and the offset after the element, or (None,
        offset) if the lengths vary or the data is too short.
    """
    fields, probe, list_names = [], offset, []
    for prop in props:
        if prop['list']:
            count_dtype = np.dtype(endian + _PLY_DTYPES[prop['count_type']])
            item_dtype = np.dtype(endian + _PLY_DTYPES[prop['item_type']])
            first = int(
                np.frombuffer(body, dtype=count_dtype, count=1,
                              offset=probe)[0])
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
        if all(
                np.all(rows[name + '__n'] == first)
                for name, first in list_names):
            values = {p['name']: np.array(rows[p['name']]) for p in props}
            return values, offset + row.itemsize * count
    return None, offset


def _read_ply_element_rows(body, offset, count, props, endian):
    """Reads a binary PLY element row by row.

    Returns:
        The property arrays (lists of arrays for list properties of varying
        length) and the offset after the element.
    """
    values = {p['name']: [] for p in props}
    for _ in range(count):
        for prop in props:
            if prop['list']:
                count_dtype = np.dtype(endian + _PLY_DTYPES[prop['count_type']])
                item_dtype = np.dtype(endian + _PLY_DTYPES[prop['item_type']])
                n = int(
                    np.frombuffer(body,
                                  dtype=count_dtype,
                                  count=1,
                                  offset=offset)[0])
                offset += count_dtype.itemsize
                values[prop['name']].append(
                    np.frombuffer(body,
                                  dtype=item_dtype,
                                  count=n,
                                  offset=offset).copy())
                offset += n * item_dtype.itemsize
            else:
                dtype = np.dtype(endian + _PLY_DTYPES[prop['type']])
                values[prop['name']].append(
                    np.frombuffer(body, dtype=dtype, count=1, offset=offset)[0])
                offset += dtype.itemsize
    for prop in props:
        name = prop['name']
        if not prop['list']:
            values[name] = np.asarray(values[name])
        else:
            lengths = {len(v) for v in values[name]}
            if len(lengths) == 1:
                if values[name]:
                    values[name] = np.stack(values[name])
                else:
                    values[name] = np.zeros((0, 0))
    return values, offset


def _read_ply_ascii(body: bytes, elements):
    """Reads the body of an ascii PLY; list items are float64 arrays."""
    tokens = body.split()
    position = 0
    result = {}
    for element in elements:
        values = {p['name']: [] for p in element['properties']}
        for _ in range(element['count']):
            for prop in element['properties']:
                if prop['list']:
                    n = int(tokens[position])
                    position += 1
                    values[prop['name']].append(
                        np.array(tokens[position:position + n],
                                 dtype=np.float64))
                    position += n
                else:
                    values[prop['name']].append(float(tokens[position]))
                    position += 1
        for prop in element['properties']:
            if not prop['list']:
                values[prop['name']] = np.asarray(values[prop['name']])
            else:
                lengths = {len(v) for v in values[prop['name']]}
                if len(lengths) == 1 and values[prop['name']]:
                    values[prop['name']] = np.stack(values[prop['name']])
        result[element['name']] = values
    return result

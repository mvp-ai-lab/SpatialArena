# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import json
from pathlib import Path

import yaml
from PIL import Image

from scripts.preprocess.spatialarena_benchmark import apply_benchmark_review as abr


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def _bytes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'depth')


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (32, 26), (90, 120, 150)).save(path)


def _label(labels: Path, track: str, scene: str, keep: bool = True, drop: tuple = ()) -> None:
    """One scene's review label, as the MindCloud monitor's Benchmark Review tab (FeedForward-Benchmark
    ``app/mindcloud_monitor/benchmark_review.py``) saves it: only departures from the defaults are stored."""
    review = json.loads(labels.read_text()) if labels.is_file() else {'tracks': {}}
    label = review['tracks'].setdefault(track, {}).setdefault(scene, {})
    if not keep:
        label['keep'] = False
    if drop:
        label['drop'] = sorted(set(label.get('drop', [])) | set(drop))
    review['updated'] = 't'
    labels.write_text(json.dumps(review, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _benchmark(root: Path) -> None:
    """MonoDepth (3 frames), VideoDepth (one 4-frame clip) and MVRecon (one scene), as the materializer lays out."""
    mono = root / 'MonoDepth' / 'SceneA'
    keys = ['cam0__img0_000001', 'cam1__img1_000005', 'cam2__img2_000009']
    for key in keys:
        _image(mono / 'images' / f'{key}.jpg')
        _image(mono / 'depth_heatmaps' / f'{key}.png')
        _image(mono / 'lidar_depth_heatmaps' / f'{key}.png')
        _bytes(mono / 'depths' / f'{key}.npy')
        _image(mono / 'lidar_depths' / f'{key}.png')
    _write(mono / 'materialized_full.json', {'num_samples': 3, 'records': [
        {'key': key, 'image': f'images/{key}.jpg', 'depth': f'depths/{key}.npy', 'source': key} for key in keys]})
    _write(mono / 'scene.json', {'scene_name': 'SceneA', 'source_scene': 'SceneA-1', 'total_images': 3,
                                 'total_depth_maps': 3})
    _write(root / 'MonoDepth' / 'metadata.json', {'task': 'MonoDepth', 'total_images': 3, 'scenes': [
        {'scene_name': 'SceneA', 'total_images': 3, 'total_depth_maps': 3}]})

    for clip in ('SceneB_clip00_cam1', 'SceneB_clip01_cam0'):
        video = root / 'VideoDepth' / clip
        manifest = []
        for index in range(4):
            stem = f'{index:06d}'
            _image(video / 'images' / f'{stem}.jpg')
            _bytes(video / 'depths' / f'{stem}.npy')
            for dirname in ('lidar_depths', 'depth_heatmaps', 'lidar_depth_heatmaps'):
                _image(video / dirname / f'{stem}.png')
            manifest.append({'frame': index, 'rgb': f'images/{stem}.jpg', 'depth': f'depths/{stem}.npy',
                             'lidar_depth': f'lidar_depths/{stem}.png'})
        _write(video / 'frame_manifest.json', manifest)
        _write(video / 'scene.json', {'scene_name': clip, 'source_scene': 'SceneB-1', 'total_images': 4})
    _write(root / 'VideoDepth' / 'metadata.json', {'task': 'VideoDepth', 'total_images': 8, 'scenes': [
        {'scene_name': 'SceneB_clip00_cam1', 'total_images': 4},
        {'scene_name': 'SceneB_clip01_cam0', 'total_images': 4}]})
    _write(root / 'VideoDepth' / 'clips.json', {'seed': 0, 'clips': {'SceneB': [
        {'scene_name': 'SceneB_clip00_cam1'}, {'scene_name': 'SceneB_clip01_cam0'}]}})

    mv = root / 'MVRecon' / 'SceneC'
    for index in range(3):
        _image(mv / 'images' / f'img0_{index:06d}.jpg')
    _write(mv / 'scene.json', {'scene_name': 'SceneC', 'source_scene': 'SceneC-1', 'total_images': 3})
    _write(root / 'MVRecon' / 'metadata.json', {'task': 'MVRecon', 'total_images': 3, 'scenes': [
        {'scene_name': 'SceneC', 'total_images': 3}]})
    (root / 'MVRecon' / '.staging' / 'SceneD').mkdir(parents=True)  # a scene the materializer is writing


SELECTION = """\
# The scene selection (comments are kept).
seed: 0
monodepth:
  scenes: [SceneA, SceneF]  # own scenes
  sample_from: [videodepth, mvrecon]
  fraction: 0.10
videodepth:
  scenes: [SceneB, SceneE]
  clip_frames: [100, 500]
  whole: [SceneE]
mvrecon:
  views: 100
  scenes:
    SceneC: {setting: indoor, eval_threshold: 0.05}
    SceneG: {setting: outdoor, eval_threshold: 0.25}
sfm:
  scenes: [SceneC, SceneH]
slam:
  frame_rate_hz: 5.0
  sequences:
    - {scene: SceneI, range: short, camera: 1, input: rgb_depth}
"""


def test_apply_review(tmp_path, capsys):
    root, labels, selection = tmp_path / 'benchmark', tmp_path / 'review.json', tmp_path / 'selection.yaml'
    _benchmark(root)
    selection.write_text(SELECTION, encoding='utf-8')
    _label(labels, 'MonoDepth', 'SceneA', drop=('cam1__img1_000005',))
    _label(labels, 'VideoDepth', 'SceneB_clip00_cam1', drop=('000002',))
    _label(labels, 'VideoDepth', 'SceneB_clip01_cam0', keep=False)
    _label(labels, 'MVRecon', 'SceneC', keep=False)
    # Scenes this small benchmark does not hold: the selection is still updated.
    review = json.loads(labels.read_text())
    review['tracks']['MonoDepth']['SceneB'] = {'keep': False}  # sampled from VideoDepth: excluded
    review['tracks']['MonoDepth']['SceneF'] = {'keep': False}  # an own scene: removed
    review['tracks']['VideoDepth']['SceneE_full_cam1'] = {'keep': False}  # its only clip: the scene goes
    review['tracks']['SfM'] = {'SceneH': {'keep': False}}
    review['tracks']['SLAM'] = {'SceneI': {'keep': False}}
    labels.write_text(json.dumps(review))
    args = ['--labels', str(labels), '--benchmark_root', str(root), '--selection', str(selection)]

    # A dry run changes nothing; --apply refuses while the materializer is staging a scene.
    assert abr.main(args) == 0
    assert (root / 'MonoDepth' / 'SceneA' / 'images' / 'cam1__img1_000005.jpg').is_file()
    assert selection.read_text() == SELECTION
    assert abr.main([*args, '--apply']) == 1
    (root / 'MVRecon' / '.staging' / 'SceneD').rmdir()
    assert abr.main([*args, '--apply']) == 0

    mono = root / 'MonoDepth' / 'SceneA'
    assert not list(mono.rglob('cam1__img1_000005.*'))
    assert len(list(mono.rglob('cam0__img0_000001.*'))) == 5
    manifest = json.loads((mono / 'materialized_full.json').read_text())
    assert manifest['num_samples'] == 2 and len(manifest['records']) == 2
    record = json.loads((mono / 'scene.json').read_text())
    assert record['total_images'] == 2 and record['manual_review'] == {'excluded_frames': ['cam1__img1_000005']}
    metadata = json.loads((root / 'MonoDepth' / 'metadata.json').read_text())
    assert metadata['total_images'] == 2 and metadata['scenes'][0]['total_depth_maps'] == 2

    clip = root / 'VideoDepth' / 'SceneB_clip00_cam1'
    assert (clip / 'images' / '000002.jpg').is_file()  # the clip stays continuous
    assert sorted(path.parent.name for path in clip.rglob('000002.*')) == ['images']
    frames = json.loads((clip / 'frame_manifest.json').read_text())
    assert frames[2]['depth'] is None and frames[2]['gt_excluded'] and frames[1]['depth'] == 'depths/000001.npy'
    assert json.loads((clip / 'scene.json').read_text())['gt_frames'] == 3
    assert not (root / 'VideoDepth' / 'SceneB_clip01_cam0').exists()
    video = json.loads((root / 'VideoDepth' / 'metadata.json').read_text())
    assert [scene['scene_name'] for scene in video['scenes']] == ['SceneB_clip00_cam1'] and video['total_images'] == 4
    clips = json.loads((root / 'VideoDepth' / 'clips.json').read_text())
    assert clips['clips'] == {'SceneB': [{'scene_name': 'SceneB_clip00_cam1'}]}
    assert not (root / 'MVRecon' / 'SceneC').exists()
    assert json.loads((root / 'MVRecon' / 'metadata.json').read_text())['scenes'] == []

    text = selection.read_text()
    assert text.startswith('# The scene selection (comments are kept).') and '# own scenes' in text
    updated = yaml.safe_load(text)
    # SceneC left MVRecon (SfM keeps it) and SceneE VideoDepth; MonoDepth keeps its samples of both as own scenes.
    assert updated['monodepth']['scenes'] == ['SceneA', 'SceneC', 'SceneE']
    assert updated['monodepth']['exclude'] == ['SceneB']
    assert updated['videodepth']['scenes'] == ['SceneB'] and updated['videodepth']['whole'] == []
    assert updated['videodepth']['exclude_clips'] == ['SceneB_clip01_cam0']
    assert list(updated['mvrecon']['scenes']) == ['SceneG'] and updated['sfm']['scenes'] == ['SceneC']
    assert updated['slam']['sequences'] == []

    # A second run finds nothing left to change.
    capsys.readouterr()
    assert abr.main([*args, '--apply']) == 0
    out = capsys.readouterr().out
    assert out.count('  nothing') == 2 and selection.read_text() == text


def test_monodepth_labels_reach_videodepth_gt(tmp_path):
    root, labels, selection = tmp_path / 'benchmark', tmp_path / 'review.json', tmp_path / 'selection.yaml'
    _benchmark(root)
    (root / 'MVRecon' / '.staging' / 'SceneD').rmdir()
    selection.write_text(SELECTION, encoding='utf-8')
    clip = root / 'VideoDepth' / 'SceneB_clip00_cam1'
    manifest = json.loads((clip / 'frame_manifest.json').read_text())
    for entry in manifest:
        entry['source_rgb'] = f"cam1/img1_{entry['frame'] + 40}.JPEG"
    _write(clip / 'frame_manifest.json', manifest)
    # MonoDepth samples SceneB: its left-out image cam1 #41 is clip frame 000001; cam0 #41 is in no clip.
    _write(labels, {'updated': 't', 'tracks': {'MonoDepth': {'SceneB': {'drop': ['cam1__img1_000041',
                                                                                 'cam0__img0_000041']}}}})
    assert abr.main(['--labels', str(labels), '--benchmark_root', str(root), '--selection', str(selection),
                     '--apply']) == 0

    frames = json.loads((clip / 'frame_manifest.json').read_text())
    assert [entry.get('gt_excluded', False) for entry in frames] == [False, True, False, False]
    assert (clip / 'images' / '000001.jpg').is_file() and not (clip / 'depths' / '000001.npy').exists()
    assert len(list((root / 'VideoDepth' / 'SceneB_clip01_cam0').rglob('*.npy'))) == 4  # another camera's clip
    record = json.loads((root / 'benchmark_review.json').read_text())
    assert record['labels'] == {'MonoDepth': {'SceneB': {'drop': ['cam1__img1_000041', 'cam0__img0_000041']}}}
    assert record['excluded_frames'] == {'VideoDepth': {'SceneB_clip00_cam1': ['000001']}}

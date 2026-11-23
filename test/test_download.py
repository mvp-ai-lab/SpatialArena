# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from spatialarena.download import (
    DATASET_DOWNLOADS,
    download_all_supported_datasets,
    download_dataset,
    list_supported_datasets,
)


def test_list_supported_datasets_covers_the_full_task_track_matrix_dataset_set(
):
    from spatialarena.eval.dataset_registry import DEFAULT_DATASET_SPECS

    listed = {(e["task"], e["dataset"]) for e in list_supported_datasets()}
    registered = {(s.task, s.dataset) for s in DEFAULT_DATASET_SPECS}
    missing = registered - listed
    assert not missing, f"registered datasets with no download registry entry: {missing}"


def test_list_supported_datasets_never_claims_automated_without_verified_steps(
):
    for entry in list_supported_datasets():
        if entry["status"] == "automated":
            # Either it has its own steps, or its notes explain which sibling
            # entry actually materializes it (e.g. hiroom via scannetpp).
            assert entry["automated_steps"] or entry["notes"], entry


def _mock_run(returncode=0):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = "ok\n"
    result.stderr = ""
    return result


def test_download_dataset_skips_non_automated_entries_without_running_anything(
        tmp_path):
    entry = next(e for e in DATASET_DOWNLOADS if e.status == "manual")
    with patch("spatialarena.download.subprocess.run") as mock_run:
        result = download_dataset(
            entry,
            benchmark_root=tmp_path,
            staging_root=tmp_path,
            process_env={},
            log_dir=tmp_path / "logs",
        )
    mock_run.assert_not_called()
    assert result["status"] == "skipped"
    assert "manual" in result["reason"]


def test_download_dataset_runs_steps_in_order_and_stops_on_failure(tmp_path):
    entry = next(e for e in DATASET_DOWNLOADS
                 if e.task == "MonoDepth" and e.dataset == "marigold")
    assert len(entry.steps) == 2  # download, materialize

    calls = []

    def fake_run(argv, **kwargs):  # pylint: disable=unused-argument
        calls.append(argv)
        return _mock_run(returncode=1)  # first step fails

    with patch("spatialarena.download.subprocess.run", side_effect=fake_run):
        result = download_dataset(
            entry,
            benchmark_root=tmp_path,
            staging_root=tmp_path,
            process_env={},
            log_dir=tmp_path / "logs",
        )

    assert len(calls) == 1  # stopped after the first (failed) step
    assert result["status"] == "failed"
    assert len(result["steps"]) == 1


def test_download_dataset_formats_benchmark_root_into_env_and_args(tmp_path):
    entry = next(e for e in DATASET_DOWNLOADS
                 if e.task == "MonoDepth" and e.dataset == "marigold")

    seen_envs = []
    seen_argvs = []

    def fake_run(argv, **kwargs):
        seen_argvs.append(argv)
        seen_envs.append(kwargs["env"])
        return _mock_run(returncode=0)

    with patch("spatialarena.download.subprocess.run", side_effect=fake_run):
        result = download_dataset(
            entry,
            benchmark_root=tmp_path / "benchmarks",
            staging_root=tmp_path / "staging",
            process_env={"PATH": "/usr/bin"},
            log_dir=tmp_path / "logs",
        )

    assert result["status"] == "ok"
    assert seen_envs[0]["DATA_ROOT_PATH"] == str(tmp_path / "benchmarks" /
                                                 "MonoDepth")
    assert seen_envs[0]["PATH"] == "/usr/bin"  # base process env preserved
    materialize_argv = seen_argvs[1]
    assert str(tmp_path / "benchmarks" / "MonoDepth" /
               "marigold") in materialize_argv


def test_download_all_supported_datasets_respects_dataset_filter(tmp_path):
    with patch("spatialarena.download.subprocess.run", return_value=_mock_run()), \
         patch("spatialarena.download.describe_benchmark_scenes", return_value=[]):
        results = download_all_supported_datasets(
            benchmark_root=tmp_path,
            datasets=[("VideoDepth", "sintel")],
            staging_root=tmp_path,
            log_dir=tmp_path / "logs",
            process_env={},
            skip_existing=False,
        )
    assert len(results) == 1
    assert results[0]["task"] == "VideoDepth"
    assert results[0]["dataset"] == "sintel"


def test_download_all_supported_datasets_skips_already_present(tmp_path):

    def fake_describe(*, benchmark_root, tasks, datasets):  # pylint: disable=unused-argument
        return [{
            "exists": True,
            "metadata_required": True,
            "metadata_exists": True
        }]

    with patch("spatialarena.download.subprocess.run") as mock_run, \
         patch("spatialarena.download.describe_benchmark_scenes", side_effect=fake_describe):
        results = download_all_supported_datasets(
            benchmark_root=tmp_path,
            datasets=[("VideoDepth", "sintel")],
            staging_root=tmp_path,
            log_dir=tmp_path / "logs",
            process_env={},
            skip_existing=True,
        )
    mock_run.assert_not_called()
    assert results[0]["status"] == "already_present"


def test_list_supported_models_matches_registry_size():
    from spatialarena.download import list_supported_models
    from spatialarena.methods import METHOD_SPECS

    models = list_supported_models()
    assert len(models) == len(METHOD_SPECS)
    assert sum(1 for m in models if m["automatable"]) == sum(
        1 for m in models
        if m["ckpt_kind"] in ("hf_hub", "url", "gdrive", "hf_file",
                              "shipped_with_code", "hf_local_files"))


def test_download_checkpoint_hf_hub_calls_snapshot_download(tmp_path):
    from spatialarena.download import download_checkpoint
    from spatialarena.methods import resolve_method

    spec = resolve_method("PI3_X")
    with patch("huggingface_hub.snapshot_download",
               return_value="/fake/snapshot/path") as mock_snapshot:
        result = download_checkpoint(spec,
                                     shared_checkpoint_root=tmp_path / "ckpt",
                                     shared_hf_home=tmp_path / "hf")
    mock_snapshot.assert_called_once_with(repo_id=spec.ckpt_default,
                                          cache_dir=str(tmp_path / "hf"))
    assert result == {
        "name": "PI3_X",
        "status": "ok",
        "kind": "hf_hub",
        "path": "/fake/snapshot/path"
    }


def test_download_checkpoint_hf_hub_not_found_is_skipped_not_a_crash(tmp_path):
    # A registered hf_hub ckpt_default can 404 (not yet released, e.g. VLM3's
    # facebook/VLM3-depth as of this writing) or be gated -- either way this
    # must report "skipped" with a real reason, never raise and take down
    # the rest of a download_all_supported_models() batch.
    from unittest.mock import Mock

    from huggingface_hub.utils import HfHubHTTPError

    from spatialarena.download import download_checkpoint
    from spatialarena.methods import resolve_method

    spec = resolve_method("PI3_X")
    with patch("huggingface_hub.snapshot_download",
               side_effect=HfHubHTTPError("not found", response=Mock())):
        result = download_checkpoint(spec,
                                     shared_checkpoint_root=tmp_path / "ckpt",
                                     shared_hf_home=tmp_path / "hf")
    assert result["status"] == "skipped"
    assert result["kind"] == "hf_hub"
    assert spec.ckpt_default in result["reason"]


def test_download_checkpoint_url_downloads_to_local_checkpoint_store(tmp_path):
    from spatialarena.download import download_checkpoint
    from spatialarena.methods import resolve_method

    spec = resolve_method("DUST3R")
    fake_response = MagicMock()
    fake_response.iter_content.return_value = [b"weights"]
    fake_response.raise_for_status.return_value = None
    fake_response.__enter__ = lambda self: fake_response
    fake_response.__exit__ = lambda self, *a: None

    with patch("requests.get", return_value=fake_response) as mock_get:
        result = download_checkpoint(spec,
                                     shared_checkpoint_root=tmp_path / "ckpt",
                                     shared_hf_home=tmp_path / "hf")

    mock_get.assert_called_once()
    assert result["status"] == "ok"
    assert result["kind"] == "url"
    dest = tmp_path / "ckpt" / "local" / "dust3r" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    assert dest.read_bytes() == b"weights"
    assert result["path"] == str(dest)


def test_download_checkpoint_local_file_is_skipped_not_guessed(tmp_path):
    from spatialarena.download import download_checkpoint
    from spatialarena.methods import resolve_method

    spec = resolve_method("VGGT")
    with patch("requests.get") as mock_get:
        result = download_checkpoint(spec,
                                     shared_checkpoint_root=tmp_path / "ckpt",
                                     shared_hf_home=tmp_path / "hf")
    mock_get.assert_not_called()
    assert result["status"] == "skipped"
    assert result["kind"] == "local_file"


def test_download_checkpoint_gdrive_variable_dependent_uses_gdown(tmp_path):
    from spatialarena.download import GDRIVE_CHECKPOINT_URLS, download_checkpoint
    from spatialarena.methods import resolve_method

    spec = resolve_method("CUT3R")
    method_code_root = tmp_path / "code"
    expected_dest = Path(
        spec.ckpt_default.replace("$METHOD_CODE_ROOT", str(method_code_root)))
    with patch("subprocess.run") as mock_run:
        result = download_checkpoint(
            spec,
            shared_checkpoint_root=tmp_path / "ckpt",
            shared_hf_home=tmp_path / "hf",
            method_code_root=method_code_root,
        )
    mock_run.assert_called_once_with(
        ["gdown", GDRIVE_CHECKPOINT_URLS["CUT3R"], "-O",
         str(expected_dest)],
        check=True,
    )
    assert result["status"] == "ok"
    assert result["kind"] == "gdrive"
    assert result["path"] == str(expected_dest)
    assert expected_dest.parent.is_dir()


def test_download_checkpoint_gdrive_without_method_code_root_is_skipped_not_guessed(
):
    from spatialarena.download import download_checkpoint
    from spatialarena.methods import resolve_method

    spec = resolve_method("CUT3R")
    with patch("subprocess.run") as mock_run:
        result = download_checkpoint(spec,
                                     shared_checkpoint_root=Path("/tmp/ckpt"),
                                     shared_hf_home=Path("/tmp/hf"))
    mock_run.assert_not_called()
    assert result["status"] == "skipped"


def test_come_checkpoint_is_variable_dependent_not_gdrive():
    # COME's checkpoint ships committed inside the CoMe code clone itself --
    # it has no separate Google Drive source, so it must stay "skipped" and
    # never be miscategorized as gdrive-automatable alongside CUT3R.
    from spatialarena.download import GDRIVE_CHECKPOINT_URLS

    assert "COME" not in GDRIVE_CHECKPOINT_URLS


def test_download_checkpoint_shipped_with_code_copies_from_the_clone(tmp_path):
    from spatialarena.download import SHIPPED_WITH_CODE_CHECKPOINTS, download_checkpoint
    from spatialarena.methods import resolve_method

    spec = resolve_method("COME")
    method_code_root = tmp_path / "code"
    relative = SHIPPED_WITH_CODE_CHECKPOINTS["COME"]
    source = method_code_root / "CoMe" / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fake-checkpoint-bytes")

    shared_checkpoint_root = tmp_path / "ckpt"
    result = download_checkpoint(
        spec,
        shared_checkpoint_root=shared_checkpoint_root,
        shared_hf_home=tmp_path / "hf",
        method_code_root=method_code_root,
    )
    expected_dest = shared_checkpoint_root / "local" / "come" / "checkpoint.pth"
    assert result == {
        "name": "COME",
        "status": "ok",
        "kind": "shipped_with_code",
        "path": str(expected_dest)
    }
    assert expected_dest.read_bytes() == b"fake-checkpoint-bytes"


def test_download_checkpoint_shipped_with_code_skips_when_source_missing(
        tmp_path):
    from spatialarena.download import download_checkpoint
    from spatialarena.methods import resolve_method

    spec = resolve_method("COME")
    result = download_checkpoint(
        spec,
        shared_checkpoint_root=tmp_path / "ckpt",
        shared_hf_home=tmp_path / "hf",
        method_code_root=tmp_path / "code",
    )
    assert result["status"] == "skipped"
    assert result["kind"] == "shipped_with_code"


def test_download_checkpoint_hf_file_reports_gated_access_not_a_crash(tmp_path):
    from unittest.mock import Mock

    from huggingface_hub.utils import HfHubHTTPError

    from spatialarena.download import download_checkpoint
    from spatialarena.methods import resolve_method

    spec = resolve_method("IGGT4D")
    method_code_root = tmp_path / "code"
    with patch("huggingface_hub.hf_hub_download",
               side_effect=HfHubHTTPError("gated", response=Mock())):
        result = download_checkpoint(
            spec,
            shared_checkpoint_root=tmp_path / "ckpt",
            shared_hf_home=tmp_path / "hf",
            method_code_root=method_code_root,
        )
    assert result["status"] == "skipped"
    assert result["kind"] == "hf_local_files"
    assert "HorizonRobotics/IGGT4D" in result["reason"]


def test_download_all_supported_models_filters_by_name(tmp_path):
    from spatialarena.download import download_all_supported_models

    with patch("huggingface_hub.snapshot_download", return_value="/fake/path"):
        results = download_all_supported_models(
            methods=["PI3_X", "COME"],
            shared_checkpoint_root=tmp_path / "ckpt",
            shared_hf_home=tmp_path / "hf",
            # An isolated, empty method_code_root -- COME's "ships with code"
            # checkpoint has no CoMe clone to copy from here, so it must
            # report "skipped" rather than reaching out to whatever this
            # session's real shared _model_codes happens to contain.
            method_code_root=tmp_path / "code",
        )
    assert {r["name"] for r in results} == {"PI3_X", "COME"}
    assert next(r for r in results if r["name"] == "PI3_X")["status"] == "ok"
    assert next(
        r for r in results if r["name"] == "COME")["status"] == "skipped"


def test_code_leaf_dir_handles_nested_and_shared_paths():
    from spatialarena.download import _code_leaf_dir

    assert _code_leaf_dir("$METHOD_CODE_ROOT/amb3r/thirdparty") == "amb3r"
    assert _code_leaf_dir("$METHOD_CODE_ROOT/RobustVGGT") == "RobustVGGT"


def test_list_supported_method_code_matches_registry_size():
    from spatialarena.download import list_supported_method_code
    from spatialarena.methods import METHOD_SPECS

    entries = list_supported_method_code()
    assert len(entries) == len(METHOD_SPECS)
    for entry in entries:
        if entry["status"] == "automated":
            assert entry["repo_url"], entry
        else:
            assert entry["notes"], entry


def test_list_supported_method_code_never_claims_automated_without_a_verified_repo(
):
    from spatialarena.download import METHOD_CODE_REPOS, list_supported_method_code

    for entry in list_supported_method_code():
        if entry["status"] == "automated":
            assert entry["repo_url"] == METHOD_CODE_REPOS[entry["leaf_dir"]]


def test_download_method_code_skips_leaves_with_no_verified_repo(
        tmp_path, monkeypatch):
    from spatialarena import download as download_module
    from spatialarena.methods import resolve_method

    # Force a miss regardless of which leaves currently have verified repos,
    # so this test doesn't depend on METHOD_CODE_REPOS' exact contents.
    monkeypatch.setattr(download_module, "METHOD_CODE_REPOS", {})
    spec = resolve_method("PI3_X")
    with patch("spatialarena.download.subprocess.run") as mock_run:
        result = download_module.download_method_code(spec,
                                                      method_code_root=tmp_path)
    mock_run.assert_not_called()
    assert result["status"] == "skipped"


def test_download_method_code_skips_existing_directory(tmp_path):
    from spatialarena.download import download_method_code
    from spatialarena.methods import resolve_method

    spec = resolve_method("PI3_X")
    (tmp_path / "Pi3").mkdir()
    with patch("spatialarena.download.subprocess.run") as mock_run:
        result = download_method_code(spec,
                                      method_code_root=tmp_path,
                                      skip_existing=True)
    mock_run.assert_not_called()
    assert result == {
        "name": "PI3_X",
        "status": "already_present",
        "path": str(tmp_path / "Pi3")
    }


def test_download_method_code_clones_via_git(tmp_path):
    from spatialarena.download import download_method_code
    from spatialarena.methods import resolve_method

    spec = resolve_method("PI3_X")
    with patch("spatialarena.download.subprocess.run",
               return_value=_mock_run()) as mock_run:
        result = download_method_code(spec, method_code_root=tmp_path)
    assert result["status"] == "ok"
    argv = mock_run.call_args.args[0]
    assert argv[:2] == ["git", "clone"]
    assert argv[2] == "https://github.com/yyfz/Pi3.git"
    assert argv[3] == str(tmp_path / "Pi3")


def test_download_all_supported_method_code_shares_one_clone_across_methods(
        tmp_path):
    from spatialarena.download import download_all_supported_method_code

    def fake_run(argv, **kwargs):  # pylint: disable=unused-argument
        Path(argv[3]).mkdir(
            parents=True)  # simulate `git clone` creating the dest dir
        return _mock_run()

    with patch("spatialarena.download.subprocess.run",
               side_effect=fake_run) as mock_run:
        results = download_all_supported_method_code(
            methods=["VGGT", "FASTVGGT", "ROBUSTVGGT"],
            method_code_root=tmp_path,
        )
    assert {r["name"] for r in results} == {"VGGT", "FASTVGGT", "ROBUSTVGGT"}
    assert sum(1 for r in results if r["status"] == "ok") == 1
    assert sum(1 for r in results if r["status"] == "already_present") == 2
    assert mock_run.call_count == 1

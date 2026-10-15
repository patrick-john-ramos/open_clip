"""Allowlist-based WebDataset sample filtering tests."""
import io
import json
import os
import tarfile
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from open_clip.model_traits import CLIP_TRAITS
from open_clip_train.data import FilterValidSample, get_wds_dataset, load_allowlist, _resolve_train_allowlist, _shard_basename
from open_clip_train.params import parse_args
from torchvision import transforms

# ---- Helper to create tar files ----
def _write_tar(tar_path, num_samples, key_prefix="", captions=None):
    if captions is None:
        captions = [f"cap_{i}".encode() for i in range(num_samples)]
    with tarfile.open(tar_path, "w") as tar:
        for i in range(num_samples):
            key = f"{key_prefix}{i}"
            img = Image.new("RGB", (32, 32))
            img_bio = io.BytesIO()
            img.save(img_bio, format="png")
            info = tarfile.TarInfo(f"{key}.png")
            info.size = len(img_bio.getvalue())
            img_bio.seek(0)
            tar.addfile(info, img_bio)
            
            txt_bio = io.BytesIO()
            txt_bio.write(captions[i] if i < len(captions) else f"cap_{i}".encode())
            info = tarfile.TarInfo(f"{key}.txt")
            info.size = len(txt_bio.getvalue())
            txt_bio.seek(0)
            tar.addfile(info, txt_bio)


def _build_args(input_shards, train_num_samples=100, seed=0, dataset_resampled=False, no_allowlist=True):
    """Build args namespace like test_wds.build_params but with allowlist options."""
    args = parse_args([])
    args.train_data = input_shards
    args.train_num_samples = train_num_samples
    args.dataset_resampled = dataset_resampled
    args.seed = seed
    args.workers = 0
    args.world_size = 1
    args.batch_size = 1
    args.no_allowlist = no_allowlist
    return args


# ---- Tests for _shard_basename ----
def test_shard_basename_local_path():
    assert _shard_basename("/data/shards/dreamlip-cc3m-train-000000.tar") == "dreamlip-cc3m-train-000000.tar"
    assert _shard_basename("data/shards/file.tar") == "file.tar"
    assert _shard_basename("file.tar") == "file.tar"


def test_shard_basename_url():
    assert _shard_basename("https://host/path/to/shard.tar?query=1#frag") == "shard.tar"
    assert _shard_basename("s3://bucket/data/shard.tar") == "shard.tar"
    assert _shard_basename("file:///data/shards/file.tar") == "file.tar"


# ---- Tests for load_allowlist ----
def test_load_allowlist_valid(tmp_path):
    path = tmp_path / "allow.npz"
    ids = np.array(["dreamlip-cc3m-train-000000.tar/000000000", "dreamlip-cc12m-train-000000.tar/000000001"])
    np.savez(path, ids=ids)
    allowlist = load_allowlist(str(path))
    assert allowlist == {"dreamlip-cc3m-train-000000.tar/000000000", "dreamlip-cc12m-train-000000.tar/000000001"}


def test_load_allowlist_deduplicates(tmp_path):
    path = tmp_path / "allow.npz"
    ids = np.array(["a.tar/0", "a.tar/0", "b.tar/1"])
    np.savez(path, ids=ids)
    allowlist = load_allowlist(str(path))
    assert allowlist == {"a.tar/0", "b.tar/1"}


def test_load_allowlist_empty_rejected(tmp_path):
    path = tmp_path / "allow.npz"
    np.savez(path, ids=np.array([], dtype=str))
    with pytest.raises(ValueError, match="no valid identifiers"):
        load_allowlist(str(path))


def test_load_allowlist_non_string_rejected(tmp_path):
    path = tmp_path / "allow.npz"
    np.savez(path, ids=np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="must contain strings"):
        load_allowlist(str(path))


def test_load_allowlist_multidim_rejected(tmp_path):
    path = tmp_path / "allow.npz"
    np.savez(path, ids=np.array([["a.tar/0"], ["b.tar/1"]]))
    with pytest.raises(ValueError, match="one-dimensional"):
        load_allowlist(str(path))


def test_load_allowlist_missing_ids_key_rejected(tmp_path):
    path = tmp_path / "allow.npz"
    np.savez(path, other=np.array(["a.tar/0"]))
    with pytest.raises(ValueError, match="missing the expected 'ids'"):
        load_allowlist(str(path))


def test_load_allowlist_missing_file_rejected():
    with pytest.raises(FileNotFoundError):
        load_allowlist("/nonexistent/allow.npz")


def test_load_allowlist_disabled_when_path_none():
    assert load_allowlist(None) is None


# ---- Tests for _resolve_train_allowlist ----
def test_resolve_train_allowlist_no_allowlist():
    args = SimpleNamespace(no_allowlist=True, allowlist=None)
    assert _resolve_train_allowlist(args) is None


def test_resolve_train_allowlist_mutually_exclusive():
    args = SimpleNamespace(no_allowlist=True, allowlist="/path.npz")
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_train_allowlist(args)


def test_resolve_train_allowlist_neither_supplied():
    args = SimpleNamespace(no_allowlist=False, allowlist=None)
    with pytest.raises(ValueError, match="not configured"):
        _resolve_train_allowlist(args)


def test_resolve_train_allowlist_empty_path():
    args = SimpleNamespace(no_allowlist=False, allowlist="")
    with pytest.raises(ValueError, match="nonempty path"):
        _resolve_train_allowlist(args)


# ---- Tests for FilterValidSample allowlist filter ----
def test_filter_valid_sample_allowlist_exact_match():
    f = FilterValidSample(allowlist={"shard.tar/0"})
    sample = {"__url__": "/data/shard.tar", "__key__": "0", "jpg": b"", "txt": b"hi"}
    assert f(sample) is True


def test_filter_valid_sample_allowlist_different_shard_rejected():
    f = FilterValidSample(allowlist={"shard.tar/0"})
    sample = {"__url__": "/data/other.tar", "__key__": "0", "jpg": b"", "txt": b"hi"}
    assert f(sample) is False


def test_filter_valid_sample_allowlist_different_key_rejected():
    f = FilterValidSample(allowlist={"shard.tar/0"})
    sample = {"__url__": "/data/shard.tar", "__key__": "1", "jpg": b"", "txt": b"hi"}
    assert f(sample) is False


def test_filter_valid_sample_allowlist_url_query_fragment_ignored():
    f = FilterValidSample(allowlist={"shard.tar/0"})
    sample = {"__url__": "https://host/path/shard.tar?x=1#frag", "__key__": "0", "jpg": b"", "txt": b"hi"}
    assert f(sample) is True


def test_filter_valid_sample_allowlist_preserves_tar_extension():
    f = FilterValidSample(allowlist={"dreamlip-cc3m-train-000000.tar/000000000"})
    sample = {"__url__": "/data/dreamlip-cc3m-train-000000.tar", "__key__": "000000000", "jpg": b"", "txt": b"x"}
    assert f(sample) is True


def test_filter_valid_sample_no_allowlist_preserves_behavior():
    f = FilterValidSample(allowlist=None)
    # no image -> False
    assert f({"__url__": "/data/shard.tar", "__key__": "0", "txt": b"hi"}) is False
    # image + txt -> True
    assert f({"__url__": "/data/shard.tar", "__key__": "0", "jpg": b"", "txt": b"hi"}) is True


def test_filter_valid_sample_allowlist_still_runs_validity_checks():
    f = FilterValidSample(allowlist={"shard.tar/0"})
    # allowlisted but missing caption -> False
    assert f({"__url__": "/data/shard.tar", "__key__": "0", "jpg": b""}) is False


# ---- Integration tests with get_wds_dataset ----
def test_get_wds_dataset_allowlist_filters_samples(tmp_path):
    # Create two shards with unique captions
    shard0 = tmp_path / "shard0.tar"
    shard1 = tmp_path / "shard1.tar"
    _write_tar(shard0, 5, captions=[b"a_0", b"a_1", b"a_2", b"a_3", b"a_4"])
    _write_tar(shard1, 5, captions=[b"b_0", b"b_1", b"b_2", b"b_3", b"b_4"])
    
    # Allowlist: shard0 keys 0 and 2, shard1 key 3
    allowlist_path = tmp_path / "allow.npz"
    np.savez(allowlist_path, ids=np.array([f"{shard0.name}/0", f"{shard0.name}/2", f"{shard1.name}/3"]))
    
    args = _build_args(f"{shard0}::{shard1}", train_num_samples=3, no_allowlist=False)
    args.allowlist = str(allowlist_path)
    
    dataset = get_wds_dataset(args, transforms.ToTensor(), is_train=True, tokenizer=lambda x: [x.strip()], model_traits=CLIP_TRAITS)
    captions = [batch["text"][0] for batch in dataset.dataloader]
    assert set(captions) == {"a_0", "a_2", "b_3"}


def test_get_wds_dataset_no_allowlist_preserves_all(tmp_path):
    shard0 = tmp_path / "shard0.tar"
    shard1 = tmp_path / "shard1.tar"
    _write_tar(shard0, 5, captions=[b"a_0", b"a_1", b"a_2", b"a_3", b"a_4"])
    _write_tar(shard1, 5, captions=[b"b_0", b"b_1", b"b_2", b"b_3", b"b_4"])
    
    args = _build_args(f"{shard0}::{shard1}", train_num_samples=10, no_allowlist=True)
    
    dataset = get_wds_dataset(args, transforms.ToTensor(), is_train=True, tokenizer=lambda x: [x.strip()], model_traits=CLIP_TRAITS)
    captions = [batch["text"][0] for batch in dataset.dataloader]
    assert set(captions) == {"a_0", "a_1", "a_2", "a_3", "a_4", "b_0", "b_1", "b_2", "b_3", "b_4"}


def test_get_wds_dataset_allowlist_resampled_rejected(tmp_path):
    shard0 = tmp_path / "shard0.tar"
    _write_tar(shard0, 5)
    allowlist_path = tmp_path / "allow.npz"
    np.savez(allowlist_path, ids=np.array([f"{shard0.name}/0"]))
    
    args = _build_args(str(shard0), train_num_samples=1, dataset_resampled=True, no_allowlist=False)
    args.allowlist = str(allowlist_path)
    
    with pytest.raises(ValueError, match="not supported with --dataset-resampled"):
        get_wds_dataset(args, transforms.ToTensor(), is_train=True, tokenizer=lambda x: [x.strip()], model_traits=CLIP_TRAITS)


def test_get_wds_dataset_allowlist_json_pipeline(tmp_path):
    # Build tar with .json members
    shard = tmp_path / "shard.tar"
    with tarfile.open(shard, "w") as tar:
        for i in range(4):
            json_bytes = json.dumps({"caption": f"cap_{i}"}).encode()
            json_bio = io.BytesIO(json_bytes)
            info = tarfile.TarInfo(f"{i}.json")
            info.size = len(json_bytes)
            json_bio.seek(0)
            tar.addfile(info, json_bio)
            img = Image.new("RGB", (8,8))
            img_bio = io.BytesIO()
            img.save(img_bio, format="png")
            info = tarfile.TarInfo(f"{i}.png")
            info.size = len(img_bio.getvalue())
            img_bio.seek(0)
            tar.addfile(info, img_bio)
    
    allowlist_path = tmp_path / "allow.npz"
    np.savez(allowlist_path, ids=np.array([f"{shard.name}/0", f"{shard.name}/2"]))
    
    args = parse_args([])
    args.train_data = str(shard)
    args.train_num_samples = 2
    args.dataset_resampled = False
    args.allowlist = str(allowlist_path)
    args.no_allowlist = False
    args.json_text_key = "caption"
    args.seed = 0
    args.workers = 0
    args.world_size = 1
    args.batch_size = 1
    
    dataset = get_wds_dataset(args, transforms.ToTensor(), is_train=True, tokenizer=lambda x: [x.strip()], model_traits=CLIP_TRAITS)
    texts = [batch["text"][0] for batch in dataset.dataloader]
    assert set(texts) == {"cap_0", "cap_2"}


# ---- CLI argument validation ----
def test_parse_args_rejects_both_allowlist_and_no_allowlist():
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_args(["--allowlist", "/path.npz", "--no-allowlist"])


def test_parse_args_rejects_empty_allowlist():
    with pytest.raises(ValueError, match="nonempty path"):
        parse_args(["--allowlist", ""])


def test_parse_args_accepts_allowlist():
    args = parse_args(["--allowlist", "/path.npz"])
    assert args.allowlist == "/path.npz"
    assert args.no_allowlist is False


def test_parse_args_accepts_no_allowlist():
    args = parse_args(["--no-allowlist"])
    assert args.no_allowlist is True
    assert args.allowlist is None

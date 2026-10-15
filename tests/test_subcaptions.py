"""DreamLIP sub-caption sampling tests."""
import io
import os
import random
import tarfile
import types
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

import open_clip
from open_clip import create_task
from open_clip.model_traits import CLIP_TRAITS
from open_clip.task import CLIPTask
from open_clip_train.data import TokenizeText, get_wds_dataset
from open_clip_train.params import parse_args
from open_clip_train.subcaptions import (
    SampleSubcaptions,
    build_subcaption_pool,
    draw_subcaptions,
)
from util_test import create_tiny_model


# ---- Helper to create tar files with DreamLIP schema ----
def _write_subcaption_tar(path, num_samples, key_prefix="", captions=None, long_captions=None, short_captions=None, raw_captions=None):
    if captions is None:
        captions = [f"cap_{i}".encode() for i in range(num_samples)]
    if long_captions is None:
        long_captions = [f"long caption for sample {i}. It has multiple sentences. Another one here.".encode() for i in range(num_samples)]
    if short_captions is None:
        short_captions = [f"short cap {i}".encode() for i in range(num_samples)]
    if raw_captions is None:
        raw_captions = [f"raw_caption_{i}".encode() for i in range(num_samples)]
    
    with tarfile.open(path, "w") as tar:
        for i in range(num_samples):
            key = f"{key_prefix}{i}"
            img = Image.new("RGB", (32, 32))
            img_bio = io.BytesIO()
            img.save(img_bio, format="png")
            info = tarfile.TarInfo(f"{key}.jpg")
            info.size = len(img_bio.getvalue())
            img_bio.seek(0)
            tar.addfile(info, img_bio)
            
            # raw_caption
            info = tarfile.TarInfo(f"{key}.raw_caption")
            info.size = len(raw_captions[i])
            tar.addfile(info, io.BytesIO(raw_captions[i]))
            
            # shortsv_captions
            info = tarfile.TarInfo(f"{key}.shortsv_captions")
            info.size = len(short_captions[i])
            tar.addfile(info, io.BytesIO(short_captions[i]))
            
            # longsv_captions
            info = tarfile.TarInfo(f"{key}.longsv_captions")
            info.size = len(long_captions[i])
            tar.addfile(info, io.BytesIO(long_captions[i]))
            
            # url
            info = tarfile.TarInfo(f"{key}.url")
            info.size = len(f"http://example.com/{key}".encode())
            tar.addfile(info, io.BytesIO(f"http://example.com/{key}".encode()))


def _build_args(input_shards, train_num_samples=100, seed=0, dataset_resampled=False, num_subcaptions=0,
                subcaption_long_key="longsv_captions", subcaption_short_key="shortsv_captions",
                no_allowlist=True):
    args = parse_args([])
    args.train_data = input_shards
    args.train_num_samples = train_num_samples
    args.dataset_resampled = dataset_resampled
    args.seed = seed
    args.workers = 0
    args.world_size = 1
    args.batch_size = 1
    args.no_allowlist = no_allowlist
    args.num_subcaptions = num_subcaptions
    args.subcaption_long_key = subcaption_long_key
    args.subcaption_short_key = subcaption_short_key
    return args


# ---- Tests for _shard_basename and subcaptions module ----
def test_build_subcaption_pool_basic():
    sample = {
        "raw_caption": b"Copenhagen April %2713",
        "shortsv_captions": b"short caption",
        "longsv_captions": b"First sentence. Second sentence. Third one here.",
    }
    pool = build_subcaption_pool(sample)
    assert "Copenhagen April '13" in pool  # %27 decoded to '
    assert "short caption" in pool
    assert "First sentence." in pool
    assert "Second sentence." in pool
    assert "Third one here." in pool
    # Empty raw/short should be dropped
    sample_empty = {"raw_caption": b"", "shortsv_captions": b"", "longsv_captions": b"Only sentence."}
    pool = build_subcaption_pool(sample_empty)
    assert pool == ["Only sentence."]


def test_draw_subcaptions_exact_k():
    pool = ["a", "b", "c", "d", "e"]
    result = draw_subcaptions(pool, 3)
    assert len(result) == 3
    assert all(isinstance(x, str) for x in result)
    assert set(result).issubset(set(pool))


def test_draw_subcaptions_decodes_url_escaped_raw():
    # Test that build_subcaption_pool decodes URL-escaped raw captions
    sample = {
        "raw_caption": b"Copenhagen April %2713",
        "shortsv_captions": b"short",
        "longsv_captions": b"long caption.",
    }
    pool = build_subcaption_pool(sample)
    assert "Copenhagen April '13" in pool


def test_draw_subcaptions_from_pool():
    pool = ["a", "b", "c"]
    result = draw_subcaptions(pool, 3)
    assert set(result) == set(pool)


def test_draw_subcaptions_small_pool_returns_k():
    pool = ["a", "b"]
    result = draw_subcaptions(pool, 5)
    assert len(result) == 5
    assert set(result).issuperset(set(pool))


def test_draw_subcaptions_reproducible_with_seed():
    pool = ["a", "b", "c", "d", "e"]
    random.seed(42)
    r1 = draw_subcaptions(pool, 3)
    random.seed(42)
    r2 = draw_subcaptions(pool, 3)
    assert r1 == r2


def test_draw_subcaptions_two_calls_differ():
    pool = ["a", "b", "c", "d", "e"]
    r1 = draw_subcaptions(pool, 3)
    r2 = draw_subcaptions(pool, 3)
    # Very unlikely to be identical (1/10 chance)
    assert r1 != r2 or len(set(r1)) != 3  # Allow small chance of collision


def test_sample_subcaptions_empty_pool_returns_none():
    f = SampleSubcaptions(k=2)
    sample = {"raw_caption": b"", "shortsv_captions": b"", "longsv_captions": b"", "__url__": "/x.tar", "__key__": "0"}
    assert f(sample) is None


# ---- Tests for TokenizeText ----
def test_tokenize_text_list_returns_k_by_context():
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokenize_text = TokenizeText(tokenizer)
    texts = ["caption one", "caption two", "caption three"]
    tokens = tokenize_text(texts)
    assert tokens.shape[0] == 3
    assert tokens.shape[1] == tokenizer.context_length


def test_tokenize_text_single_string_returns_context():
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokenize_text = TokenizeText(tokenizer)
    tokens = tokenize_text("single caption")
    assert tokens.shape == (tokenizer.context_length,)


# ---- Integration tests with get_wds_dataset ----
@pytest.mark.parametrize("num_subcaptions", [1, 2, 4])
def test_pipeline_yields_samples(num_subcaptions, tmp_path):
    # Set small shuffle sizes for deterministic fast tests
    for name in ("OPENCLIP_WDS_SHARD_SHUFFLE_SIZE", "OPENCLIP_WDS_SHARD_SHUFFLE_INITIAL",
                 "OPENCLIP_WDS_SAMPLE_SHUFFLE_SIZE", "OPENCLIP_WDS_SAMPLE_SHUFFLE_INITIAL"):
        os.environ[name] = "4"
    
    shard = tmp_path / "shard.tar"
    _write_subcaption_tar(shard, 8)
    
    allowlist_path = tmp_path / "allow.npz"
    import numpy as np
    np.savez(allowlist_path, ids=np.array([f"shard.tar/{i}" for i in range(4)]))
    
    args = _build_args(str(shard), train_num_samples=4, num_subcaptions=num_subcaptions, no_allowlist=False)
    args.allowlist = str(allowlist_path)
    
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    dataset = get_wds_dataset(args, preprocess, is_train=True, tokenizer=tokenizer, model_traits=CLIP_TRAITS)
    
    batch = next(iter(dataset.dataloader))
    assert batch["image"].shape == (1, 3, 224, 224)
    assert batch["text"].shape == (1, num_subcaptions, tokenizer.context_length)
    assert batch["text"].dtype == torch.long


def test_pipeline_iterating_same_shard_twice_gives_different_subcaptions(tmp_path):
    for name in ("OPENCLIP_WDS_SHARD_SHUFFLE_SIZE", "OPENCLIP_WDS_SHARD_SHUFFLE_INITIAL",
                 "OPENCLIP_WDS_SAMPLE_SHUFFLE_SIZE", "OPENCLIP_WDS_SAMPLE_SHUFFLE_INITIAL"):
        os.environ[name] = "4"
    
    shard = tmp_path / "shard.tar"
    _write_subcaption_tar(shard, 8)
    
    allowlist_path = tmp_path / "allow.npz"
    import numpy as np
    np.savez(allowlist_path, ids=np.array([f"shard.tar/{i}" for i in range(4)]))
    
    args = _build_args(str(shard), train_num_samples=4, num_subcaptions=2, no_allowlist=False)
    args.allowlist = str(allowlist_path)
    
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    dataset = get_wds_dataset(args, preprocess, is_train=True, tokenizer=tokenizer, model_traits=CLIP_TRAITS)
    
    texts1 = []
    texts2 = []
    for i, batch in enumerate(dataset.dataloader):
        texts1.append(batch["text"].clone())
        if i == 1:
            break
    # Create new dataset with same args (different seed)
    args.seed = 1
    dataset2 = get_wds_dataset(args, transforms.ToTensor(), is_train=True, tokenizer=tokenizer, model_traits=CLIP_TRAITS)
    for i, batch in enumerate(dataset2.dataloader):
        texts2.append(batch["text"].clone())
        if i == 1:
            break
    
    # Texts should differ due to different random sampling
    assert not torch.equal(texts1[0], texts2[0])


def test_pipeline_workers_do_not_produce_identical_sequences(tmp_path):
    for name in ("OPENCLIP_WDS_SHARD_SHUFFLE_SIZE", "OPENCLIP_WDS_SHARD_SHUFFLE_INITIAL",
                 "OPENCLIP_WDS_SAMPLE_SHUFFLE_SIZE", "OPENCLIP_WDS_SAMPLE_SHUFFLE_INITIAL"):
        os.environ[name] = "4"
    
    shard = tmp_path / "shard.tar"
    shard2 = tmp_path / "shard2.tar"
    _write_subcaption_tar(shard, 4)
    _write_subcaption_tar(shard2, 4)
    
    allowlist_path = tmp_path / "allow.npz"
    import numpy as np
    np.savez(allowlist_path, ids=np.array([f"shard.tar/{i}" for i in range(4)] + [f"shard2.tar/{i}" for i in range(4)]))
    
    args = _build_args(f"{shard}::{shard2}", train_num_samples=4, num_subcaptions=2, no_allowlist=False)
    args.allowlist = str(allowlist_path)
    args.workers = 2
    
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    dataset = get_wds_dataset(args, preprocess, is_train=True, tokenizer=tokenizer, model_traits=CLIP_TRAITS)
    
    texts = [batch["text"].clone() for _, batch in zip(range(4), dataset.dataloader)]
    # With multiple workers, sequences should differ
    all_equal = all(torch.equal(texts[0], t) for t in texts)
    assert not all_equal


def test_pipeline_picklable(tmp_path):
    import pickle
    
    for name in ("OPENCLIP_WDS_SHARD_SHUFFLE_SIZE", "OPENCLIP_WDS_SHARD_SHUFFLE_INITIAL",
                 "OPENCLIP_WDS_SAMPLE_SHUFFLE_SIZE", "OPENCLIP_WDS_SAMPLE_SHUFFLE_INITIAL"):
        os.environ[name] = "4"
    
    shard = tmp_path / "shard.tar"
    _write_subcaption_tar(shard, 8)
    
    allowlist_path = tmp_path / "allow.npz"
    import numpy as np
    np.savez(allowlist_path, ids=np.array([f"shard.tar/{i}" for i in range(4)]))
    
    args = _build_args(str(shard), train_num_samples=4, num_subcaptions=2, no_allowlist=False)
    args.allowlist = str(allowlist_path)
    
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    dataset = get_wds_dataset(args, preprocess, is_train=True, tokenizer=tokenizer, model_traits=CLIP_TRAITS)
    
    pickled = pickle.dumps(dataset)
    unpickled = pickle.loads(pickled)
    batch = next(iter(unpickled.dataloader))
    assert batch["image"].shape == (1, 3, 224, 224)
    assert batch["text"].shape == (1, 2, 77)


def test_pipeline_num_subcaptions_zero_is_old_pipeline(tmp_path):
    for name in ("OPENCLIP_WDS_SHARD_SHUFFLE_SIZE", "OPENCLIP_WDS_SHARD_SHUFFLE_INITIAL",
                 "OPENCLIP_WDS_SAMPLE_SHUFFLE_SIZE", "OPENCLIP_WDS_SAMPLE_SHUFFLE_INITIAL"):
        os.environ[name] = "4"
    
    shard = tmp_path / "shard.tar"
    _write_subcaption_tar(shard, 8)
    
    args = _build_args(str(shard), train_num_samples=4, num_subcaptions=0, no_allowlist=True)
    
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    dataset = get_wds_dataset(args, preprocess, is_train=True, tokenizer=tokenizer, model_traits=CLIP_TRAITS)
    
    batch = next(iter(dataset.dataloader))
    assert batch["text"].shape == (1, tokenizer.context_length)


# ---- Task equivalence tests ----
def test_task_equivalence_k1():
    torch.manual_seed(0)
    model = create_tiny_model("clip")
    
    task_standard = CLIPTask(model)
    task_sub = CLIPTask(model, num_subcaptions=1)
    
    B, L = 4, 8
    image = torch.randn(B, 3, 32, 32)
    # Use vocab size of tiny model (64)
    text = torch.randint(0, 64, (B, 8))
    
    batch_standard = {"image": image, "text": text}
    batch_sub = {"image": image, "text": text.unsqueeze(1)}  # (B, 1, L)
    
    losses_std, _ = task_standard.training_forward(batch_standard)
    losses_sub, _ = task_sub.training_forward(batch_sub)
    
    assert torch.allclose(losses_std["contrastive_loss"], losses_sub["contrastive_loss"], rtol=1e-4)


def test_task_equivalence_k_repeated_text():
    torch.manual_seed(0)
    model = create_tiny_model("clip")
    
    task_standard = CLIPTask(model)
    task_sub = CLIPTask(model, num_subcaptions=3)
    
    B, L = 4, 8
    image = torch.randn(B, 3, 32, 32)
    text = torch.randint(0, 64, (B, 8))
    
    batch_standard = {"image": image, "text": text}
    batch_sub = {"image": image, "text": text.unsqueeze(1).expand(B, 3, 8)}
    
    losses_std, _ = task_standard.training_forward(batch_standard)
    losses_sub, _ = task_sub.training_forward(batch_sub)
    
    # When all K sub-captions are identical, the loss should be the same
    assert torch.allclose(losses_std["contrastive_loss"], losses_sub["contrastive_loss"], rtol=1e-4)


def test_task_gradients_reach_both_towers():
    torch.manual_seed(0)
    model = create_tiny_model("clip")
    task = CLIPTask(model, num_subcaptions=2)
    
    B, L = 4, 8
    image = torch.randn(B, 3, 32, 32, requires_grad=True)
    # Use vocab size of tiny model (64)
    text = torch.randint(0, 64, (B, 2, 8))
    
    batch = {"image": image, "text": text}
    losses, _ = task.training_forward(batch)
    losses["loss"].backward()
    
    assert image.grad is not None
    # Check that text tower parameters have gradients
    text_params = [p for n, p in model.named_parameters() if "text" in n or "transformer" in n or "token_embedding" in n]
    assert any(p.grad is not None for p in text_params)


def test_task_compile_runs():
    torch.manual_seed(0)
    model = create_tiny_model("clip")
    task = CLIPTask(model, num_subcaptions=2)
    task.compile(target="task", backend="eager")
    
    B, L = 4, 8
    image = torch.randn(B, 3, 32, 32)
    # Use vocab size of tiny model (64)
    text = torch.randint(0, 64, (B, 2, 8))
    batch = {"image": image, "text": text}
    
    losses_eager, _ = task.training_forward(batch)
    losses_compiled, _ = task.training_forward(batch)
    
    assert torch.allclose(losses_eager["contrastive_loss"], losses_compiled["contrastive_loss"], rtol=1e-4)


# ---- Dispatch and validation tests ----
def test_create_task_returns_clip_task_with_num_subcaptions():
    from open_clip.factory import create_task
    model = create_tiny_model("clip")
    args = parse_args([
        "--model", "RN50", "--num-subcaptions", "4",
        "--local-loss", "--gather-with-grad",
        "--no-allowlist",
    ])
    # Add missing attributes that create_task expects
    args.distill = False
    args.distill_model = None
    args.distill_pretrained = None
    args.use_naflex = False
    args.json_text_key = None
    args.accum_freq = 1
    args.rank = 0
    args.world_size = 1
    task = create_task(args, model)
    assert isinstance(task, CLIPTask)
    assert task.num_subcaptions == 4


def test_create_task_rejects_siglip():
    with pytest.raises(ValueError, match="--num-subcaptions is not supported with --siglip"):
        parse_args(["--model", "RN50", "--num-subcaptions", "4", "--siglip"])


def test_create_task_rejects_distill():
    with pytest.raises(ValueError, match="--num-subcaptions is not supported with distillation"):
        parse_args(["--model", "RN50", "--num-subcaptions", "4", "--distill-model", "RN50", "--distill-pretrained", "openai/clip-vit-base-patch32"])


def test_create_task_rejects_accum_freq():
    with pytest.raises(ValueError, match="accum-freq > 1"):
        parse_args(["--model", "RN50", "--num-subcaptions", "4", "--accum-freq", "2"])


def test_create_task_rejects_naflex():
    with pytest.raises(ValueError, match="--num-subcaptions is not supported with --use-naflex"):
        parse_args(["--model", "RN50", "--num-subcaptions", "4", "--use-naflex"])


def test_create_task_rejects_json_text_key():
    with pytest.raises(ValueError, match="--num-subcaptions is not supported with --json-text-key"):
        parse_args(["--model", "RN50", "--num-subcaptions", "4", "--json-text-key", "caption"])


# ---- CLI validation tests ----
def test_parse_args_rejects_negative_num_subcaptions():
    with pytest.raises(ValueError, match="num-subcaptions must be >= 0"):
        parse_args(["--num-subcaptions", "-1"])


def test_parse_args_accepts_zero_num_subcaptions():
    args = parse_args(["--num-subcaptions", "0"])
    assert args.num_subcaptions == 0


def test_parse_args_accepts_positive_num_subcaptions():
    args = parse_args(["--num-subcaptions", "8"])
    assert args.num_subcaptions == 8

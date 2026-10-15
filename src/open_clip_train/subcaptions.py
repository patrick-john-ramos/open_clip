"""DreamLIP sub-caption sampling (Zheng et al., 2024, arXiv:2403.17007, Sec. 3.3, Eq. 4).

Pure-Python, module-level callables so they survive pickling under forkserver multiprocessing.
"""
import random
import re
from urllib.parse import unquote

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _to_str(x):
    if x is None:
        return ''
    return x.decode('utf-8', errors='replace') if isinstance(x, (bytes, bytearray)) else str(x)


def build_subcaption_pool(sample, long_key='longsv_captions', short_key='shortsv_captions',
                          raw_key='raw_caption'):
    """Build the candidate text pool for one sample from its caption members.

    Pool = [raw, short] + sentences of the long caption, dropping empty entries.
    """
    raw = unquote(_to_str(sample.get(raw_key))).strip()
    short = _to_str(sample.get(short_key)).strip()
    long_cap = _to_str(sample.get(long_key)).strip()
    sents = [s.strip() for s in _SENT_SPLIT.split(long_cap) if s.strip()]
    return [t for t in [raw, short] + sents if t]


def draw_subcaptions(pool, k):
    """Eq. 4: k uniform draws from the pool.

    Without replacement when the pool is large enough; otherwise cover every pool item once and pad with
    replacement so exactly k items are always returned.
    """
    if len(pool) >= k:
        return random.sample(pool, k)
    return pool + random.choices(pool, k=k - len(pool))


class SampleSubcaptions:
    """wds sample-level map: sample["text"] = list of K sub-caption strings.

    Module-level + picklable (forkserver-safe). Uses the global ``random``, which the DataLoader seeds
    per worker. Returns None for samples with an empty pool; ``map_no_key`` drops those.
    """

    def __init__(self, k, long_key='longsv_captions', short_key='shortsv_captions', raw_key='raw_caption'):
        self.k, self.long_key, self.short_key, self.raw_key = k, long_key, short_key, raw_key

    def __call__(self, sample):
        pool = build_subcaption_pool(sample, self.long_key, self.short_key, self.raw_key)
        if not pool:
            return None
        sample["text"] = draw_subcaptions(pool, self.k)
        return sample
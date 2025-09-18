from __future__ import annotations
from typing import List, Tuple, Dict, Any
from pathlib import Path
import pickle
import re
import numpy as np
try:
    from rank_bm25 import BM25Okapi
except Exception as e:
    raise RuntimeError("Cần cài rank_bm25: pip install rank-bm25") from e

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _bm25_tokenize(text: str) -> List[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t]


def build_bm25_from_chunks(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    tokenized = [_bm25_tokenize(rec["text"]) for rec in chunks]
    return BM25Okapi(tokenized)


def save_bm25(bm25: BM25Okapi, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm25, f)


def load_bm25(path: Path) -> BM25Okapi:
    with open(path, "rb") as f:
        return pickle.load(f)


def bm25_search(bm25: BM25Okapi, query: str, top_k: int = 8) -> Tuple[List[int], List[float]]:
    q_tokens = _bm25_tokenize(query)
    if not q_tokens:
        return [], []

    scores = np.asarray(bm25.get_scores(q_tokens), dtype=np.float64)
    n = scores.shape[0]
    if n == 0:
        return [], []

    k = min(top_k, n)
    topk_part = np.argpartition(scores, n - k)[n - k:]
    topk_sorted = topk_part[np.argsort(scores[topk_part])[::-1]]
    idx = topk_sorted.tolist()
    return idx, scores[idx].astype(float).tolist()

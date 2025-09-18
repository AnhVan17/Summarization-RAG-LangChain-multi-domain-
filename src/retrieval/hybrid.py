from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def _min_max_scale(score_map: Dict[int, float]) -> Dict[int, float]:
    if not score_map:
        return {}
    vals = np.array(list(score_map.values()), dtype=np.float32)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax - vmin <= 1e-12:
        # tất cả bằng nhau → quy về 0.0 để nhánh kia quyết định
        return {k: 0.0 for k in score_map.keys()}
    scale = 1.0 / (vmax - vmin)
    return {k: (v - vmin) * scale for k, v in score_map.items()}


def fuse_dense_bm25(
    dense_scores: Dict[int, float],
    bm25_scores: Dict[int, float],
    alpha: float = 0.6,
) -> List[Tuple[int, float, float, float]]:
    """
    Trả list (idx, fused, dense_s, bm25_s) sắp xếp theo fused giảm dần.
    """
    sd = _min_max_scale(dense_scores)
    sb = _min_max_scale(bm25_scores)
    keys = set(sd.keys()) | set(sb.keys())
    fused: List[Tuple[int, float, float, float]] = []
    for i in keys:
        d = sd.get(i, 0.0)
        b = sb.get(i, 0.0)
        f = alpha * d + (1.0 - alpha) * b
        fused.append((i, f, d, b))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused

from __future__ import annotations
import numpy as np
from typing import List, Sequence, Tuple


def mmr_select(
    q_vec: np.ndarray,
    cand_vecs: np.ndarray,
    cand_ids: Sequence[int],
    k_out: int = 12,
    lam: float = 0.5,
) -> List[int]:
    """
    q_vec: (D,) đã L2-normalize
    cand_vecs: (N,D) đã L2-normalize
    cand_ids: len=N, id gốc (index trong chunks)
    Trả về: danh sách cand_ids đã chọn theo thứ tự MMR.
    """
    assert cand_vecs.ndim == 2 and q_vec.ndim == 1
    N = cand_vecs.shape[0]
    k_out = min(k_out, N)
    # sim query-doc
    s_q = cand_vecs @ q_vec  # (N,)
    # sim doc-doc
    s_dd = cand_vecs @ cand_vecs.T  # (N,N)

    selected: List[int] = []
    remaining = list(range(N))

    for _ in range(k_out):
        best_j = None
        best_score = -1e9
        for j in remaining:
            div = 0.0 if not selected else float(np.max(s_dd[j, selected]))
            score = lam * float(s_q[j]) - (1.0 - lam) * div
            if score > best_score:
                best_score, best_j = score, j
        # j: chỉ số nội bộ trong pool
        selected.append(best_j)
        remaining.remove(best_j)
    # map về cand_ids gốc
    return [cand_ids[j] for j in selected]

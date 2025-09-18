from __future__ import annotations
from typing import List, Dict, Any, Tuple


def _estimate_tokens(text: str) -> int:
    # xấp xỉ nhẹ
    return max(1, len(text.split()))


def aggregate_context(hits: List[Dict[str, Any]], use_compressed: bool = True, top_n: int = 8) -> str:
    text = []
    for h in hits[:top_n]:
        t = (h.get("compressed_text") if use_compressed else h.get("text")) or ""
        if t.strip():
            text.append(t.strip())
    return "\n".join(text).strip()


def guardrail_should_abstain(
    question: str,
    hits: List[Dict[str, Any]],
    enc,                        # DenseEncoder
    q_vec=None,
    min_hits: int = 1,
    min_tokens: int = 40,
    min_sim: float = 0.18
) -> Tuple[bool, str, float, int]:
    """
    Trả (abstain?, reason, sim, tokens)
    """
    if len(hits) < min_hits:
        return True, "no_hits", 0.0, 0
    ctx = aggregate_context(hits, use_compressed=True, top_n=min(8, len(hits)))
    toks = _estimate_tokens(ctx)
    if toks < min_tokens:
        return True, "too_short_context", 0.0, toks
    if q_vec is None:
        q_vec = enc.encode([question], normalize=True, batch_size=1)[0]
    c_vec = enc.encode([ctx], normalize=True, batch_size=1)[0]
    import numpy as np
    sim = float((q_vec @ c_vec))
    if sim < min_sim:
        return True, "low_similarity", sim, toks
    return False, "ok", sim, toks

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.chunk import sent_split

# cố gắng dùng estimator từ gen_models nếu có
try:
    from src.gen.gen_models import estimate_tokens as _tok_est
except Exception:
    def _tok_est(text: str) -> int:
        # fallback: xấp xỉ theo số từ
        return max(1, len(text.split()))


def _sentences_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Tách câu và lấy (start,end) theo vị trí trong text để highlight/citation."""
    sents = sent_split(text)
    spans: List[Tuple[int, int]] = []
    pos = 0
    for s in sents:
        i = text.find(s, pos)
        if i < 0:  # fallback nếu không khớp theo pos
            i = text.find(s)
        j = i + len(s)
        spans.append((i, j))
        pos = j
    return sents, spans


def _pick_by_target(tokens_per_sent: List[int], order: List[int], max_tokens: int, min_keep: int = 2) -> List[int]:
    """Chọn câu theo thứ tự 'order' cho tới khi tổng token <= max_tokens, tối thiểu min_keep câu."""
    picked: List[int] = []
    total = 0
    for idx in order:
        t = tokens_per_sent[idx]
        if total + t > max_tokens and len(picked) >= max_keep:
            break
        picked.append(idx)
        total += t
    if len(picked) < min_keep:
        picked = order[:min_keep]
    return sorted(picked)


def compress_chunk_query_aware(
    text: str,
    question: str,
    # DenseEncoder hoặc obj có .encode([...], normalize=True) -> np.ndarray
    enc,
    # nếu đã có embedding của query thì truyền vào
    q_vec: Optional[np.ndarray] = None,
    target_reduction: float = 0.4,       # giảm 40% tokens
    min_keep: int = 2,
    method: str = "cosine",              # "cosine" | "tfidf" (simple)
) -> Dict[str, Any]:
    """
    Trả về:
      {
        "compressed_text": str,
        "picked_idx": List[int],
        "sent_spans": List[(start,end)],
        "tokens_before": int,
        "tokens_after": int,
      }
    """
    assert 0.0 <= target_reduction < 1.0
    sents, spans = _sentences_with_spans(text)
    if not sents:
        tb = _tok_est(text)
        return {"compressed_text": text, "picked_idx": [], "sent_spans": [], "tokens_before": tb, "tokens_after": tb}

    tokens_per_sent = [_tok_est(s) for s in sents]
    tokens_before = sum(tokens_per_sent)
    max_tokens = max(1, int(tokens_before * (1.0 - target_reduction)))

    # --- chấm điểm câu ---
    if method == "cosine":
        # embed query
        if q_vec is None:
            q_vec = enc.encode([question], batch_size=1, normalize=True)[0]
        # embed sentences
        s_vecs = enc.encode(sents, batch_size=64, normalize=True)  # (n,D)
        scores = (s_vecs @ q_vec).tolist()
    else:
        # simple TF-IDF (query-aware): đếm token trùng giữa câu và query, có trọng số log(1+tf)
        import re
        import math
        WORD = re.compile(r"\w+", flags=re.UNICODE)
        q_tok = [w.lower() for w in WORD.findall(question)]
        q_set = set(q_tok)
        scores = []
        for s in sents:
            toks = [w.lower() for w in WORD.findall(s)]
            tf = sum(1 for t in toks if t in q_set)
            scores.append(math.log1p(tf))

    order = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
    picked = _pick_by_target(tokens_per_sent, order,
                             max_tokens=max_tokens, min_keep=min_keep)

    compressed_parts = [sents[i] for i in picked]
    tokens_after = sum(tokens_per_sent[i] for i in picked)

    return {
        "compressed_text": " ".join(compressed_parts).strip(),
        "picked_idx": picked,
        "sent_spans": [spans[i] for i in picked],
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
    }


def _safe_join(sents):
    return " ".join(s.strip() for s in sents if s.strip())


def compress_chunk_query_aware_with_sents(
    text: str, question: str, enc, q_vec=None, target_reduction: float = 0.4, min_keep: int = 2, method: str = "cosine"
):
    out = compress_chunk_query_aware(text, question, enc, q_vec=q_vec,
                                     target_reduction=target_reduction, min_keep=min_keep, method=method)
    # tái tạo danh sách câu đã chọn để annotate theo [id]
    sents, _sp = _sentences_with_spans(text)
    picked = out.get("picked_idx", [])
    out["picked_sents"] = [sents[i]
                           for i in picked] if picked and sents else []
    return out

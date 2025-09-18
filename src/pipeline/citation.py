from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path


def enumerate_citations(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Gán id tuần tự 1..k cho mỗi hit theo thứ hạng hiện tại.
    Mỗi hit dự kiến có keys: file, page (có thể None), chunk_id, (optional) sent_spans, compressed_text/picked_sents.
    """
    citations = []
    for h in hits:
        cid = h["rank"]  # dùng rank làm id
        file = h.get("file", "?")
        page = h.get("page", None)
        chunk_id = h.get("chunk_id", -1)
        # preview ngắn cho tiện xem nguồn
        text = (h.get("compressed_text") or h.get(
            "text") or "").strip().replace("\n", " ")
        preview = (text[:140] + ("..." if len(text) > 140 else ""))
        citations.append({
            "id": cid,
            "file": file,
            "page": page,
            "chunk_id": chunk_id,
            "preview": preview
        })
    return citations


def annotate_sentences_with_citation(hit: Dict[str, Any]) -> str:
    """
    Thêm [id] sau từng câu đã chọn (nếu có), ngược lại thêm cuối đoạn nén.
    """
    cid = hit["rank"]
    picked_sents = hit.get("picked_sents") or []
    if picked_sents:
        parts = [s.strip() + f" [{cid}]" for s in picked_sents if s.strip()]
        return " ".join(parts)
    # fallback: gắn [id] cuối đoạn
    ct = (hit.get("compressed_text") or hit.get("text") or "").strip()
    return (ct + (f" [{cid}]" if ct else "")).strip()


def format_citation_footer(citations: List[Dict[str, Any]]) -> str:
    """
    Trả footer dạng:
    [1] <file>#chunk<id> (page X) — preview...
    """
    lines = []
    for c in citations:
        page = f" (page {c['page']})" if c.get(
            "page") not in (None, "", -1) else ""
        lines.append(
            f"[{c['id']}] {c['file']}#chunk{c['chunk_id']}{page} — {c['preview']}")
    return "\n".join(lines)

import argparse
import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from src.preprocess import TextPreprocessor
from src.embed.embed_faiss import DenseEncoder, load_artifacts
from src.retrieval.bm25 import build_bm25_from_chunks, load_bm25, save_bm25, bm25_search
from src.retrieval.hybrid import fuse_dense_bm25
from src.retrieval.mmr import mmr_select
from src.embed.rerank_bge import BGECrossReranker


def _strip_accents_lower(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.split())


def _match_any_kw(text: str, kws: List[str], accent_insensitive: bool = True) -> bool:
    if not kws:
        return False
    t = _strip_accents_lower(text) if accent_insensitive else text.lower()
    for kw in kws:
        if not kw.strip():
            continue
        k = _strip_accents_lower(kw) if accent_insensitive else kw.lower()
        if k in t:
            return True
    return False


def _metrics(results: List[Tuple[Optional[int], int]]) -> Tuple[float, float, float]:
    n = len(results) or 1
    hit = sum(1 for r, _ in results if r is not None) / n
    mrr = sum((1.0/r) for r, _ in results if r is not None) / n
    top1 = sum(1 for r, _ in results if r == 1) / n
    return hit, mrr, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--golden", type=Path, required=True)

    ap.add_argument(
        "--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--embedding_model", default=None)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--pool_k", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.6)

    ap.add_argument("--mmr", action="store_true")
    ap.add_argument("--mmr_lambda", type=float, default=0.5)
    ap.add_argument("--mmr_k_out", type=int, default=12)

    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--rerank_in", type=int, default=32)
    ap.add_argument("--rerank_out", type=int, default=8)
    ap.add_argument("--rerank_model", default="BAAI/bge-reranker-v2-m3")

    ap.add_argument("--accent_insensitive", action="store_true", default=True)
    args = ap.parse_args()

    index, chunks, meta = load_artifacts(args.artifacts_dir)
    enc = DenseEncoder(args.embedding_model or meta.get("embedding_model"))
    # chuẩn bị BM25 (nếu cần)
    bm25 = None
    bm25_path = Path(args.artifacts_dir) / "bm25.pkl"
    if args.retriever in ("bm25", "hybrid"):
        if bm25_path.exists():
            bm25 = load_bm25(bm25_path)
        else:
            bm25 = build_bm25_from_chunks(chunks)
            save_bm25(bm25, bm25_path)

    prep = TextPreprocessor(language='vi')
    lines = [l for l in args.golden.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    qs = [json.loads(l) for l in lines]

    per_lang = defaultdict(list)
    all_res: List[Tuple[Optional[int], int]] = []

    for ex in qs:
        q = prep.preprocess(ex["query"])
        kws = ex.get("keywords", [])
        lang = ex.get("lang", "unk")

        # retrieve pool & base order
        if args.retriever == "dense":
            q_emb = enc.encode([q], batch_size=1, normalize=True)
            scores, idx = index.search(q_emb, max(args.pool_k, args.top_k))
            base_ids = list(idx[0])
            q_vec = q_emb[0]
        elif args.retriever == "bm25":
            ids, _ = bm25_search(bm25, q, top_k=max(
                args.pool_k, args.top_k))  # type: ignore
            base_ids = ids
            q_vec = enc.encode([q], batch_size=1, normalize=True)[0]
        else:
            q_emb = enc.encode([q], batch_size=1, normalize=True)
            d_scores, d_idx = index.search(q_emb, args.pool_k)
            dense_map = {int(i): float(d_scores[0][r])
                         for r, i in enumerate(d_idx[0])}
            b_idx, b_scs = bm25_search(
                bm25, q, top_k=args.pool_k)  # type: ignore
            bm25_map = {int(i): float(sc) for i, sc in zip(b_idx, b_scs)}
            fused = fuse_dense_bm25(dense_map, bm25_map, alpha=args.alpha)
            base_ids = [i for (i, _, _, _) in fused]
            q_vec = q_emb[0]

        order = base_ids

        # MMR
        if args.mmr:
            mmr_pool = min(len(order), max(
                args.mmr_k_out, args.rerank_in, args.top_k))
            docs = [chunks[i]["text"] for i in order[:mmr_pool]]
            d_vecs = enc.encode(docs, batch_size=64, normalize=True)
            order = mmr_select(
                q_vec, d_vecs, order[:mmr_pool], k_out=args.mmr_k_out, lam=args.mmr_lambda)

        # Rerank
        if args.rerank:
            rerank_in = min(len(order), max(args.rerank_in, args.top_k))
            rr_docs = [chunks[i]["text"] for i in order[:rerank_in]]
            rr = BGECrossReranker(args.rerank_model)
            rr_order, _ = rr.rerank(q, rr_docs, top_m=min(
                args.rerank_out, args.top_k), batch_size=16)
            order = [order[j] for j in rr_order]

        # evaluate
        topk = min(args.top_k, len(order))
        found_rank: Optional[int] = None
        for r, i in enumerate(order[:topk], start=1):
            if _match_any_kw(chunks[i]["text"], kws, args.accent_insensitive):
                found_rank = r
                break

        per_lang[lang].append((found_rank, topk))
        all_res.append((found_rank, topk))

    hit, mrr, top1 = _metrics(all_res)
    print(f"Total queries: {len(all_res)}")
    label = args.retriever.upper()
    if args.mmr:
        label += "+MMR"
    if args.rerank:
        label += "+RERANK"
    print(f"{label:<16} Hit@{args.top_k}: {hit:.3f}   MRR@{args.top_k}: {mrr:.3f}   Top-1: {top1:.3f}")
    for lang, res in per_lang.items():
        h, m, t1 = _metrics(res)
        print(f"[{lang}]   Hit@{args.top_k}: {h:.3f}   MRR@{args.top_k}: {m:.3f}   Top-1: {t1:.3f}   (n={len(res)})")


if __name__ == "__main__":
    main()

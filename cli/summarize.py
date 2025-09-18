import argparse
from pathlib import Path
from typing import List, Dict, Any

from src.preprocess import TextPreprocessor
from src.embed.embed_faiss import DenseEncoder, load_artifacts
from src.retrieval.bm25 import build_bm25_from_chunks, load_bm25, save_bm25, bm25_search
from src.retrieval.hybrid import fuse_dense_bm25
from src.retrieval.mmr import mmr_select
from src.embed.rerank_bge import BGECrossReranker
from src.pipeline.compress import compress_chunk_query_aware_with_sents
from src.pipeline.citation import enumerate_citations, annotate_sentences_with_citation, format_citation_footer
from src.pipeline.guardrail import guardrail_should_abstain, aggregate_context
from src.gen.gen_models import SummarizerRouter
from src.gen.summarize import build_prompt, detect_lang_fast


def _bm25_obj(artifacts_dir: Path, chunks: List[Dict[str, Any]]):
    bm25_path = artifacts_dir / "bm25.pkl"
    if bm25_path.exists():
        return load_bm25(bm25_path)
    b = build_bm25_from_chunks(chunks)
    save_bm25(b, bm25_path)
    return b


def _retrieve(args, q: str, enc: DenseEncoder, q_vec, index, chunks):
    # pool ứng viên
    if args.retriever == "dense":
        scores, idx = index.search(
            q_vec[None, :], max(args.pool_k, args.top_k))
        cand_ids = [int(i) for i in idx[0]]
        fused_scores = {int(i): float(scores[0][r])
                        for r, i in enumerate(idx[0])}
    elif args.retriever == "bm25":
        b = _bm25_obj(args.artifacts_dir, chunks)
        ids, scs = bm25_search(b, q, top_k=max(args.pool_k, args.top_k))
        cand_ids = ids
        fused_scores = {int(i): float(sc) for i, sc in zip(ids, scs)}
    else:
        scores, idx = index.search(q_vec[None, :], args.pool_k)
        dense_map = {int(i): float(scores[0][r]) for r, i in enumerate(idx[0])}
        b = _bm25_obj(args.artifacts_dir, chunks)
        b_idx, b_scs = bm25_search(b, q, top_k=args.pool_k)
        bm25_map = {int(i): float(sc) for i, sc in zip(b_idx, b_scs)}
        fused = fuse_dense_bm25(dense_map, bm25_map, alpha=args.alpha)
        cand_ids = [i for i, _, _, _ in fused]
        fused_scores = {i: f for i, f, _, _ in fused}

    # MMR
    order_after = cand_ids
    if args.mmr and cand_ids:
        mmr_pool = min(len(cand_ids), max(
            args.mmr_k_out, args.rerank_in, args.top_k))
        docs = [chunks[i]["text"] for i in cand_ids[:mmr_pool]]
        d_vecs = enc.encode(docs, batch_size=64, normalize=True)
        order_after = mmr_select(
            q_vec, d_vecs, cand_ids[:mmr_pool], k_out=args.mmr_k_out, lam=args.mmr_lambda)

    # Rerank
    final_ids = order_after[:args.top_k]
    rerank_scores = {}
    if args.rerank and order_after:
        rin = min(len(order_after), max(args.rerank_in, args.top_k))
        in_docs = [chunks[i]["text"] for i in order_after[:rin]]
        rr = BGECrossReranker(args.rerank_model)
        order_rr, scores_rr = rr.rerank(q, in_docs, top_m=min(
            args.rerank_out, args.top_k), batch_size=16)
        final_ids = [order_after[j] for j in order_rr]
        rerank_scores = {final_ids[k]: scores_rr[k]
                         for k in range(len(final_ids))}

    return final_ids, fused_scores, rerank_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--q", required=True)
    ap.add_argument(
        "--mode", choices=["tldr", "executive", "qfs"], default="qfs")
    ap.add_argument("--answer_lang",
                    choices=["Auto", "vi", "en"], default="Auto")

    # retrieval opts
    ap.add_argument(
        "--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--embedding_model", default=None)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--pool_k", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.6)
    # MMR
    ap.add_argument("--mmr", action="store_true")
    ap.add_argument("--mmr_lambda", type=float, default=0.5)
    ap.add_argument("--mmr_k_out", type=int, default=12)
    # Rerank
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--rerank_in", type=int, default=32)
    ap.add_argument("--rerank_out", type=int, default=8)
    ap.add_argument("--rerank_model", default="BAAI/bge-reranker-v2-m3")
    # Compression
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--target_reduction", type=float, default=0.4)
    ap.add_argument("--min_keep", type=int, default=2)
    ap.add_argument("--compress_method",
                    choices=["cosine", "tfidf"], default="cosine")
    # Guardrail
    ap.add_argument("--guardrail", action="store_true")
    ap.add_argument("--min_tokens", type=int, default=40)
    ap.add_argument("--min_sim", type=float, default=0.18)
    args = ap.parse_args()

    # 1) Preprocess query
    prep = TextPreprocessor(language='vi')  # hoặc auto
    q = prep.preprocess(args.q)

    # 2) Load artifacts + encoder
    index, chunks, meta = load_artifacts(args.artifacts_dir)
    enc = DenseEncoder(args.embedding_model or meta.get("embedding_model"))
    q_vec = enc.encode([q], batch_size=1, normalize=True)[0]

    # 3) Retrieve → (MMR) → (Rerank)
    final_ids, fused_scores, rerank_scores = _retrieve(
        args, q, enc, q_vec, index, chunks)

    # 4) Compression + Citations annotate
    hits: List[Dict[str, Any]] = []
    for r, i in enumerate(final_ids, start=1):
        rec = chunks[i]
        row = {
            "rank": r,
            "score": float(fused_scores.get(i, 0.0)),
            "file": rec["file"],
            "page": rec.get("page"),
            "chunk_id": rec["chunk_id"],
            "text": rec["text"],
        }
        if args.compress:
            out = compress_chunk_query_aware_with_sents(
                text=rec["text"], question=q, enc=enc, q_vec=q_vec,
                target_reduction=args.target_reduction, min_keep=args.min_keep, method=args.compress_method
            )
            row.update({
                "compressed_text": out.get("compressed_text", ""),
                "picked_sents": out.get("picked_sents", []),
                "sent_spans": out.get("sent_spans", []),
                "tokens_before": out.get("tokens_before", 0),
                "tokens_after": out.get("tokens_after", 0),
            })
        if i in rerank_scores:
            row["rerank_s"] = float(rerank_scores[i])
        hits.append(row)

    cits_footer = ""
    if hits:
        cits = enumerate_citations(hits)
        for h in hits:
            h["annotated_text"] = annotate_sentences_with_citation(h)
        cits_footer = format_citation_footer(cits)

    # 5) Guardrail (abstain?) — dùng context đã nén
    if args.guardrail:
        abstain, reason, sim, toks = guardrail_should_abstain(
            q, hits, enc, q_vec=q_vec, min_tokens=args.min_tokens, min_sim=args.min_sim
        )
        if abstain:
            print(
                f"[GUARDRAIL] {reason} | sim={sim:.3f} tokens={toks} → Không đủ thông tin.")
            if cits_footer:
                print("\nCitations:\n" + cits_footer)
            return

    # 6) Build annotated context
    context_annotated = "\n".join([h.get("annotated_text") or h.get(
        "compressed_text") or h["text"] for h in hits]).strip()

    # 7) Route ngôn ngữ đầu ra
    if args.answer_lang == "Auto":
        out_lang = detect_lang_fast(args.q)
    else:
        out_lang = args.answer_lang.lower()

    # 8) Prompt & Generate
    from src.gen.summarize import build_prompt
    from src.gen.gen_models import SummarizerRouter
    prompt = build_prompt(context_annotated, args.q,
                          mode=args.mode, out_lang=out_lang)
    router = SummarizerRouter()   # có thể override id model nếu muốn
    gen = router.get(out_lang)

    # presets: nhẹ tay để tránh lố token
    max_new = 220 if args.mode == "executive" else 180
    num_beams = 1  # beam=1 để giữ tốc độ
    answer = gen.generate(prompt, max_new_tokens=max_new,
                          temperature=0.3, top_p=0.95, num_beams=num_beams)

    # 9) In kết quả + footer
    print("\n===== ANSWER =====\n")
    print(answer.strip())
    if cits_footer:
        print("\nCitations:\n" + cits_footer)


if __name__ == "__main__":
    main()

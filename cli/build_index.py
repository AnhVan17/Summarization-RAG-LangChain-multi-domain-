import argparse
import os
import torch
from pathlib import Path
from typing import List, Dict, Any
from src.utils_io import load_file
from src.preprocess import TextPreprocessor
from src.chunk import chunk_by_sentences
from src.embed.embed_faiss import DenseEncoder, build_faiss_index, save_artifacts, _now_iso


def _gather_docs(input_dir: Path) -> List[Path]:
    exts = {".txt", ".pdf"}
    files: List[Path] = []
    for root, _, fs in os.walk(input_dir):
        for name in fs:
            if Path(name).suffix.lower() in exts:
                files.append(Path(root) / name)
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--artifacts_dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--embedding_model",
                    default="intfloat/multilingual-e5-base")
    ap.add_argument("--max_chars", type=int, default=900)
    ap.add_argument("--overlap_sents", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    files = _gather_docs(args.input_dir)
    if not files:
        raise SystemExit(f"No files found txt/pdf in {args.input_dir}.")

    prep = TextPreprocessor(language="vi")
    records: List[Dict[str, Any]] = []
    texts: List[str] = []

    for doc_id, path in enumerate(files):
        raw = load_file(str(path))
        text = prep.preprocess(raw)
        chunks = chunk_by_sentences(
            text, max_chars=args.max_chars, overlap_sents=args.overlap_sents)

        for i, chunk in enumerate(chunks):
            records.append({
                "doc_id": doc_id,
                "file": str(path),
                "chunk_id": i,
                "text": chunk,
                "page": None,
            })
            texts.append(chunk)

    # SỬA: dùng model_id=..., không phải model=...
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = DenseEncoder(model_id=args.embedding_model, device=device)

    # SỬA: normalize đúng chính tả; encode trả về numpy float32
    emb = encoder.encode(texts, batch_size=args.batch_size, normalize=True)

    index = build_faiss_index(emb)

    meta = {
        "embedding_model": args.embedding_model,
        "dim": int(emb.shape[1]),
        "normalized": True,
        "preprocess": ["normalize_newlines", "normalize_unicode", "basic_cleaning", "collapse_spaces"],
        "chunking": {"type": "sentences", "max_chars": args.max_chars, "overlap_sents": args.overlap_sents},
        "built_at": _now_iso(),
        "num_chunks": len(texts),
        "num_files": len(files),
        "device": device,
    }

    arts = save_artifacts(args.artifacts_dir, index, records, meta)
    print(f"[OK] Saved index:  {arts.index_path}")
    print(f"[OK] Saved chunks: {arts.chunks_path}")
    print(f"[OK] Saved meta:   {arts.meta_path}")


if __name__ == "__main__":
    main()

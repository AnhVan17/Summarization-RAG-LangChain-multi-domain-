from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BuildArtifacts:
    index_path: Path
    chunks_path: Path
    meta_path: Path


class DenseEncoder:
    def __init__(self, model_id: str = "intfloat/multilingual-e5-base", device: str | None = None):
        self.model_id = model_id
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.model_id, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        if emb.dtype != np.float32:
            emb = emb.astype("float32")
        return emb


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_artifacts(
    artifacts_dir: str | Path,
    index: "faiss.Index",
    chunks: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> BuildArtifacts:
    adir = Path(artifacts_dir)
    adir.mkdir(parents=True, exist_ok=True)

    index_path = adir / "index.faiss"
    chunks_path = adir / "chunks.pkl"
    meta_path = adir / "meta.json"

    faiss.write_index(index, str(index_path))

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return BuildArtifacts(index_path, chunks_path, meta_path)


def load_artifacts(artifacts_dir: str | Path) -> Tuple["faiss.Index", List[Dict[str, Any]], Dict[str, Any]]:
    adir = Path(artifacts_dir)
    index = faiss.read_index(str(adir / "index.faiss"))
    with open(adir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open(adir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, chunks, meta

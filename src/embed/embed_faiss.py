import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np


def build_index(texts, emb_model="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(emb_model)
    emb = model.encode(texts, normalize_embeddings=True).astype("float32")
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, model


def retrieve(query, index, model, chunks, k=5):
    query_emb = model.encode(
        [query], normalize_embeddings=True).astype("float32")
    D, I = index.search(query_emb, k=3)
    return [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]


def save(index, chunks, outdir="."):
    os.makedirs(outdir, exist_ok=True)
    faiss.write_index(index, os.path.join(outdir, "index.faiss"))
    with open(os.path.join(outdir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)


def load(index_path="index.faiss", meta_path="chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

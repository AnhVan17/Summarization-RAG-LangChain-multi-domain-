from __future__ import annotations
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BGECrossReranker:
    """
    BGE cross-encoder: 'BAAI/bge-reranker-v2-m3'
    Trả điểm (logit) cho từng (query, doc). Điểm cao hơn = liên quan hơn.
    """

    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-m3", device: str | None = None):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id)
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, docs: List[str], batch_size: int = 16, max_length: int = 512) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            enc = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.view(-1)
            scores.extend(logits.detach().cpu().tolist())
        return scores

    def rerank(self, query: str, docs: List[str], top_m: int | None = None, batch_size: int = 16) -> Tuple[List[int], List[float]]:
        scores = self.score(query, docs, batch_size=batch_size)
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        if top_m is not None:
            order = order[:top_m]
        return order, [scores[i] for i in order]

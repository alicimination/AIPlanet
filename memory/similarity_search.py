"""Similarity search over memory entries using embeddings."""

from __future__ import annotations

from typing import List, Dict, Any
import json

import numpy as np
from memory.memory_store import MemoryStore
from utils.local_paths import ensure_local_model_env


class MemorySimilarity:
    """Retrieve similar historical problems for self-learning reuse."""

    def __init__(self):
        self.store = MemoryStore()
        paths = ensure_local_model_env()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=str(paths["sentence_transformers"]))

    def find_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        rows = self.store.get_recent(limit=200)
        if not rows:
            return []
        corpus = [r["original_input"] for r in rows]
        vectors = self.model.encode(corpus, normalize_embeddings=True)
        qv = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(vectors, qv)
        ranked_idx = np.argsort(-sims)[:top_k]

        out: List[Dict[str, Any]] = []
        for idx in ranked_idx:
            rec = rows[int(idx)].copy()
            rec["similarity"] = float(sims[int(idx)])
            try:
                rec["parsed_problem"] = json.loads(rec["parsed_problem"])
                rec["verification_result"] = json.loads(rec["verification_result"])
            except Exception:
                pass
            out.append(rec)
        return out

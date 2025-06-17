import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGMemory:
    """Simple FAISS-backed retrieval augmented generation memory."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(model_name)
        self.docs: List[str] = []
        self.index = None
        self.dimension = None

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.dimension = dim
            self.index = faiss.IndexFlatL2(dim)

    def add_document(self, text: str):
        if not text:
            return
        emb = self.embed_model.encode([text])[0].astype('float32')
        self._ensure_index(len(emb))
        if len(self.docs) == 0:
            self.index.add(np.array([emb]))
        else:
            self.index.add(np.array([emb]))
        self.docs.append(text)

    def query(self, text: str, top_k: int = 3) -> List[str]:
        if self.index is None:
            return []
        emb = self.embed_model.encode([text]).astype('float32')
        distances, indices = self.index.search(emb, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.docs):
                results.append(self.docs[idx])
        return results

    @classmethod
    def from_file(cls, path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        inst = cls(model_name)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    inst.add_document(line.strip())
        return inst

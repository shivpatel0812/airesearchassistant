from typing import List
import faiss
from sentence_transformers import SentenceTransformer

from data_ingest import Document

class VectorIndex:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.docs: List[Document] = []

    def build(self, docs: List[Document]):
        embeddings = self.model.encode([d.text for d in docs], show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.docs = docs

    def query(self, question: str, k: int = 3):
        if self.index is None:
            raise ValueError('Index not built')
        q_emb = self.model.encode([question])
        distances, indices = self.index.search(q_emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({'doc': self.docs[idx], 'score': float(dist)})
        return results

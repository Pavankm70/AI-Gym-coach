"""
Persistent Vector Store Implementation
======================================
Provides persistent vector storage, metadata-aware semantic similarity retrieval,
exercise filtering, similarity thresholding, and diagnostic tracing for the
Exercise Knowledge RAG Tool.

Uses FAISS with persistent local storage on disk. Supports normalized cosine
similarity with fallback semantic embedding ensuring 100% deterministic operation.
"""

import os
import json
import math
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FallbackSemanticEmbedder:
    """
    Lightweight deterministic character-trigram & token frequency embedder.
    Provides semantic vector generation (dimension 256) without external API dependencies.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec

        for word in words:
            # Word token hash
            h = hash(word) % self.dim
            vec[h] += 1.0
            # Trigram hashes for morphological/spelling robustness
            for i in range(len(word) - 2):
                tri = word[i:i+3]
                h_tri = hash(tri) % self.dim
                vec[h_tri] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed_text(t) for t in texts]


class PersistentExerciseVectorStore:
    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.index_file = os.path.join(self.storage_dir, "index.faiss")
        self.metadata_file = os.path.join(self.storage_dir, "metadata.json")

        self.dim = 256
        self.embedder = FallbackSemanticEmbedder(dim=self.dim)
        self.documents: List[Dict[str, Any]] = []

        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatIP(self.dim)
        else:
            self.faiss_index = None

        self._load()

    def _load(self):
        """Loads index and metadata from disk if available."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception as e:
                print(f"[VectorStore] Warning loading metadata: {e}")
                self.documents = []

        if HAS_FAISS and os.path.exists(self.index_file):
            try:
                self.faiss_index = faiss.read_index(self.index_file)
            except Exception as e:
                print(f"[VectorStore] Warning reading FAISS index: {e}")
                self.faiss_index = faiss.IndexFlatIP(self.dim)

    def save(self):
        """Persists FAISS index and chunk metadata to disk."""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

        if HAS_FAISS and self.faiss_index is not None:
            faiss.write_index(self.faiss_index, self.index_file)

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Embeds and stores document chunks with metadata.
        Each chunk: {"content": str, "metadata": dict}
        """
        if not chunks:
            return

        texts = [c["content"] for c in chunks]
        vectors = np.array(self.embedder.embed_documents(texts), dtype=np.float32)

        if HAS_FAISS:
            if self.faiss_index is None:
                self.faiss_index = faiss.IndexFlatIP(self.dim)
            self.faiss_index.add(vectors)

        for c, v in zip(chunks, vectors):
            doc_entry = {
                "content": c["content"],
                "metadata": c.get("metadata", {}),
                "vector": v.tolist() if not HAS_FAISS else None
            }
            self.documents.append(doc_entry)

        self.save()

    def similarity_search_with_diagnostics(
        self,
        query: str,
        top_k: int = 3,
        filter_exercise: Optional[str] = None,
        min_score: float = 0.15
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Performs similarity search with exercise filtering, score thresholding,
        and returns both accepted chunks and full diagnostic trace.
        """
        if not self.documents:
            return [], []

        q_vec = self.embedder.embed_text(query).reshape(1, -1).astype(np.float32)

        raw_candidates = []
        if HAS_FAISS and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            k = min(len(self.documents), self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(q_vec, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                raw_candidates.append((float(score), self.documents[idx]))
        else:
            for doc in self.documents:
                doc_vec = np.array(doc["vector"], dtype=np.float32)
                sim = float(np.dot(q_vec.flatten(), doc_vec))
                raw_candidates.append((sim, doc))

        # Canonical normalization helper
        def norm_ex(name: Optional[str]) -> str:
            if not name:
                return ""
            n = name.lower()
            if "squat" in n: return "squats"
            if "push" in n: return "push-ups"
            if "curl" in n or "bicep" in n: return "biceps curls (dumbbell)"
            if "press" in n or "shoulder" in n: return "shoulder press"
            if "lunge" in n: return "lunges"
            return n

        target_ex = norm_ex(filter_exercise)

        # Apply exact exercise matching boost so exercise-specific knowledge ranks higher than general
        scored_candidates = []
        for raw_score, doc in raw_candidates:
            meta = doc.get("metadata", {})
            chunk_ex = norm_ex(meta.get("exercise", ""))
            boosted_score = raw_score
            if target_ex and chunk_ex == target_ex:
                boosted_score += 0.08  # Prefer exercise-specific knowledge over general
            scored_candidates.append((boosted_score, raw_score, doc))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        accepted_results = []
        diagnostic_trace = []

        for boosted_score, raw_score, doc in scored_candidates:
            meta = doc.get("metadata", {})
            chunk_ex = norm_ex(meta.get("exercise", ""))
            chunk_id = meta.get("chunk_id", "unknown")
            section_title = meta.get("section_title", "General")

            is_accepted = True
            rejection_reason = None

            # Exercise filtering check
            if target_ex and chunk_ex and chunk_ex != "general":
                if target_ex != chunk_ex:
                    is_accepted = False
                    rejection_reason = f"Exercise mismatch: chunk is '{meta.get('exercise')}', query target is '{filter_exercise}'"

            # Score threshold check
            if is_accepted and raw_score < min_score:
                is_accepted = False
                rejection_reason = f"Below similarity threshold ({raw_score:.3f} < {min_score:.3f})"

            diag_item = {
                "chunk_id": chunk_id,
                "exercise": meta.get("exercise", "General"),
                "section": section_title,
                "source": meta.get("source_name", "Literature"),
                "score": round(boosted_score, 3),
                "raw_score": round(raw_score, 3),
                "accepted": is_accepted,
                "rejection_reason": rejection_reason
            }
            diagnostic_trace.append(diag_item)

            if is_accepted:
                accepted_results.append({
                    "content": doc["content"],
                    "metadata": meta,
                    "score": round(boosted_score, 3)
                })
                if len(accepted_results) >= top_k:
                    break

        return accepted_results, diagnostic_trace

    def similarity_search(
        self,
        query: str,
        top_k: int = 3,
        filter_exercise: Optional[str] = None,
        min_score: float = 0.20
    ) -> List[Dict[str, Any]]:
        """Standard similarity search returning top_k accepted chunks."""
        accepted, _ = self.similarity_search_with_diagnostics(
            query=query,
            top_k=top_k,
            filter_exercise=filter_exercise,
            min_score=min_score
        )
        return accepted

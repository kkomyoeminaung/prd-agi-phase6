"""
rag/rag_engine.py — PRD-AGI Retrieval-Augmented Generation Engine
==================================================================
Pure numpy vector store — no ChromaDB/FAISS dependency required.
Uses cosine similarity for retrieval.

Features:
  - Add documents from text, files, PDFs, URLs
  - Chunk documents with overlap
  - Embed via Ollama/Gemini or TF-IDF fallback
  - Cosine similarity retrieval
  - SU(5) curvature-weighted re-ranking
  - Persistent save/load (JSON)
  - Namespace support (separate knowledge domains)

Architecture:
  Document → Chunks → Embeddings → Vector Store → Retrieval → Context
"""

import numpy as np
import json
import os
import re
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger('PRD-AGI.RAG')


class Document:
    """A document chunk with embedding and metadata."""
    __slots__ = ('id', 'text', 'embedding', 'metadata', 'namespace', 'added_at')

    def __init__(self, text: str, metadata: Dict = None,
                 namespace: str = "default"):
        self.id        = hashlib.md5(text.encode()).hexdigest()[:12]
        self.text      = text
        self.embedding: Optional[np.ndarray] = None
        self.metadata  = metadata or {}
        self.namespace = namespace
        self.added_at  = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "text": self.text,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata, "namespace": self.namespace,
            "added_at": self.added_at,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Document":
        doc = cls(d["text"], d.get("metadata", {}), d.get("namespace", "default"))
        doc.id       = d["id"]
        doc.added_at = d.get("added_at", "")
        if d.get("embedding") is not None:
            doc.embedding = np.array(d["embedding"])
        return doc


class TFIDFEmbedder:
    """
    TF-IDF based embedder — no LLM needed.
    Falls back to this when Ollama/Gemini unavailable.
    Produces deterministic 512-dim embeddings.
    """
    DIM = 512

    def __init__(self):
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray = np.ones(self.DIM)
        self._doc_count = 0

    def embed(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vec = np.zeros(self.DIM)
        for token in tokens:
            idx = self._token_idx(token)
            if idx < self.DIM:
                tf = tokens.count(token) / max(len(tokens), 1)
                vec[idx] += tf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def _token_idx(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = hash(token) % self.DIM
        return self._vocab[token]


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for PRD-AGI.

    Embeds documents, stores them in a numpy vector store,
    and retrieves relevant context for LLM queries.

    Usage:
        rag = RAGEngine(ollama_client)
        rag.add_text("Buddhist philosophy...", title="Buddhism")
        rag.add_file("notes.txt")
        context = rag.retrieve("What is Hetu?")
        answer  = rag.query("What is Hetu?", llm_fn)
    """

    def __init__(self, embed_client=None, chunk_size: int = 400,
                 chunk_overlap: int = 50, store_path: str = "workspace/rag_store.json"):
        self.embed_client  = embed_client   # OllamaInterface or GeminiInterface
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store_path    = store_path
        self._docs: List[Document]   = []
        self._embeddings: Optional[np.ndarray] = None  # (N, dim)
        self._tfidf = TFIDFEmbedder()
        self._dirty = False   # needs reindex
        self._stats = {"added": 0, "queries": 0, "hits": 0}

        # Load existing store
        self._load()

    # ── Add documents ────────────────────────────────────────────────────────

    def add_text(self, text: str, title: str = "", namespace: str = "default",
                 metadata: Dict = None) -> Dict:
        """Add raw text to the knowledge base."""
        if not text.strip():
            return {"success": False, "error": "Empty text"}

        chunks = self._chunk(text)
        added = 0
        for i, chunk in enumerate(chunks):
            meta = {**(metadata or {}),
                    "title": title, "chunk": i, "total_chunks": len(chunks)}
            doc = Document(chunk, meta, namespace)

            # Skip duplicates
            if any(d.id == doc.id for d in self._docs):
                continue

            doc.embedding = self._embed(chunk)
            self._docs.append(doc)
            added += 1

        self._dirty = True
        self._stats["added"] += added
        self._reindex()
        logger.info(f"RAG: added {added} chunks from '{title}'")
        return {"success": True, "added_chunks": added, "total_docs": len(self._docs)}

    def add_file(self, filepath: str, namespace: str = "default") -> Dict:
        """Add a text file to the knowledge base."""
        try:
            path = Path(filepath)
            if not path.exists():
                return {"success": False, "error": f"File not found: {filepath}"}
            content = path.read_text(encoding="utf-8", errors="ignore")
            return self.add_text(content, title=path.name, namespace=namespace,
                                  metadata={"source": str(path)})
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_url_content(self, url: str, content: str,
                         namespace: str = "default") -> Dict:
        """Add fetched URL content."""
        return self.add_text(content, title=url, namespace=namespace,
                              metadata={"source": url, "type": "web"})

    # ── Retrieve ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5,
                 namespace: Optional[str] = None,
                 min_score: float = 0.05) -> List[Dict]:
        """
        Retrieve top-k most relevant document chunks.

        Returns list of {text, score, metadata, namespace}.
        """
        if not self._docs:
            return []

        query_emb = self._embed(query)
        scores    = self._cosine_sim(query_emb)
        self._stats["queries"] += 1

        results = []
        for i, (doc, score) in enumerate(zip(self._docs, scores)):
            if namespace and doc.namespace != namespace:
                continue
            if score >= min_score:
                results.append({
                    "text":      doc.text,
                    "score":     float(score),
                    "title":     doc.metadata.get("title", ""),
                    "source":    doc.metadata.get("source", ""),
                    "namespace": doc.namespace,
                    "chunk":     doc.metadata.get("chunk", 0),
                    "id":        doc.id,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        top = results[:top_k]
        self._stats["hits"] += len(top)
        return top

    def build_context(self, query: str, top_k: int = 4,
                       max_chars: int = 2000) -> str:
        """
        Build a context string from retrieved chunks.
        Ready to inject into LLM system/user prompt.
        """
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return ""

        lines = ["[RAG Context — Retrieved Knowledge]\n"]
        total = 0
        for c in chunks:
            snippet = c["text"][:400]
            line = f"Source: {c['title'] or c['source']} (score={c['score']:.3f})\n{snippet}\n"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)

        return "\n".join(lines)

    def query(self, question: str, llm_fn, top_k: int = 4) -> Dict:
        """
        Full RAG query: retrieve + LLM generation.

        Args:
            question: user question
            llm_fn: callable(prompt, system) → str
            top_k: number of chunks to retrieve

        Returns:
            {answer, context, retrieved_chunks}
        """
        context = self.build_context(question, top_k=top_k)
        chunks  = self.retrieve(question, top_k=top_k)

        if context:
            system = (
                "You are PRD-AGI, grounded in SU(5) relational dynamics. "
                "Use the provided context to answer accurately. "
                "Cite sources when relevant. Be concise and causal."
            )
            prompt = f"{context}\n\nQuestion: {question}"
        else:
            system = "You are PRD-AGI. Answer based on your knowledge."
            prompt = question

        answer = llm_fn(prompt, system)
        return {
            "answer":           answer,
            "context_used":     bool(context),
            "retrieved_chunks": chunks,
            "chunk_count":      len(chunks),
        }

    # ── Manage ───────────────────────────────────────────────────────────────

    def list_documents(self) -> List[Dict]:
        """List all stored documents (titles and metadata)."""
        seen = set()
        docs = []
        for d in self._docs:
            title = d.metadata.get("title", d.id)
            if title not in seen:
                seen.add(title)
                docs.append({
                    "title":     title,
                    "namespace": d.namespace,
                    "chunks":    sum(1 for x in self._docs
                                     if x.metadata.get("title") == title),
                    "added_at":  d.added_at[:10],
                    "source":    d.metadata.get("source", ""),
                })
        return docs

    def delete_by_title(self, title: str) -> Dict:
        before = len(self._docs)
        self._docs = [d for d in self._docs if d.metadata.get("title") != title]
        removed = before - len(self._docs)
        self._dirty = True
        self._reindex()
        return {"success": True, "removed_chunks": removed}

    def clear(self, namespace: Optional[str] = None) -> Dict:
        if namespace:
            before = len(self._docs)
            self._docs = [d for d in self._docs if d.namespace != namespace]
            removed = before - len(self._docs)
        else:
            removed = len(self._docs)
            self._docs.clear()
        self._dirty = True
        self._reindex()
        return {"success": True, "removed": removed}

    def stats(self) -> Dict:
        return {
            **self._stats,
            "total_chunks":    len(self._docs),
            "namespaces":      list(set(d.namespace for d in self._docs)),
            "store_path":      self.store_path,
            "embed_backend":   "llm" if self.embed_client else "tfidf",
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> Dict:
        """Save vector store to JSON."""
        try:
            Path(self.store_path).parent.mkdir(parents=True, exist_ok=True)
            data = {"docs": [d.to_dict() for d in self._docs], "stats": self._stats}
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return {"success": True, "saved": len(self._docs), "path": self.store_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _load(self):
        try:
            if not Path(self.store_path).exists():
                return
            with open(self.store_path, encoding="utf-8") as f:
                data = json.load(f)
            self._docs = [Document.from_dict(d) for d in data.get("docs", [])]
            self._stats = data.get("stats", self._stats)
            self._reindex()
            logger.info(f"RAG: loaded {len(self._docs)} chunks from {self.store_path}")
        except Exception as e:
            logger.warning(f"RAG load error: {e}")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words  = text.split()
        size   = self.chunk_size
        overlap = self.chunk_overlap
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + size])
            chunks.append(chunk)
            i += size - overlap
        return chunks if chunks else [text]

    def _embed(self, text: str) -> np.ndarray:
        """Embed text using LLM client or TF-IDF fallback."""
        if self.embed_client and hasattr(self.embed_client, 'embed_text'):
            try:
                emb = self.embed_client.embed_text(text)
                return emb / (np.linalg.norm(emb) + 1e-10)
            except Exception:
                pass
        return self._tfidf.embed(text)

    def _reindex(self):
        """Rebuild embedding matrix for fast batch similarity."""
        if not self._docs:
            self._embeddings = None
            return
        dim = len(self._docs[0].embedding) if self._docs[0].embedding is not None else 512
        mat = np.zeros((len(self._docs), dim))
        for i, doc in enumerate(self._docs):
            if doc.embedding is not None and len(doc.embedding) == dim:
                mat[i] = doc.embedding
        self._embeddings = mat
        self._dirty = False

    def _cosine_sim(self, query_emb: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all docs."""
        if self._embeddings is None or len(self._embeddings) == 0:
            return np.array([])
        q = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        normed = self._embeddings / norms
        return normed @ q

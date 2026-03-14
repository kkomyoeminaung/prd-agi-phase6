"""
Ollama integration for text embedding and chat.
Upgraded: streaming support, model hot-swap, retry logic, richer mock responses.
"""

import requests
import numpy as np
import hashlib
import logging
import os
import time
from typing import Optional, Dict, List, Generator
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger('PRD-AGI')

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest")
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"


class OllamaInterface:
    """
    Interface to local Ollama server.
    Upgraded:
      - Retry logic (3 attempts) on transient errors
      - Streaming generate()
      - Hot-swap model without restart
      - Richer mock responses
      - Embed cache with LRU eviction
    """

    MOCK_RESPONSES = [
        "Analyzing through SU(5) lens: your query activates the Hetu (root cause) generator. "
        "The causal chain appears to originate in a Nissaya (support) condition. "
        "Curvature is nominal — proceeding with causal inference.",

        "The relational manifold shows elevated Indriya (governing) activation. "
        "Your question touches a domain governed by constraint-propagation logic. "
        "Gauge invariance preserved across all 24 generators.",

        "Sequential causality detected (Anantara pattern). "
        "This query unfolds in time — the step operators E12 through E25 are most active. "
        "Recommend Anantara agent for detailed analysis.",

        "Synthesis mode engaged. Multiple Paccaya conditions co-arise here: "
        "Hetu, Nissaya, and Sahajata are simultaneously activated. "
        "This is a complex interdependent phenomenon.",

        "Truth gate: PASS. Curvature well below threshold. "
        "The logical manifold is flat in this region — high confidence in causal consistency.",
    ]

    def __init__(
        self,
        mock: bool = MOCK_MODE,
        host: str = OLLAMA_HOST,
        embed_model: str = OLLAMA_EMBED_MODEL,
        chat_model: str = OLLAMA_CHAT_MODEL,
    ):
        self.mock = mock
        self.host = host.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.available = not mock and self._check()
        # Projection matrix: 768-dim embedding → 24-dim relational state
        rng = np.random.default_rng(seed=42)  # deterministic across restarts
        self._proj = rng.standard_normal((24, 768)) + 1j * rng.standard_normal((24, 768))
        self._proj /= np.linalg.norm(self._proj, axis=1, keepdims=True)
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._cache_max = 500
        logger.info(f"OllamaInterface | mock={mock} available={self.available} model={chat_model}")

    # ── Connection ─────────────────────────────────────────────────────────────

    def _check(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def reconnect(self) -> bool:
        """Try to reconnect to Ollama (e.g. after it starts up)."""
        self.available = self._check()
        return self.available

    def switch_model(self, chat_model: str):
        """Hot-swap chat model without restarting."""
        self.chat_model = chat_model
        logger.info(f"Chat model switched to: {chat_model}")

    # ── Embeddings ─────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._embed_cache:
            return self._embed_cache[key]

        if self.mock or not self.available:
            rng = np.random.default_rng(seed=int(key[:8], 16))
            emb = rng.standard_normal(768)
        else:
            emb = self._embed_with_retry(text)

        # LRU eviction
        if len(self._cache_order) >= self._cache_max:
            oldest = self._cache_order.pop(0)
            self._embed_cache.pop(oldest, None)
        self._embed_cache[key] = emb
        self._cache_order.append(key)
        return emb

    def _embed_with_retry(self, text: str, retries: int = 3) -> np.ndarray:
        for attempt in range(retries):
            try:
                r = requests.post(
                    f"{self.host}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                    timeout=15,
                )
                if r.status_code == 200:
                    return np.array(r.json()["embedding"], dtype=float)
            except Exception as e:
                logger.warning(f"Embed attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        rng = np.random.default_rng()
        return rng.standard_normal(768)

    def text_to_state(self, text: str, dna) -> "RelationalState":
        from core.engine import RelationalState
        emb = self.embed_text(text)
        psi = (self._proj @ emb).astype(np.complex128)
        norm = np.linalg.norm(psi)
        psi /= norm if norm > 1e-10 else 1.0
        return RelationalState(dna, psi)

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str, system: str = "", stream: bool = False) -> str:
        if self.mock or not self.available:
            return self._mock_response(prompt)
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{self.host}/api/chat",
                    json={"model": self.chat_model, "messages": msgs, "stream": False},
                    timeout=90,
                )
                if r.status_code == 200:
                    return r.json()["message"]["content"]
            except Exception as e:
                logger.warning(f"Generate attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(1.0)
        return "[Error] Could not reach Ollama. Enable Mock Mode or check `ollama serve`."

    def generate_stream(self, prompt: str, system: str = "") -> Generator[str, None, None]:
        """Yield tokens one by one (streaming)."""
        if self.mock or not self.available:
            for word in self._mock_response(prompt).split():
                yield word + " "
                time.sleep(0.03)
            return
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        try:
            with requests.post(
                f"{self.host}/api/chat",
                json={"model": self.chat_model, "messages": msgs, "stream": True},
                stream=True,
                timeout=120,
            ) as r:
                import json as _json
                for line in r.iter_lines():
                    if line:
                        data = _json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
        except Exception as e:
            yield f"[Stream error: {e}]"

    def _mock_response(self, prompt: str) -> str:
        idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(self.MOCK_RESPONSES)
        return f"[Mock Mode] {self.MOCK_RESPONSES[idx]}"

    def list_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

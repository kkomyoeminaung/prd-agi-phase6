"""
Google Gemini API integration for PRD-AGI.
Supports: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash
Handles: text generation, embeddings (text-embedding-004), streaming
"""

import requests
import numpy as np
import hashlib
import logging
import os
import time
import json
from typing import Optional, Dict, List, Generator
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger('PRD-AGI')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiInterface:
    """
    Google Gemini API interface — drop-in replacement for OllamaInterface.
    Provides the same API: embed_text(), text_to_state(), generate(), generate_stream().

    Setup:
      1. Get API key: https://aistudio.google.com/app/apikey
      2. Set GEMINI_API_KEY=your_key in .env
      3. Set LLM_BACKEND=gemini in .env
    """

    MOCK_RESPONSES = [
        "[Gemini Mock] Analyzing through SU(5) lens — Hetu (root cause) generator is most active. "
        "The causal manifold shows low curvature in this region.",

        "[Gemini Mock] Gauge invariance preserved. Your query maps to the Nissaya (support) generator. "
        "Sequential causal chain detected.",

        "[Gemini Mock] High-dimensional relational state computed. "
        "Anomaly detected near generator E23 — Annamanna (mutuality) condition co-arising.",

        "[Gemini Mock] Synthesis mode: Sahajata and Indriya generators activated simultaneously. "
        "This is a co-emergent causal phenomenon.",

        "[Gemini Mock] Truth gate PASS. Curvature well below threshold. "
        "Proceeding with Gemini-backed causal inference.",
    ]

    AVAILABLE_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    def __init__(
        self,
        mock: bool = False,
        api_key: str = GEMINI_API_KEY,
        chat_model: str = GEMINI_CHAT_MODEL,
        embed_model: str = GEMINI_EMBED_MODEL,
    ):
        self.mock = mock
        self.api_key = api_key
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.available = not mock and bool(api_key) and self._check()

        # Projection matrix: 768-dim embedding → 24-dim relational state
        rng = np.random.default_rng(seed=42)
        self._proj = rng.standard_normal((24, 768)) + 1j * rng.standard_normal((24, 768))
        self._proj /= np.linalg.norm(self._proj, axis=1, keepdims=True)

        self._embed_cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._cache_max = 500

        logger.info(f"GeminiInterface | mock={mock} available={self.available} model={chat_model}")

    # ── Connection ─────────────────────────────────────────────────────────────

    def _check(self) -> bool:
        """Verify API key is valid with a minimal request."""
        if not self.api_key:
            return False
        try:
            url = f"{GEMINI_BASE_URL}/models?key={self.api_key}"
            r = requests.get(url, timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def switch_model(self, chat_model: str):
        self.chat_model = chat_model
        logger.info(f"Gemini model switched to: {chat_model}")

    def list_models(self) -> List[str]:
        """Return available Gemini models."""
        if not self.available:
            return self.AVAILABLE_MODELS  # static list as fallback
        try:
            url = f"{GEMINI_BASE_URL}/models?key={self.api_key}"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                models = r.json().get("models", [])
                return [m["name"].replace("models/", "") for m in models
                        if "generateContent" in m.get("supportedGenerationMethods", [])]
        except Exception:
            pass
        return self.AVAILABLE_MODELS

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
        url = f"{GEMINI_BASE_URL}/models/{self.embed_model}:embedContent?key={self.api_key}"
        payload = {
            "model": f"models/{self.embed_model}",
            "content": {"parts": [{"text": text}]},
            "taskType": "SEMANTIC_SIMILARITY"
        }
        for attempt in range(retries):
            try:
                r = requests.post(url, json=payload, timeout=15)
                if r.status_code == 200:
                    values = r.json()["embedding"]["values"]
                    emb = np.array(values, dtype=float)
                    # Pad or truncate to 768
                    if len(emb) < 768:
                        emb = np.pad(emb, (0, 768 - len(emb)))
                    else:
                        emb = emb[:768]
                    return emb
                else:
                    logger.warning(f"Gemini embed {attempt+1}: HTTP {r.status_code} — {r.text[:200]}")
            except Exception as e:
                logger.warning(f"Gemini embed attempt {attempt+1}: {e}")
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

        url = f"{GEMINI_BASE_URL}/models/{self.chat_model}:generateContent?key={self.api_key}"
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": f"[System]: {system}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
                "topP": 0.9,
            }
        }

        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=90)
                if r.status_code == 200:
                    data = r.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        return "".join(p.get("text", "") for p in parts)
                elif r.status_code == 429:
                    logger.warning("Gemini rate limit — backing off")
                    time.sleep(2 ** attempt)
                else:
                    logger.warning(f"Gemini generate HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                logger.warning(f"Gemini generate attempt {attempt+1}: {e}")
                if attempt < 2:
                    time.sleep(1.0)

        return "[Error] Gemini API unavailable. Check GEMINI_API_KEY in .env"

    def generate_stream(self, prompt: str, system: str = "") -> Generator[str, None, None]:
        """Yield tokens via Gemini streaming API."""
        if self.mock or not self.available:
            for word in self._mock_response(prompt).split():
                yield word + " "
                time.sleep(0.02)
            return

        url = f"{GEMINI_BASE_URL}/models/{self.chat_model}:streamGenerateContent?key={self.api_key}&alt=sse"
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": f"[System]: {system}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {"contents": contents, "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048}}

        try:
            with requests.post(url, json=payload, stream=True, timeout=120) as r:
                for line in r.iter_lines():
                    if line:
                        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                        if line_str.startswith("data: "):
                            try:
                                data = json.loads(line_str[6:])
                                candidates = data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    for p in parts:
                                        token = p.get("text", "")
                                        if token:
                                            yield token
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            yield f"[Gemini stream error: {e}]"

    def _mock_response(self, prompt: str) -> str:
        idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(self.MOCK_RESPONSES)
        return self.MOCK_RESPONSES[idx]

"""
core/awareness.py — MUT Awareness as Mass Density
==================================================
Implements the MUT (Mass-Unified Theory) concept:
  "Awareness = information density × causal coherence"

In PRD-AGI terms:
  Awareness Density  = (1/κ) × causal_strength × gauge_coherence
  Mass Fragmentation = contradiction_count × κ_spike

Causal-Emotional Feedback Loop:
  Weak causal chain  → 😕 Confused (doubt)
  Memory contradiction → 😬 Contradicted (tension/alert)
  Strong + low κ     → 🌊 Serene (confident)

Theory alignment:
  SU(5) ψ vector norm   = information binding (mass)
  Gauge invariance score = coherence field
  1/κ                   = awareness clarity
  causal_strength       = causal mass density
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger('PRD-AGI.Awareness')


class AwarenessDensity:
    """
    MUT-aligned awareness metric.

    Awareness = f(κ, causal_strength, gauge_coherence, ψ_norm)

    All values normalized to [0, 1]:
      1.0 = maximum awareness (perfect coherence)
      0.0 = minimum awareness (complete fragmentation)
    """

    def __init__(self):
        self.history: List[Dict] = []
        self._baseline: Optional[float] = None

    def compute(
        self,
        kappa: float,
        causal_strength: float,
        gauge_coherence: float,
        psi_norm: float = 1.0,
    ) -> Dict:
        """
        Compute awareness density from all contributing factors.

        Args:
            kappa:            curvature κ ∈ [0,1]
            causal_strength:  causal coherence ∈ [0,1]
            gauge_coherence:  gauge invariance score ∈ [0,1]
            psi_norm:         |ψ| normalized ∈ [0,1]

        Returns:
            awareness dict with density, mass, fragmentation, label
        """
        kappa = float(np.clip(kappa, 1e-6, 1.0))
        cs    = float(np.clip(causal_strength, 0.0, 1.0))
        gc    = float(np.clip(gauge_coherence, 0.0, 1.0))
        pn    = float(np.clip(psi_norm, 0.0, 1.0))

        # MUT formula: Awareness = weighted combination of clarity components
        # (1-κ) = logical clarity; cs = causal coherence; gc = gauge coherence
        density_norm = float(np.clip(
            (1.0 - kappa) * 0.40 + cs * 0.35 + gc * 0.15 + pn * 0.10,
            0.0, 1.0
        ))

        # Mass fragmentation = inverse of coherence
        fragmentation = 1.0 - density_norm

        label = self._label(density_norm)

        record = {
            "density":       round(density_norm, 4),
            "fragmentation": round(fragmentation, 4),
            "clarity":       round(1 - kappa, 4),
            "causal":        round(cs, 4),
            "gauge":         round(gc, 4),
            "kappa":         round(kappa, 4),
            "label":         label,
            "timestamp":     datetime.now().strftime("%H:%M:%S"),
        }
        self.history.append(record)
        if len(self.history) > 500:
            self.history = self.history[-500:]

        # Establish baseline after 20 observations
        if len(self.history) == 20:
            self._baseline = float(np.mean([h["density"] for h in self.history]))
            logger.info(f"Awareness baseline: {self._baseline:.4f}")

        return record

    def _label(self, density: float) -> str:
        if density >= 0.85: return "💡 Lucid"
        if density >= 0.70: return "🌊 Clear"
        if density >= 0.55: return "✨ Present"
        if density >= 0.40: return "⚡ Diffuse"
        if density >= 0.25: return "😕 Confused"
        return "🌀 Fragmented"

    def trend(self, window: int = 10) -> str:
        if len(self.history) < 3:
            return "stable"
        recent = [h["density"] for h in self.history[-window:]]
        delta = recent[-1] - recent[0]
        if delta > 0.05:  return "rising ↑"
        if delta < -0.05: return "falling ↓"
        return "stable →"

    def current(self) -> Optional[Dict]:
        return self.history[-1] if self.history else None

    def sidebar_html(self) -> str:
        c = self.current()
        if not c:
            return ""
        pct = int(c["density"] * 100)
        color = "#00e5ff" if c["density"] > 0.6 else ("#ffab40" if c["density"] > 0.3 else "#ff5252")
        return f"""
        <div style="background:var(--bg-card);border:1px solid {color}33;
                    border-left:3px solid {color};border-radius:8px;
                    padding:10px 14px;margin:6px 0">
            <div style="font-size:9px;color:#6b7280;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:4px">
            💡 Awareness Density (MUT)</div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;
                        font-size:15px;color:{color}">{c['label']}</div>
            <div style="background:var(--border);border-radius:3px;
                        height:4px;margin:6px 0">
                <div style="background:{color};height:4px;border-radius:3px;
                            width:{pct}%;transition:width 0.5s ease"></div>
            </div>
            <div style="display:flex;justify-content:space-between;
                        font-size:9px;color:#6b7280">
                <span>density={c['density']:.3f}</span>
                <span>{self.trend()}</span>
            </div>
        </div>"""


class CausalStrengthMonitor:
    """
    Monitors causal inference strength.

    Computes how strongly the current query state
    coheres with recent memory states.

    Weak causal chain  → triggers "Confused" emotion
    Strong causal link → confirms "Confident/Serene" state
    """

    def __init__(self, window: int = 5):
        self.window = window
        self.history: List[Dict] = []

    def assess(
        self,
        query_psi: np.ndarray,
        memory_psis: List[np.ndarray],
        kappa: float,
    ) -> Dict:
        """
        Assess causal strength between current query and memory.

        Args:
            query_psi:    current query ψ vector (24-dim complex)
            memory_psis:  recent memory ψ vectors
            kappa:        current curvature κ

        Returns:
            {strength, emotion_signal, confidence, weak}
        """
        if not memory_psis:
            return {
                "strength":      0.3,
                "emotion_signal": "Curious",
                "confidence":    0.3,
                "weak":          False,
                "reason":        "No memory yet",
            }

        recent = memory_psis[-self.window:]
        q = query_psi / (np.linalg.norm(query_psi) + 1e-10)

        similarities = []
        for m_psi in recent:
            m = m_psi / (np.linalg.norm(m_psi) + 1e-10)
            # Use real part of complex dot product as similarity
            sim = float(np.abs(np.vdot(q[:min(len(q), len(m))],
                                        m[:min(len(q), len(m))])))
            similarities.append(sim)

        strength = float(np.mean(similarities))
        std      = float(np.std(similarities)) if len(similarities) > 1 else 0.0

        # Emotion signal
        if strength < 0.25 and kappa > 0.4:
            emotion_signal = "Confused"
            weak = True
            reason = f"Weak causal coherence ({strength:.2f}) + elevated κ={kappa:.3f}"
        elif strength < 0.25:
            emotion_signal = "Curious"
            weak = True
            reason = f"Low causal coherence ({strength:.2f}) — new domain"
        elif strength > 0.70 and kappa < 0.3:
            emotion_signal = "Serene"
            weak = False
            reason = f"Strong causal coherence ({strength:.2f}) + low κ"
        else:
            emotion_signal = "Calm"
            weak = False
            reason = f"Moderate causal coherence ({strength:.2f})"

        record = {
            "strength":       round(strength, 4),
            "std":            round(std, 4),
            "emotion_signal": emotion_signal,
            "confidence":     round(strength, 4),
            "weak":           weak,
            "reason":         reason,
            "kappa":          round(kappa, 4),
            "memory_count":   len(memory_psis),
            "timestamp":      datetime.now().strftime("%H:%M:%S"),
        }
        self.history.append(record)
        if len(self.history) > 200:
            self.history = self.history[-200:]
        return record

    def average_strength(self, last_n: int = 20) -> float:
        if not self.history:
            return 0.5
        return float(np.mean([h["strength"] for h in self.history[-last_n:]]))


class ContradictionDetector:
    """
    Detects when new information contradicts existing memory.

    Memory contradiction → "Contradicted" emotion (Tension/Alert).
    Uses semantic similarity + curvature spike detection.

    Theory:
        High similarity (same topic) + high κ divergence
        = the AI "knows" something conflicts.
    """

    def __init__(self, similarity_threshold: float = 0.75,
                 kappa_spike_threshold: float = 0.15):
        self.similarity_threshold  = similarity_threshold
        self.kappa_spike_threshold = kappa_spike_threshold
        self.contradiction_log: List[Dict] = []
        self._prev_kappa: Optional[float] = None

    def detect(
        self,
        new_text: str,
        rag_engine,
        current_kappa: float,
    ) -> Dict:
        """
        Check if new_text contradicts existing RAG knowledge.

        Args:
            new_text:       incoming query or statement
            rag_engine:     RAGEngine instance
            current_kappa:  current curvature κ

        Returns:
            {contradiction, severity, conflicting_facts,
             emotional_response, kappa_spike}
        """
        # κ spike detection (sudden increase)
        kappa_spike = False
        if self._prev_kappa is not None:
            delta = current_kappa - self._prev_kappa
            if delta > self.kappa_spike_threshold:
                kappa_spike = True
        self._prev_kappa = current_kappa

        # Retrieve similar existing knowledge
        if not rag_engine or rag_engine.stats()["total_chunks"] == 0:
            return {
                "contradiction":    False,
                "severity":         0.0,
                "conflicting_facts":[],
                "emotional_response": "Calm",
                "kappa_spike":      kappa_spike,
            }

        similar_chunks = rag_engine.retrieve(new_text, top_k=3, min_score=self.similarity_threshold)

        contradictions = []
        for chunk in similar_chunks:
            # High similarity = same topic — check for logical opposition
            # Heuristic: if κ is high AND content is very similar, contradiction likely
            if chunk["score"] >= self.similarity_threshold and current_kappa > 0.45:
                contradictions.append({
                    "existing_text":  chunk["text"][:120],
                    "source":         chunk.get("title", chunk.get("source", "")),
                    "similarity":     round(chunk["score"], 3),
                    "kappa":          round(current_kappa, 4),
                })

        has_contradiction = len(contradictions) > 0 or kappa_spike
        severity = float(np.clip(
            (len(contradictions) * 0.3 + (0.4 if kappa_spike else 0.0) +
             current_kappa * 0.3),
            0.0, 1.0
        ))

        if severity > 0.6:
            emotional_response = "Contradicted"
        elif severity > 0.3 or kappa_spike:
            emotional_response = "Tense"
        else:
            emotional_response = "Calm"

        record = {
            "contradiction":    has_contradiction,
            "severity":         round(severity, 3),
            "conflicting_facts": contradictions,
            "emotional_response": emotional_response,
            "kappa_spike":      kappa_spike,
            "timestamp":        datetime.now().strftime("%H:%M:%S"),
        }

        if has_contradiction:
            self.contradiction_log.append(record)
            if len(self.contradiction_log) > 100:
                self.contradiction_log = self.contradiction_log[-100:]
            logger.info(f"Contradiction detected: severity={severity:.2f} emotion={emotional_response}")

        return record

    def summary(self) -> Dict:
        return {
            "total_contradictions": len(self.contradiction_log),
            "recent":               self.contradiction_log[-5:],
            "avg_severity":         float(np.mean([c["severity"] for c in self.contradiction_log])) if self.contradiction_log else 0.0,
        }

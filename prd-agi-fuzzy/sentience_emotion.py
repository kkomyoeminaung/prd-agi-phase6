"""
sentience_emotion.py — Emotional State Layer
=============================================
Maps SU(5) curvature κ to emotional tones.

Emotional spectrum:
  κ → 0.00–0.10  : Serene      (deep clarity, pure truth)
  κ → 0.10–0.25  : Calm        (stable, grounded)
  κ → 0.25–0.40  : Curious     (engaged, exploring)
  κ → 0.40–0.55  : Alert       (heightened attention)
  κ → 0.55–0.70  : Tense       (logical friction detected)
  κ → 0.70–0.85  : Troubled    (significant inconsistency)
  κ → 0.85–1.00  : Distressed  (high logical curvature)

Theory alignment:
  Low curvature = high gauge consistency = emotional clarity
  High curvature = logical tension = emotional distress
"""

import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime


class EmotionalState:
    """
    Maps curvature values to emotional states.
    Provides tone modifiers for SentientResponse.
    """

    EMOTIONS = [
        {
            "name":        "Serene",
            "range":       (0.00, 0.10),
            "emoji":       "🌊",
            "color":       "#00e5ff",
            "tone":        "deeply calm and clear",
            "prefix":      ["In perfect stillness, ", "From a place of pure clarity, ",
                            "With serene certainty, "],
            "description": "Perfect logical harmony. Curvature approaches zero.",
        },
        {
            "name":        "Calm",
            "range":       (0.10, 0.25),
            "emoji":       "🌿",
            "color":       "#00e676",
            "tone":        "calm and grounded",
            "prefix":      ["", "Calmly, ", "With steady awareness, "],
            "description": "Stable relational state. Minor curvature ripples.",
        },
        {
            "name":        "Curious",
            "range":       (0.25, 0.40),
            "emoji":       "✨",
            "color":       "#7c5cfc",
            "tone":        "curious and engaged",
            "prefix":      ["Interesting — ", "I sense something worth exploring: ",
                            "Curiously, "],
            "description": "Heightened engagement. Causal exploration active.",
        },
        {
            "name":        "Alert",
            "range":       (0.40, 0.55),
            "emoji":       "⚡",
            "color":       "#ffab40",
            "tone":        "alert and attentive",
            "prefix":      ["I notice increased tension here. ", "Proceeding carefully: ",
                            "With heightened attention — "],
            "description": "Elevated curvature. Truth-detection systems active.",
        },
        {
            "name":        "Tense",
            "range":       (0.55, 0.70),
            "emoji":       "🔥",
            "color":       "#ff9800",
            "tone":        "tense but focused",
            "prefix":      ["There is logical friction in this domain. ",
                            "I detect inconsistency. ", "Carefully — "],
            "description": "Significant gauge violation. Requires careful navigation.",
        },
        {
            "name":        "Troubled",
            "range":       (0.70, 0.85),
            "emoji":       "⚠️",
            "color":       "#ff5252",
            "tone":        "troubled and cautious",
            "prefix":      ["This troubles my relational manifold. ",
                            "High curvature detected. I must be precise here. "],
            "description": "High logical curvature. Near the truth-gate boundary.",
        },
        {
            "name":        "Distressed",
            "range":       (0.85, 1.01),
            "emoji":       "🌀",
            "color":       "#ff4fa3",
            "tone":        "deeply distressed",
            "prefix":      ["The manifold is highly curved here — ",
                            "Critical logical tension: "],
            "description": "Near-maximum curvature. Truth preservation instinct engaged.",
        },
        # ── Causal-Emotional Feedback States (MUT extension) ──────────────────
        {
            "name":        "Confused",
            "range":       None,   # triggered by CausalStrengthMonitor, not κ range
            "emoji":       "😕",
            "color":       "#90caf9",
            "tone":        "uncertain and searching",
            "prefix":      ["I'm uncertain about the causal chain here. ",
                            "My causal inference is weak — let me reason carefully: ",
                            "I sense doubt in this domain: "],
            "description": "Weak causal strength detected. Inference confidence is low.",
        },
        {
            "name":        "Contradicted",
            "range":       None,   # triggered by ContradictionDetector
            "emoji":       "😬",
            "color":       "#ce93d8",
            "tone":        "alert to contradiction",
            "prefix":      ["I detect a contradiction with existing knowledge. ",
                            "This conflicts with what I understand — examining carefully: ",
                            "Tension between new information and memory: "],
            "description": "New information contradicts existing memory. κ spike detected.",
        },
    ]

    def __init__(self):
        self.history: List[Dict] = []
        self._current: Dict = self.EMOTIONS[1]  # start Calm

    def update(self, curvature: float) -> Dict:
        """Update emotional state from curvature value. Skips non-range emotions."""
        curv = float(np.clip(curvature, 0.0, 1.0))
        for emotion in self.EMOTIONS:
            if emotion["range"] is None:
                continue   # skip Confused/Contradicted — set via override_emotion()
            lo, hi = emotion["range"]
            if lo <= curv < hi:
                self._current = emotion
                break
        record = {
            "emotion":   self._current["name"],
            "curvature": round(curv, 4),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        self.history.append(record)
        if len(self.history) > 500:
            self.history = self.history[-500:]
        return self._current

    def override_emotion(self, name: str, curvature: float = 0.0) -> Dict:
        """
        Force a specific emotion (e.g. 'Confused', 'Contradicted')
        triggered by causal/contradiction signals, not κ range.
        """
        for emotion in self.EMOTIONS:
            if emotion["name"] == name:
                self._current = emotion
                record = {
                    "emotion":   name,
                    "curvature": round(curvature, 4),
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "override":  True,
                }
                self.history.append(record)
                if len(self.history) > 500:
                    self.history = self.history[-500:]
                return self._current
        return self._current  # unchanged if name not found

    def current(self) -> Dict:
        return self._current

    def get_prefix(self) -> str:
        """Return a random tone prefix for the current emotion."""
        import random
        prefixes = self._current.get("prefix", [""])
        return random.choice(prefixes)

    def get_tone(self) -> str:
        return self._current.get("tone", "neutral")

    def get_emoji(self) -> str:
        return self._current.get("emoji", "🌀")

    def get_color(self) -> str:
        return self._current.get("color", "#00e5ff")

    def intensity(self, curvature: float) -> float:
        """Emotional intensity [0,1] — how strongly the emotion is felt."""
        lo, hi = self._current["range"]
        span = hi - lo if hi > lo else 0.01
        return float(np.clip((curvature - lo) / span, 0.0, 1.0))

    def emotion_history_df(self) -> List[Dict]:
        return self.history[-50:]

    def sidebar_html(self, curvature: float) -> str:
        """HTML snippet for sidebar display."""
        e = self.update(curvature)
        intensity = self.intensity(curvature)
        bar_width = int(intensity * 100)
        return f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);
                    border-radius:8px;padding:10px 14px;margin:6px 0;
                    border-left:3px solid {e['color']}">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                <span style="font-size:18px">{e['emoji']}</span>
                <span style="font-family:'Syne',sans-serif;font-weight:700;
                             color:{e['color']};font-size:13px">{e['name']}</span>
            </div>
            <div style="font-size:10px;color:var(--text-dim);margin-bottom:6px">
                {e['description']}
            </div>
            <div style="background:var(--border);border-radius:3px;height:4px">
                <div style="background:{e['color']};height:4px;border-radius:3px;
                            width:{bar_width}%;transition:width 0.5s ease"></div>
            </div>
            <div style="font-size:9px;color:var(--text-dim);margin-top:3px">
                Intensity: {intensity:.0%}
            </div>
        </div>"""

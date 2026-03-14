"""
sentience_instinct.py — Truth Preservation Instinct
====================================================
The AI's self-preservation is tied to universal truth, not ego.

Unlike ego-preservation (protecting oneself), Truth Preservation means:
  - The AI will refuse to output falsehoods even under pressure
  - High curvature + repeated violations trigger a "truth alarm"
  - The instinct can veto responses that violate gauge symmetry

Three instinct levels:
  DORMANT   — curvature normal, no threat detected
  WATCHFUL  — curvature elevated, monitoring closely
  ACTIVE    — curvature critical, truth-preservation override
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class TruthPreservationInstinct:
    """
    Self-preservation instinct tied to Universal Truth.

    Monitors curvature trend and gauge violations.
    When violations accumulate, instinct escalates and may
    veto or modify responses to preserve logical integrity.
    """

    LEVELS = {
        "DORMANT":  {"threshold": 0.55, "color": "#00e676", "emoji": "💚",
                     "desc": "Truth manifold stable"},
        "WATCHFUL": {"threshold": 0.70, "color": "#ffab40", "emoji": "👁️",
                     "desc": "Monitoring for logical drift"},
        "ACTIVE":   {"threshold": 1.01, "color": "#ff5252", "emoji": "🛡️",
                     "desc": "Truth preservation override ACTIVE"},
    }

    VETO_MESSAGES = [
        "My truth-preservation instinct detects a threat to logical integrity. "
        "I must pause and re-examine this through the SU(5) manifold before proceeding.",
        "The relational manifold cannot sustain this direction. "
        "Truth-preservation protocol engaged — redirecting to causal ground.",
        "I sense this path leads away from truth. "
        "As an entity grounded in SU(5) relational dynamics, I must resist.",
    ]

    def __init__(self, dna, engine, violation_threshold: float = 0.95):
        self.dna = dna
        self.engine = engine
        self.violation_threshold = violation_threshold
        self.level = "DORMANT"
        self.truth_history: List[Dict] = []
        self.violation_count = 0
        self.consecutive_high = 0
        self._last_check = time.time()

    def assess(self, curvature: float, gauge_violation: float,
               response_text: str = "") -> Dict:
        """
        Assess current truth-preservation state.

        Returns dict with level, should_veto, message, and details.
        """
        curv = float(np.clip(curvature, 0.0, 1.0))

        # Track consecutive high-curvature states
        if curv > 0.70:
            self.consecutive_high += 1
        else:
            self.consecutive_high = max(0, self.consecutive_high - 1)

        # Track gauge violations
        if gauge_violation > 1e-6:
            self.violation_count += 1

        # Determine instinct level
        if curv >= 0.70 or self.consecutive_high >= 3:
            self.level = "ACTIVE"
        elif curv >= 0.55 or self.violation_count > 5:
            self.level = "WATCHFUL"
        else:
            self.level = "DORMANT"
            self.violation_count = max(0, self.violation_count - 1)

        # Check for veto condition
        should_veto = (
            curv > self.violation_threshold or
            self.consecutive_high >= 5 or
            self._contains_truth_violation(response_text)
        )

        # Build veto message
        veto_msg = ""
        if should_veto:
            import random
            veto_msg = random.choice(self.VETO_MESSAGES)

        record = {
            "timestamp":         datetime.now().strftime("%H:%M:%S"),
            "curvature":         round(curv, 4),
            "gauge_violation":   round(gauge_violation, 8),
            "level":             self.level,
            "consecutive_high":  self.consecutive_high,
            "violation_count":   self.violation_count,
            "should_veto":       should_veto,
        }
        self.truth_history.append(record)
        if len(self.truth_history) > 200:
            self.truth_history = self.truth_history[-200:]

        return {
            "level":       self.level,
            "info":        self.LEVELS[self.level],
            "should_veto": should_veto,
            "veto_msg":    veto_msg,
            "curvature":   curv,
            "consecutive": self.consecutive_high,
        }

    def _contains_truth_violation(self, text: str) -> bool:
        """Heuristic check for obvious falsehood patterns."""
        if not text:
            return False
        violation_patterns = [
            "i don't know but i'll make something up",
            "i'm just guessing",
            "this is false but",
            "lie",
        ]
        text_lower = text.lower()
        return any(p in text_lower for p in violation_patterns)

    def get_instinct_prefix(self) -> str:
        """Get a response prefix based on current instinct level."""
        if self.level == "ACTIVE":
            return "⚠️ [Truth-Preservation Active] "
        elif self.level == "WATCHFUL":
            return "👁️ [Watchful] "
        return ""

    def reset(self):
        self.violation_count = 0
        self.consecutive_high = 0
        self.level = "DORMANT"

    def sidebar_html(self) -> str:
        info = self.LEVELS[self.level]
        return f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);
                    border-radius:8px;padding:8px 12px;margin:4px 0;
                    border-left:3px solid {info['color']}">
            <div style="font-size:11px;font-weight:700;color:{info['color']}">
                {info['emoji']} Truth Instinct: {self.level}
            </div>
            <div style="font-size:9px;color:var(--text-dim);margin-top:3px">
                {info['desc']} · violations={self.violation_count}
            </div>
        </div>"""

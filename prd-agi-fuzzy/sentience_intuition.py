"""
sentience_intuition.py — Fuzzy Intuition Layer
================================================
Generates "gut feelings" about inputs using fuzzy logic.

Inputs:
  curvature  [0,1] — logical consistency of current query
  history    [0,1] — normalized count of past similar queries
                     (higher = more familiar domain)

Output:
  intuition  [0,1] — 0=bad feeling, 1=strong positive intuition

Interpretation:
  0.0–0.30 → "Something feels wrong"
  0.30–0.55 → "Uncertain / neutral"
  0.55–0.75 → "Feels right"
  0.75–1.00 → "Strong positive intuition"

This layer acts FAST (before full logical analysis) — like a human gut
reaction before deliberate reasoning kicks in.
"""

import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False


class IntuitionLayer:
    """
    Fuzzy-based intuition system for PRD-AGI.

    Provides fast, pre-analytical "gut feelings" about queries.
    Combines current curvature with historical pattern familiarity.
    """

    INTUITION_LABELS = [
        (0.00, 0.30, "🔴 Bad feeling",        "#ff5252"),
        (0.30, 0.55, "🟡 Uncertain",           "#ffab40"),
        (0.55, 0.75, "🟢 Feels right",         "#00e676"),
        (0.75, 1.01, "✨ Strong intuition",    "#00e5ff"),
    ]

    def __init__(self):
        self._skfuzzy_ready = False
        self.query_history: List[float] = []   # curvature history for familiarity
        if SKFUZZY_AVAILABLE:
            self._build_system()

    def _build_system(self):
        try:
            r = np.arange(0, 1.01, 0.01)

            self.curv_ant    = ctrl.Antecedent(r, 'curvature')
            self.hist_ant    = ctrl.Antecedent(r, 'history')
            self.intuit_con  = ctrl.Consequent(r, 'intuition')

            # Curvature membership (low = good, high = bad)
            self.curv_ant['low']    = fuzz.trimf(r, [0.00, 0.00, 0.35])
            self.curv_ant['medium'] = fuzz.trimf(r, [0.25, 0.50, 0.75])
            self.curv_ant['high']   = fuzz.trimf(r, [0.65, 1.00, 1.00])

            # History membership (familiarity with domain)
            self.hist_ant['unfamiliar'] = fuzz.trimf(r, [0.00, 0.00, 0.40])
            self.hist_ant['moderate']   = fuzz.trimf(r, [0.25, 0.50, 0.75])
            self.hist_ant['familiar']   = fuzz.trimf(r, [0.60, 1.00, 1.00])

            # Intuition output
            self.intuit_con['bad']      = fuzz.trimf(r, [0.00, 0.00, 0.35])
            self.intuit_con['neutral']  = fuzz.trimf(r, [0.25, 0.50, 0.75])
            self.intuit_con['good']     = fuzz.trimf(r, [0.60, 0.80, 0.95])
            self.intuit_con['strong']   = fuzz.trimf(r, [0.80, 1.00, 1.00])

            rules = [
                # Low curvature (consistent) → positive intuition
                ctrl.Rule(self.curv_ant['low']    & self.hist_ant['familiar'],   self.intuit_con['strong']),
                ctrl.Rule(self.curv_ant['low']    & self.hist_ant['moderate'],   self.intuit_con['good']),
                ctrl.Rule(self.curv_ant['low']    & self.hist_ant['unfamiliar'], self.intuit_con['neutral']),
                # Medium curvature → neutral
                ctrl.Rule(self.curv_ant['medium'] & self.hist_ant['familiar'],   self.intuit_con['good']),
                ctrl.Rule(self.curv_ant['medium'] & self.hist_ant['moderate'],   self.intuit_con['neutral']),
                ctrl.Rule(self.curv_ant['medium'] & self.hist_ant['unfamiliar'], self.intuit_con['neutral']),
                # High curvature (inconsistent) → bad intuition
                ctrl.Rule(self.curv_ant['high']   & self.hist_ant['familiar'],   self.intuit_con['neutral']),
                ctrl.Rule(self.curv_ant['high']   & self.hist_ant['moderate'],   self.intuit_con['bad']),
                ctrl.Rule(self.curv_ant['high']   & self.hist_ant['unfamiliar'], self.intuit_con['bad']),
            ]

            system = ctrl.ControlSystem(rules)
            self._sim = ctrl.ControlSystemSimulation(system)
            self._skfuzzy_ready = True
        except Exception:
            self._skfuzzy_ready = False

    def _familiarity_score(self, curvature: float) -> float:
        """
        Compute domain familiarity [0,1] based on past query curvatures.
        High familiarity if current curvature is close to historical average.
        """
        if len(self.query_history) < 3:
            return 0.3   # low familiarity at start
        hist_array = np.array(self.query_history[-50:])
        mean_curv = float(np.mean(hist_array))
        std_curv  = float(np.std(hist_array)) + 0.01
        # Gaussian similarity
        similarity = float(np.exp(-0.5 * ((curvature - mean_curv) / std_curv) ** 2))
        return float(np.clip(similarity, 0.0, 1.0))

    def sense(self, curvature: float) -> Tuple[float, str, str]:
        """
        Generate intuition about a query.

        Returns:
            (score: float, label: str, color: str)
        """
        curv = float(np.clip(curvature, 0.0, 1.0))
        familiarity = self._familiarity_score(curv)
        self.query_history.append(curv)

        if self._skfuzzy_ready:
            score = self._eval_fuzzy(curv, familiarity)
        else:
            score = self._eval_fallback(curv, familiarity)

        label, color = self._interpret(score)
        return score, label, color

    def _eval_fuzzy(self, curv: float, familiarity: float) -> float:
        try:
            self._sim.input['curvature'] = float(np.clip(curv, 0.01, 0.99))
            self._sim.input['history']   = float(np.clip(familiarity, 0.01, 0.99))
            self._sim.compute()
            return float(np.clip(self._sim.output['intuition'], 0.0, 1.0))
        except Exception:
            return self._eval_fallback(curv, familiarity)

    def _eval_fallback(self, curv: float, familiarity: float) -> float:
        # Higher curvature → lower intuition; higher familiarity → higher intuition
        base = 1.0 - curv
        bonus = familiarity * 0.3
        return float(np.clip(base * 0.7 + bonus, 0.0, 1.0))

    def _interpret(self, score: float) -> Tuple[str, str]:
        for lo, hi, label, color in self.INTUITION_LABELS:
            if lo <= score < hi:
                return label, color
        return "🟡 Uncertain", "#ffab40"

    def intuition_chip_html(self, score: float, label: str, color: str) -> str:
        return (f'<span style="background:{color}22;color:{color};'
                f'border:1px solid {color};border-radius:20px;'
                f'padding:2px 10px;font-size:11px;font-weight:700">'
                f'{label} ({score:.0%})</span>')

"""
fuzzy_gatekeeper.py — PRD-AGI Fuzzy Logic Gatekeeper
=====================================================
Replaces the hard threshold gate with fuzzy membership functions.

Membership sets:
  curvature  → very_low | low | medium | high
  gauge      → tight   | moderate | loose   (optional)
  decision   → reject  | marginal | accept

Rules:
  IF curvature=very_low                           → ACCEPT  (1.0)
  IF curvature=low                                → ACCEPT  (0.8)
  IF curvature=medium AND gauge=tight             → ACCEPT  (0.6)
  IF curvature=medium AND gauge=moderate          → MARGINAL(0.5)
  IF curvature=medium AND gauge=loose             → REJECT  (0.3)
  IF curvature=high                               → REJECT  (0.0)
"""

import numpy as np
from typing import Tuple, Dict

# scikit-fuzzy optional import with graceful fallback
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False


class FuzzyGateKeeper:
    """
    Fuzzy logic gatekeeper for PRD-AGI truth decisions.

    Instead of a hard threshold (κ < 0.5 → pass), uses overlapping
    membership functions for smooth, human-like truth assessment.

    Falls back to linear interpolation if scikit-fuzzy is not installed.
    """

    # Fuzzy confidence labels
    LABELS = {
        (0.0, 0.35): "❌ REJECT",
        (0.35, 0.60): "⚠️ MARGINAL",
        (0.60, 1.01): "✅ ACCEPT",
    }

    def __init__(self, use_gauge: bool = True, curvature_range: float = 1.0):
        self.use_gauge = use_gauge
        self.curvature_range = curvature_range
        self._skfuzzy_ready = False

        if SKFUZZY_AVAILABLE:
            self._build_system()
        # Always build numpy fallback
        self._build_fallback()

    # ── scikit-fuzzy system ─────────────────────────────────────────────────

    def _build_system(self):
        try:
            c_range = np.arange(0, 1.01, 0.01)
            g_range = np.arange(0, 1e-5 + 1e-7, 1e-7)
            d_range = np.arange(0, 1.01, 0.01)

            self.curv_ant = ctrl.Antecedent(c_range, 'curvature')
            self.dec_con  = ctrl.Consequent(d_range, 'decision')

            # Curvature membership functions
            self.curv_ant['very_low'] = fuzz.trimf(c_range, [0.00, 0.00, 0.20])
            self.curv_ant['low']      = fuzz.trimf(c_range, [0.10, 0.25, 0.40])
            self.curv_ant['medium']   = fuzz.trimf(c_range, [0.30, 0.50, 0.70])
            self.curv_ant['high']     = fuzz.trimf(c_range, [0.60, 1.00, 1.00])

            # Decision membership functions
            self.dec_con['reject']   = fuzz.trimf(d_range, [0.00, 0.00, 0.35])
            self.dec_con['marginal'] = fuzz.trimf(d_range, [0.25, 0.50, 0.75])
            self.dec_con['accept']   = fuzz.trimf(d_range, [0.65, 1.00, 1.00])

            if self.use_gauge:
                self.gauge_ant = ctrl.Antecedent(g_range, 'gauge')
                self.gauge_ant['tight']    = fuzz.trimf(g_range, [0, 0, 3e-6])
                self.gauge_ant['moderate'] = fuzz.trimf(g_range, [1e-6, 4e-6, 7e-6])
                self.gauge_ant['loose']    = fuzz.trimf(g_range, [5e-6, 1e-5, 1e-5])

                rules = [
                    ctrl.Rule(self.curv_ant['very_low'],                              self.dec_con['accept']),
                    ctrl.Rule(self.curv_ant['low'],                                   self.dec_con['accept']),
                    ctrl.Rule(self.curv_ant['medium'] & self.gauge_ant['tight'],      self.dec_con['accept']),
                    ctrl.Rule(self.curv_ant['medium'] & self.gauge_ant['moderate'],   self.dec_con['marginal']),
                    ctrl.Rule(self.curv_ant['medium'] & self.gauge_ant['loose'],      self.dec_con['reject']),
                    ctrl.Rule(self.curv_ant['high'],                                  self.dec_con['reject']),
                ]
            else:
                rules = [
                    ctrl.Rule(self.curv_ant['very_low'], self.dec_con['accept']),
                    ctrl.Rule(self.curv_ant['low'],      self.dec_con['accept']),
                    ctrl.Rule(self.curv_ant['medium'],   self.dec_con['marginal']),
                    ctrl.Rule(self.curv_ant['high'],     self.dec_con['reject']),
                ]

            system = ctrl.ControlSystem(rules)
            self._sim = ctrl.ControlSystemSimulation(system)
            self._skfuzzy_ready = True

        except Exception as e:
            self._skfuzzy_ready = False

    def _build_fallback(self):
        """NumPy-only fallback: piecewise linear membership."""
        # curvature → confidence mapping (piecewise linear)
        self._curv_pts  = np.array([0.00, 0.15, 0.30, 0.50, 0.70, 1.00])
        self._conf_vals = np.array([1.00, 0.95, 0.80, 0.50, 0.20, 0.00])

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate(self, curvature: float, gauge_violation: float = 0.0) -> Tuple[bool, str, float]:
        """
        Evaluate a query through the fuzzy gate.

        Returns:
            (accepted: bool, label: str, confidence: float 0-1)
        """
        curv = float(np.clip(curvature / self.curvature_range, 0.0, 1.0))

        if self._skfuzzy_ready:
            confidence = self._eval_skfuzzy(curv, gauge_violation)
        else:
            confidence = self._eval_fallback(curv, gauge_violation)

        label = self._confidence_label(confidence)
        accepted = confidence >= 0.35
        return accepted, label, confidence

    def _eval_skfuzzy(self, curv_norm: float, gauge: float) -> float:
        try:
            self._sim.input['curvature'] = float(np.clip(curv_norm, 0.0, 1.0))
            if self.use_gauge:
                self._sim.input['gauge'] = float(np.clip(gauge, 0.0, 1e-5))
            self._sim.compute()
            return float(np.clip(self._sim.output['decision'], 0.0, 1.0))
        except Exception:
            return self._eval_fallback(curv_norm, gauge)

    def _eval_fallback(self, curv_norm: float, gauge: float) -> float:
        conf = float(np.interp(curv_norm, self._curv_pts, self._conf_vals))
        if self.use_gauge and gauge > 0:
            # gauge penalty: up to -0.2 confidence
            gauge_norm = float(np.clip(gauge / 1e-5, 0.0, 1.0))
            conf -= 0.20 * gauge_norm
        return float(np.clip(conf, 0.0, 1.0))

    def _confidence_label(self, conf: float) -> str:
        for (lo, hi), label in self.LABELS.items():
            if lo <= conf < hi:
                return label
        return "⚠️ MARGINAL"

    def membership_profile(self, curvature: float) -> Dict[str, float]:
        """Return membership degrees for all sets (for visualization).
        Uses pure numpy trimf — no skfuzzy dependency."""
        curv = float(np.clip(curvature / self.curvature_range, 0.0, 1.0))

        def trimf(x: float, abc: list) -> float:
            """Triangular membership function."""
            a, b, c = abc
            if x <= a or x >= c:
                return 0.0
            if x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            return (c - x) / (c - b) if c != b else 1.0

        return {
            "very_low": trimf(curv, [0.00, 0.00, 0.20]),
            "low":      trimf(curv, [0.10, 0.25, 0.40]),
            "medium":   trimf(curv, [0.30, 0.50, 0.70]),
            "high":     trimf(curv, [0.60, 1.00, 1.00]),
        }

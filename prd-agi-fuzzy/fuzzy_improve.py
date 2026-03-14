"""
fuzzy_improve.py — Fuzzy Improvement Decision Engine
=====================================================
Decides whether to continue self-improving code based on:
  - Current quality score
  - Rate of improvement (delta quality per iteration)

Rules:
  IF quality=high   AND improvement=positive  → CONTINUE (strongly)
  IF quality=high   AND improvement=stagnant  → STOP     (good enough)
  IF quality=medium AND improvement=positive  → CONTINUE
  IF quality=medium AND improvement=stagnant  → MARGINAL
  IF quality=low    AND improvement=positive  → CONTINUE
  IF quality=low    AND improvement=negative  → STOP     (worsening)
"""

import numpy as np
from typing import Tuple, List, Dict

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False


class FuzzyImprovementDecider:
    """
    Fuzzy engine that decides whether to continue code improvement iterations.

    Used in the Self-Programming tab to intelligently stop
    when quality plateaus or is already excellent.
    """

    def __init__(self):
        self._skfuzzy_ready = False
        if SKFUZZY_AVAILABLE:
            self._build_system()
        self._history: List[Dict] = []

    def _build_system(self):
        try:
            q_range = np.arange(0, 1.01, 0.01)
            i_range = np.arange(-0.5, 0.51, 0.01)
            d_range = np.arange(0, 1.01, 0.01)

            self.quality     = ctrl.Antecedent(q_range, 'quality')
            self.improvement = ctrl.Antecedent(i_range, 'improvement')
            self.decision    = ctrl.Consequent(d_range, 'decision')

            # Quality membership
            self.quality['low']    = fuzz.trimf(q_range, [0.00, 0.00, 0.40])
            self.quality['medium'] = fuzz.trimf(q_range, [0.25, 0.50, 0.75])
            self.quality['high']   = fuzz.trimf(q_range, [0.60, 1.00, 1.00])

            # Improvement rate membership
            self.improvement['negative'] = fuzz.trimf(i_range, [-0.50, -0.50, -0.05])
            self.improvement['stagnant'] = fuzz.trimf(i_range, [-0.10,  0.00,  0.10])
            self.improvement['positive'] = fuzz.trimf(i_range, [ 0.05,  0.50,  0.50])

            # Decision membership
            self.decision['stop']     = fuzz.trimf(d_range, [0.00, 0.00, 0.35])
            self.decision['marginal'] = fuzz.trimf(d_range, [0.25, 0.50, 0.75])
            self.decision['continue'] = fuzz.trimf(d_range, [0.65, 1.00, 1.00])

            rules = [
                # Strong continue
                ctrl.Rule(self.quality['low']    & self.improvement['positive'], self.decision['continue']),
                ctrl.Rule(self.quality['medium'] & self.improvement['positive'], self.decision['continue']),
                ctrl.Rule(self.quality['high']   & self.improvement['positive'], self.decision['continue']),
                # Marginal
                ctrl.Rule(self.quality['low']    & self.improvement['stagnant'], self.decision['marginal']),
                ctrl.Rule(self.quality['medium'] & self.improvement['stagnant'], self.decision['marginal']),
                # Stop
                ctrl.Rule(self.quality['high']   & self.improvement['stagnant'], self.decision['stop']),
                ctrl.Rule(self.quality['low']    & self.improvement['negative'], self.decision['stop']),
                ctrl.Rule(self.quality['medium'] & self.improvement['negative'], self.decision['stop']),
                ctrl.Rule(self.quality['high']   & self.improvement['negative'], self.decision['stop']),
            ]

            system = ctrl.ControlSystem(rules)
            self._sim = ctrl.ControlSystemSimulation(system)
            self._skfuzzy_ready = True
        except Exception:
            self._skfuzzy_ready = False

    def should_continue(self, quality: float, prev_quality: float) -> Tuple[bool, str, float]:
        """
        Decide whether to run another improvement iteration.

        Args:
            quality:      current quality score [0,1]
            prev_quality: quality from previous iteration

        Returns:
            (should_continue: bool, reason: str, confidence: float)
        """
        improvement = float(np.clip(quality - prev_quality, -0.5, 0.5))
        q_norm = float(np.clip(quality, 0.0, 1.0))

        decision_score = self._compute(q_norm, improvement)

        # Record history
        self._history.append({
            "quality": quality,
            "prev_quality": prev_quality,
            "improvement": improvement,
            "decision_score": decision_score,
        })

        if decision_score >= 0.65:
            return True,  f"🔄 Continue — improving (+{improvement:.3f})", decision_score
        elif decision_score >= 0.35:
            return True,  f"⚠️ Marginal — slight improvement ({improvement:+.3f})", decision_score
        else:
            if quality >= 0.80:
                reason = f"✅ Stop — quality excellent ({quality:.3f})"
            elif improvement < 0:
                reason = f"🛑 Stop — quality declining ({improvement:.3f})"
            else:
                reason = f"⏹️ Stop — stagnated at {quality:.3f}"
            return False, reason, decision_score

    def _compute(self, quality: float, improvement: float) -> float:
        if self._skfuzzy_ready:
            try:
                self._sim.input['quality']     = quality
                self._sim.input['improvement'] = improvement
                self._sim.compute()
                return float(np.clip(self._sim.output['decision'], 0.0, 1.0))
            except Exception:
                pass
        # Fallback
        if quality >= 0.80 and abs(improvement) < 0.05:
            return 0.10   # stop — already excellent
        if improvement > 0.05:
            return 0.85   # continue — improving
        if improvement < -0.05:
            return 0.05   # stop — worsening
        return 0.50       # marginal

    def run_improvement_loop(
        self,
        evaluator,
        source: str,
        improve_fn,
        max_iterations: int = 5,
    ) -> Dict:
        """
        Run a full fuzzy-guided improvement loop.

        Args:
            evaluator:      FuzzyCodeEvaluator instance
            source:         initial Python source code
            improve_fn:     callable(source) → improved_source (e.g. LLM call)
            max_iterations: hard cap on iterations

        Returns:
            dict with final code, quality history, and stop reason
        """
        history = []
        current_source = source
        prev_quality = evaluator.evaluate(source).get("quality", 0.5)
        stop_reason = "max_iterations"

        for i in range(max_iterations):
            improved = improve_fn(current_source)
            if not improved or improved == current_source:
                stop_reason = "no_change"
                break

            metrics = evaluator.evaluate(improved)
            quality = metrics.get("quality", 0.5)
            cont, reason, score = self.should_continue(quality, prev_quality)

            history.append({
                "iteration": i + 1,
                "quality": quality,
                "improvement": quality - prev_quality,
                "decision": reason,
                "score": score,
            })

            current_source = improved
            prev_quality = quality

            if not cont:
                stop_reason = reason
                break

        final_metrics = evaluator.evaluate(current_source)
        return {
            "final_source":  current_source,
            "final_quality": final_metrics.get("quality", 0.0),
            "final_label":   final_metrics.get("label", ""),
            "iterations":    len(history),
            "history":       history,
            "stop_reason":   stop_reason,
        }

"""
fuzzy_agent.py — Fuzzy Agent Response Aggregator
=================================================
Weights multi-agent responses by their curvature-derived confidence.

Lower curvature → higher confidence → higher weight in synthesis.

Rules:
  IF agent_curvature=very_low  → weight=very_high
  IF agent_curvature=low       → weight=high
  IF agent_curvature=medium    → weight=medium
  IF agent_curvature=high      → weight=low
  IF agent_curvature=very_high → weight=very_low
"""

import numpy as np
from typing import List, Dict, Tuple

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False


class FuzzyAgentAggregator:
    """
    Fuzzy aggregator for multi-agent PRD responses.

    Instead of simple averaging, agents with lower curvature
    (= higher logical consistency) receive higher synthesis weight.
    """

    def __init__(self):
        self._skfuzzy_ready = False
        if SKFUZZY_AVAILABLE:
            self._build_system()

    def _build_system(self):
        try:
            r = np.arange(0, 1.01, 0.01)
            self.curv_ant   = ctrl.Antecedent(r, 'curvature')
            self.weight_con = ctrl.Consequent(r, 'weight')

            self.curv_ant['very_low']  = fuzz.trimf(r, [0.00, 0.00, 0.15])
            self.curv_ant['low']       = fuzz.trimf(r, [0.10, 0.25, 0.40])
            self.curv_ant['medium']    = fuzz.trimf(r, [0.30, 0.50, 0.70])
            self.curv_ant['high']      = fuzz.trimf(r, [0.60, 0.75, 0.90])
            self.curv_ant['very_high'] = fuzz.trimf(r, [0.80, 1.00, 1.00])

            self.weight_con['very_low']  = fuzz.trimf(r, [0.00, 0.00, 0.20])
            self.weight_con['low']       = fuzz.trimf(r, [0.10, 0.25, 0.40])
            self.weight_con['medium']    = fuzz.trimf(r, [0.30, 0.50, 0.70])
            self.weight_con['high']      = fuzz.trimf(r, [0.60, 0.75, 0.90])
            self.weight_con['very_high'] = fuzz.trimf(r, [0.80, 1.00, 1.00])

            rules = [
                ctrl.Rule(self.curv_ant['very_low'],  self.weight_con['very_high']),
                ctrl.Rule(self.curv_ant['low'],       self.weight_con['high']),
                ctrl.Rule(self.curv_ant['medium'],    self.weight_con['medium']),
                ctrl.Rule(self.curv_ant['high'],      self.weight_con['low']),
                ctrl.Rule(self.curv_ant['very_high'], self.weight_con['very_low']),
            ]
            system = ctrl.ControlSystem(rules)
            self._sim = ctrl.ControlSystemSimulation(system)
            self._skfuzzy_ready = True
        except Exception:
            self._skfuzzy_ready = False

    def curvature_to_weight(self, curvature: float) -> float:
        """Map a single agent curvature to a fuzzy weight [0, 1]."""
        curv_norm = float(np.clip(curvature, 0.0, 1.0))
        if self._skfuzzy_ready:
            try:
                self._sim.input['curvature'] = curv_norm
                self._sim.compute()
                return float(np.clip(self._sim.output['weight'], 0.0, 1.0))
            except Exception:
                pass
        # Fallback: inverse sigmoid
        return float(1.0 / (1.0 + np.exp(10 * (curv_norm - 0.5))))

    def aggregate(self, agent_results: List[Dict]) -> Dict:
        """
        Aggregate agent responses with fuzzy confidence weights.

        Each agent result must have: 'agent', 'response', 'curvature'
        Returns weighted synthesis info.
        """
        if not agent_results:
            return {"weights": {}, "top_agent": None, "weighted_prompt": ""}

        weights = {}
        for r in agent_results:
            w = self.curvature_to_weight(r.get('curvature', 0.5))
            weights[r['agent']] = w

        # Normalize
        total = sum(weights.values()) or 1.0
        norm_weights = {k: v / total for k, v in weights.items()}

        # Build weighted synthesis prompt
        lines = []
        for r in agent_results:
            w = norm_weights.get(r['agent'], 0)
            confidence_pct = int(w * 100)
            lines.append(
                f"[{r['agent']} | confidence={confidence_pct}%]: {r['response'][:300]}"
            )

        top = max(norm_weights, key=norm_weights.get)
        weighted_prompt = "\n\n".join(lines)

        return {
            "weights": norm_weights,
            "raw_weights": weights,
            "top_agent": top,
            "weighted_prompt": weighted_prompt,
        }

    def weight_table(self, agent_results: List[Dict]) -> List[Dict]:
        """Return a table of agents + weights for UI display."""
        result = self.aggregate(agent_results)
        rows = []
        for r in agent_results:
            agent = r['agent']
            rows.append({
                "Agent": f"{r.get('icon','🔵')} {agent}",
                "Role": r.get('role', ''),
                "κ": f"{r.get('curvature', 0):.4f}",
                "Fuzzy Weight": f"{result['weights'].get(agent, 0):.3f}",
                "Confidence %": f"{int(result['weights'].get(agent, 0) * 100)}%",
            })
        rows.sort(key=lambda x: x["Fuzzy Weight"], reverse=True)
        return rows

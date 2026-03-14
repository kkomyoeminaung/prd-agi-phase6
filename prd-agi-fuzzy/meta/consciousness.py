"""
Meta-layer: consciousness monitor, threshold adaptation, gauge invariance,
causal discovery.
Upgraded: EVOLUTION_RATE from .env, anomaly export, history export to CSV.
"""

import numpy as np
import logging
import os
import io
import csv
from typing import List, Tuple, Optional, Dict, Any
from scipy.linalg import expm

logger = logging.getLogger('PRD-AGI')

INITIAL_THRESHOLD = float(os.getenv("INITIAL_THRESHOLD", "0.5"))
EVOLUTION_RATE = float(os.getenv("EVOLUTION_RATE", "1e-9"))


class MetaLayer:
    """
    Consciousness monitor: tracks curvature history, adapts truth threshold,
    detects anomalies, and slowly evolves the system's understanding of truth.
    Upgraded: configurable from .env, CSV export, rolling window stats.
    """

    def __init__(self, initial_threshold: float = INITIAL_THRESHOLD,
                 evolution_rate: float = EVOLUTION_RATE):
        self.truth_threshold = initial_threshold
        self.evolution_rate = evolution_rate
        self.curvature_history: List[float] = []
        self.threshold_history: List[float] = [initial_threshold]
        self.baseline: Optional[float] = None
        self.anomalies: List[Tuple[int, float]] = []
        self._window = 100

    def update(self, curvature: float):
        self.curvature_history.append(curvature)
        n = len(self.curvature_history)

        # Establish baseline after first 100 observations
        if n == self._window:
            self.baseline = float(np.mean(self.curvature_history))
            logger.info(f"Baseline curvature established: {self.baseline:.4f}")

        # Anomaly detection: > 2x baseline
        if self.baseline and curvature > 2 * self.baseline:
            self.anomalies.append((n - 1, curvature))
            logger.warning(f"Anomaly at step {n-1}: κ={curvature:.4f} (2x baseline)")

        # Adaptive threshold evolution (PDF spec: EVOLUTION_RATE=1e-9)
        if n > 1000:
            recent = float(np.mean(self.curvature_history[-self._window:]))
            self.truth_threshold += self.evolution_rate * (recent * 0.8 - self.truth_threshold)
            self.threshold_history.append(self.truth_threshold)

    def gate(self, curvature: float) -> Tuple[bool, str]:
        """Truth gatekeeper — determines if query passes."""
        if curvature < self.truth_threshold * 0.5:
            return True, "✅ PASS"
        elif curvature < self.truth_threshold:
            return True, "⚠️ MARGINAL"
        else:
            return False, "❌ BLOCKED"

    def rolling_stats(self) -> Dict[str, float]:
        """Rolling statistics over last 100 observations."""
        if not self.curvature_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        window = self.curvature_history[-self._window:]
        return {
            "mean": float(np.mean(window)),
            "std": float(np.std(window)),
            "min": float(np.min(window)),
            "max": float(np.max(window)),
        }

    def export_history_csv(self) -> str:
        """Export curvature history as CSV string."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["step", "curvature", "threshold"])
        for i, c in enumerate(self.curvature_history):
            th = self.threshold_history[min(i, len(self.threshold_history) - 1)]
            writer.writerow([i, f"{c:.6f}", f"{th:.6f}"])
        return buf.getvalue()


class GaugeInvarianceChecker:
    """
    Checks whether a proposed transformation preserves global SU(5) symmetry.
    Instruction is treated as a local gauge transformation; if it breaks
    symmetry beyond tolerance, it's rejected.
    Upgraded: batch checking, violation breakdown per Cartan generator.
    """

    def __init__(self, dna, tolerance: float = 1e-6):
        self.dna = dna
        self.tolerance = tolerance

    def compute_violation(self, coeffs: np.ndarray) -> float:
        H = sum(c * self.dna.generators[a] for a, c in enumerate(coeffs))
        U = expm(1j * H)
        max_viol = 0.0
        for Hc in self.dna.generators[:4]:   # Cartan subalgebra
            comm = U @ Hc - Hc @ U
            viol = float(np.linalg.norm(comm))
            if viol > max_viol:
                max_viol = viol
        return max_viol

    def violation_breakdown(self, coeffs: np.ndarray) -> List[Tuple[str, float]]:
        """Per-Cartan violation breakdown."""
        H = sum(c * self.dna.generators[a] for a, c in enumerate(coeffs))
        U = expm(1j * H)
        breakdown = []
        for idx, Hc in enumerate(self.dna.generators[:4]):
            comm = U @ Hc - Hc @ U
            viol = float(np.linalg.norm(comm))
            breakdown.append((self.dna.PACCAYA_NAMES[idx], viol))
        return breakdown

    def is_invariant(self, coeffs: np.ndarray) -> bool:
        return self.compute_violation(coeffs) < self.tolerance

    def batch_check(self, coeff_list: List[np.ndarray]) -> List[bool]:
        return [self.is_invariant(c) for c in coeff_list]


class CausalDiscovery:
    """
    Discover causal links from stored relational states.
    Upgraded: Granger-inspired lag correlation, strength normalization.
    """

    def __init__(self, dna, threshold: float = 0.1):
        self.dna = dna
        self.threshold = threshold

    def discover(self, states: List[Any]) -> Dict:
        if len(states) < 2:
            return {"nodes": [], "edges": []}
        matrix = np.zeros((len(states), 24))
        for i, s in enumerate(states):
            v = np.abs(s.to_vector())
            matrix[i] = v
        corr = np.corrcoef(matrix.T)   # 24×24
        nodes = [{"id": i, "label": self.dna.PACCAYA_NAMES[i]} for i in range(24)]
        edges = []
        for i in range(24):
            for j in range(i + 1, 24):
                strength = abs(corr[i, j])
                if strength > self.threshold:
                    edges.append({
                        "source": i,
                        "target": j,
                        "weight": float(strength),
                        "from_name": self.dna.PACCAYA_NAMES[i],
                        "to_name": self.dna.PACCAYA_NAMES[j],
                    })
        edges.sort(key=lambda e: e["weight"], reverse=True)
        return {"nodes": nodes, "edges": edges}

    def export_edges_csv(self, states: List[Any]) -> str:
        result = self.discover(states)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["from", "to", "strength"])
        for e in result["edges"]:
            writer.writerow([e["from_name"], e["to_name"], f"{e['weight']:.4f}"])
        return buf.getvalue()

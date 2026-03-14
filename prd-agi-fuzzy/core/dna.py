"""
SU(5) Lie algebra generators and structure constants.
Core DNA of PRD-AGI — the 24 Paccaya causal conditions.
"""

import numpy as np
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger('PRD-AGI')


class SU5DNA:
    """
    SU(5) Lie algebra – 24 generators (the 24 Paccaya causal conditions).
    Upgraded: added generator validation and named-access helpers.
    """

    PACCAYA_NAMES = [
        "Hetu (Root Cause)", "Nissaya (Support)", "Indriya (Governing)", "Avigata (Stability)",
        "E12 (Step↑)", "E12† (Step↓)", "E13", "E13†", "E14", "E14†",
        "E15", "E15†", "E23", "E23†", "E24", "E24†",
        "E25", "E25†", "E34", "E34†",
        "S12 (Sahajata)", "R12 (Annamanna)", "S13", "R13"
    ]

    def __init__(self):
        self.dim = 5
        self.num_generators = 24
        self.generators = self._build_generators()
        self.f_ijk = self._compute_structure_constants()
        self._algebra_error = self._verify_algebra()
        logger.info(f"SU(5) DNA initialized | algebra_error={self._algebra_error:.2e}")

    def _build_generators(self) -> List[np.ndarray]:
        gens = []
        # Cartan generators (4 diagonal)
        gens.append(np.diag([1, -1, 0, 0, 0]).astype(np.complex128) / np.sqrt(2))
        gens.append(np.diag([1, 1, -2, 0, 0]).astype(np.complex128) / np.sqrt(6))
        gens.append(np.diag([1, 1, 1, -3, 0]).astype(np.complex128) / np.sqrt(12))
        gens.append(np.diag([1, 1, 1, 1, -4]).astype(np.complex128) / np.sqrt(20))
        # Step operators (16)
        steps = []
        for i in range(5):
            for j in range(5):
                if i != j:
                    E = np.zeros((5, 5), dtype=np.complex128)
                    E[i, j] = 1.0
                    steps.append(E)
                    steps.append(E.conj().T)
        gens.extend(steps[:16])
        # Interaction operators (4)
        H1, H2, H3 = gens[0], gens[1], gens[2]
        gens.extend([H1 + H2, 1j * (H1 - H2), H1 + H3, 1j * (H1 - H3)])
        return gens[:24]

    def _compute_structure_constants(self) -> np.ndarray:
        n = self.num_generators
        f = np.zeros((n, n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                comm = (self.generators[i] @ self.generators[j]
                        - self.generators[j] @ self.generators[i])
                for k in range(n):
                    val = -2j * np.trace(comm @ self.generators[k])
                    if abs(val) > 1e-12:
                        f[i, j, k] = val.real
        return f

    def _verify_algebra(self, tol: float = 1e-8) -> float:
        n = self.num_generators
        max_err = 0.0
        for i in range(n):
            for j in range(n):
                comm = (self.generators[i] @ self.generators[j]
                        - self.generators[j] @ self.generators[i])
                rhs = 1j * sum(self.f_ijk[i, j, k] * self.generators[k] for k in range(n))
                err = float(np.linalg.norm(comm - rhs))
                if err > max_err:
                    max_err = err
        return max_err

    def get_generator(self, name: str) -> Optional[np.ndarray]:
        """Get generator by Paccaya name."""
        try:
            idx = next(i for i, n in enumerate(self.PACCAYA_NAMES) if name in n)
            return self.generators[idx]
        except StopIteration:
            return None

    def commutator(self, i: int, j: int) -> np.ndarray:
        """Compute [G_i, G_j]."""
        return self.generators[i] @ self.generators[j] - self.generators[j] @ self.generators[i]

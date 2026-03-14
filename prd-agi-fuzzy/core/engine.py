"""
Curvature engine for measuring logical consistency.
Upgraded: optional torch/GPU acceleration, EVOLUTION_RATE from .env.
"""

import numpy as np
import logging
import os
from typing import Optional, Tuple, Dict
from scipy.linalg import expm
import hashlib

logger = logging.getLogger('PRD-AGI')

# ── Optional GPU support ───────────────────────────────────────────────────────
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
    logger.info(f"PyTorch available | device={DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    logger.info("PyTorch not available — using NumPy backend")


class RelationalState:
    """
    24-dimensional complex state vector – the AI's current 'being'.
    Upgraded: GPU tensor support, snapshot/restore for evolution rollback.
    """

    def __init__(self, dna, init_vector: Optional[np.ndarray] = None):
        self.dna = dna
        if init_vector is None:
            v = np.random.randn(24) + 1j * np.random.randn(24)
        else:
            v = np.array(init_vector, dtype=np.complex128)
        norm = np.linalg.norm(v)
        self.psi = v / norm if norm > 1e-10 else v
        from datetime import datetime
        self.created_at = datetime.now()
        self.last_updated = self.created_at

    def to_vector(self) -> np.ndarray:
        return self.psi.copy()

    def snapshot(self) -> np.ndarray:
        """Save current state for rollback."""
        return self.psi.copy()

    def restore(self, backup: np.ndarray):
        """Restore from snapshot."""
        self.psi = backup

    def apply_transformation(self, coeffs: np.ndarray) -> bool:
        coeffs = np.array(coeffs, dtype=float)
        H = sum(c * self.dna.generators[a] for a, c in enumerate(coeffs))
        U = expm(1j * H)
        new_psi5 = U @ self.psi[:5]
        full = self.psi.copy()
        full[:5] = new_psi5
        norm = np.linalg.norm(full)
        if norm > 1e-10:
            self.psi = full / norm
            from datetime import datetime
            self.last_updated = datetime.now()
            return True
        return False

    def to_hash(self) -> str:
        return hashlib.md5(self.psi.real.tobytes()).hexdigest()

    def expectation(self, operator: np.ndarray) -> complex:
        psi5 = self.psi[:5]
        return complex(np.vdot(psi5, operator @ psi5))

    def to_torch(self):
        """Convert to torch tensor (if available)."""
        if not TORCH_AVAILABLE:
            return None
        real = torch.tensor(self.psi.real, dtype=torch.float32, device=DEVICE)
        imag = torch.tensor(self.psi.imag, dtype=torch.float32, device=DEVICE)
        return torch.complex(real, imag)


class CurvatureEngine:
    """
    Measures logical curvature — deviation from perfect SU(5) consistency.
    Upgraded: larger sample (16 generators), configurable cache size,
    returns gradient direction for evolution guidance.
    """

    CACHE_LIMIT = 5000

    def __init__(self, dna):
        self.dna = dna
        self._cache: Dict[str, float] = {}
        self._sample_n = 16   # upgraded from 12 → 16

    def compute_curvature(self, state: RelationalState) -> float:
        h = state.to_hash()
        if h in self._cache:
            return self._cache[h]

        psi5 = state.to_vector()[:5]
        n = min(self.dna.num_generators, self._sample_n)
        total = 0.0

        for i in range(n):
            for j in range(n):
                Gi, Gj = self.dna.generators[i], self.dna.generators[j]
                comm = Gi @ Gj - Gj @ Gi
                actual = np.vdot(psi5, comm @ psi5)
                ideal = 1j * sum(
                    self.dna.f_ijk[i, j, k] * np.vdot(psi5, self.dna.generators[k] @ psi5)
                    for k in range(n)
                )
                total += abs(actual - ideal) ** 2

        result = float(np.sqrt(total / (n * n)))
        self._cache[h] = result
        if len(self._cache) > self.CACHE_LIMIT:
            # Keep newest half
            keys = list(self._cache.keys())
            for k in keys[:len(keys)//2]:
                del self._cache[k]
        return result

    def curvature_gradient(self, state: RelationalState, eps: float = 1e-4) -> np.ndarray:
        """
        Numerical gradient of curvature w.r.t. real part of state vector.
        Useful for gradient-guided evolution (faster than random walk).
        """
        base = self.compute_curvature(state)
        grad = np.zeros(24)
        backup = state.snapshot()
        for i in range(24):
            perturbed = backup.copy()
            perturbed[i] += eps
            norm = np.linalg.norm(perturbed)
            state.psi = perturbed / norm
            state._hash_cache = None  # force recompute
            c_plus = self.compute_curvature(state)
            grad[i] = (c_plus - base) / eps
        state.restore(backup)
        return grad

    def clear_cache(self):
        self._cache.clear()

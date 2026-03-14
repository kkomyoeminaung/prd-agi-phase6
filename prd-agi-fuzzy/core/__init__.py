"""Core module for PRD-AGI: SU(5) algebra, curvature engine, awareness."""

from .dna import SU5DNA
from .engine import CurvatureEngine, RelationalState
from .awareness import AwarenessDensity, CausalStrengthMonitor, ContradictionDetector

__all__ = ['SU5DNA', 'CurvatureEngine', 'RelationalState',
           'AwarenessDensity', 'CausalStrengthMonitor', 'ContradictionDetector']

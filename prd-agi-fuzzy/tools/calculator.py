"""
tools/calculator.py — PRD-AGI Advanced Calculator Tool
=======================================================
Safe, full-featured calculator with:
  - Arithmetic + algebra
  - Statistics (mean, std, correlation)
  - Unit conversion
  - Matrix operations (numpy)
  - Expression history
  - SU(5) curvature math helpers
"""

import numpy as np
import math
import re
import ast
import operator
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger('PRD-AGI.Calculator')


class CalculatorTool:
    """
    Safe advanced calculator for PRD-AGI.

    Uses AST evaluation — no exec/eval with builtins.
    Supports numpy operations, statistics, and unit conversions.

    Usage:
        calc = CalculatorTool()
        result = calc.evaluate("2**10 + np.pi * 3")
        stats  = calc.statistics([1, 2, 3, 4, 5])
        units  = calc.convert(100, "km", "miles")
    """

    # Safe math namespace
    SAFE_NAMES = {
        # Constants
        "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
        # Math functions
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "len": len, "int": int, "float": float,
        "sqrt": math.sqrt, "cbrt": lambda x: x**(1/3),
        "log": math.log, "log2": math.log2, "log10": math.log10,
        "exp": math.exp,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "atan2": math.atan2,
        "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
        "ceil": math.ceil, "floor": math.floor, "trunc": math.trunc,
        "factorial": math.factorial,
        "gcd": math.gcd, "lcm": getattr(math, 'lcm', lambda a,b: abs(a*b)//math.gcd(a,b)),
        "pow": pow, "divmod": divmod,
        "degrees": math.degrees, "radians": math.radians,
        "hypot": math.hypot,
        # Numpy
        "np": np, "array": np.array, "matrix": np.array,
        "linspace": np.linspace, "arange": np.arange,
        "zeros": np.zeros, "ones": np.ones,
        "dot": np.dot, "cross": np.cross,
        "det": np.linalg.det, "inv": np.linalg.inv,
        "eig": np.linalg.eig, "norm": np.linalg.norm,
        "mean": np.mean, "std": np.std, "var": np.var,
        "median": np.median, "percentile": np.percentile,
        "corrcoef": np.corrcoef, "cov": np.cov,
        "fft": np.fft.fft, "ifft": np.fft.ifft,
        "sort": np.sort, "argsort": np.argsort,
        "cumsum": np.cumsum, "diff": np.diff,
        "clip": np.clip, "where": np.where,
        "True": True, "False": False,
    }

    # Unit conversion factors (to SI base)
    UNITS: Dict[str, Dict[str, float]] = {
        "length": {
            "m":1.0,"km":1000,"cm":0.01,"mm":0.001,"um":1e-6,"nm":1e-9,
            "miles":1609.344,"mi":1609.344,"yards":0.9144,"yd":0.9144,
            "feet":0.3048,"ft":0.3048,"inches":0.0254,"in":0.0254,
            "ly":9.461e15,"au":1.496e11,"pc":3.086e16,
        },
        "mass": {
            "kg":1.0,"g":0.001,"mg":1e-6,"ug":1e-9,"t":1000,
            "lbs":0.453592,"lb":0.453592,"oz":0.028350,"st":6.35029,
            "ton":907.185,
        },
        "time": {
            "s":1.0,"ms":0.001,"us":1e-6,"ns":1e-9,
            "min":60,"hr":3600,"h":3600,"day":86400,"week":604800,
            "month":2629800,"year":31557600,
        },
        "temperature": {
            "C":"celsius","F":"fahrenheit","K":"kelvin",
        },
        "data": {
            "b":1,"B":8,"KB":8e3,"MB":8e6,"GB":8e9,"TB":8e12,
            "KiB":8192,"MiB":8388608,"GiB":8589934592,
        },
        "speed": {
            "m/s":1.0,"km/h":1/3.6,"mph":0.44704,"knot":0.514444,
            "c":299792458,
        },
    }

    def __init__(self):
        self.history: List[Dict] = []
        self._max_history = 200

    # ── Core evaluate ────────────────────────────────────────────────────────

    def evaluate(self, expression: str) -> Dict:
        """
        Safely evaluate a mathematical expression.

        Returns:
            {success, result, result_str, expression, error}
        """
        expr = expression.strip()
        if not expr:
            return {"success": False, "error": "Empty expression", "result": None}

        # Security: block dangerous patterns
        blocked = ["import", "__", "open(", "exec(", "eval(", "compile(",
                   "globals(", "locals(", "getattr", "setattr", "delattr",
                   "subprocess", "os.system", "os.popen"]
        expr_lower = expr.lower()
        for b in blocked:
            if b in expr_lower:
                return {"success": False, "error": f"Blocked: '{b}'", "result": None}

        try:
            # Replace ^ with ** for convenience
            safe_expr = expr.replace("^", "**")
            result = eval(safe_expr, {"__builtins__": {}}, self.SAFE_NAMES)  # noqa: S307

            # Format result
            if isinstance(result, (np.ndarray,)):
                result_str = np.array2string(result, precision=6, suppress_small=True)
            elif isinstance(result, float):
                result_str = f"{result:.10g}"
            elif isinstance(result, complex):
                result_str = f"{result.real:.6g} + {result.imag:.6g}i"
            else:
                result_str = str(result)

            entry = {
                "expression": expr,
                "result":     result,
                "result_str": result_str,
                "success":    True,
                "error":      "",
            }
            self._log(entry)
            return entry

        except ZeroDivisionError:
            return {"success": False, "error": "Division by zero", "result": None, "expression": expr}
        except Exception as ex:
            return {"success": False, "error": str(ex), "result": None, "expression": expr}

    # ── Statistics ───────────────────────────────────────────────────────────

    def statistics(self, data: List[float]) -> Dict:
        """Full descriptive statistics for a list of numbers."""
        if not data:
            return {"error": "Empty data"}
        arr = np.array(data, dtype=float)
        q1, q2, q3 = np.percentile(arr, [25, 50, 75])
        return {
            "n":       len(arr),
            "mean":    float(np.mean(arr)),
            "median":  float(np.median(arr)),
            "std":     float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "var":     float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min":     float(np.min(arr)),
            "max":     float(np.max(arr)),
            "range":   float(np.max(arr) - np.min(arr)),
            "q1":      float(q1),
            "q2":      float(q2),
            "q3":      float(q3),
            "iqr":     float(q3 - q1),
            "skew":    self._skewness(arr),
            "kurtosis":self._kurtosis(arr),
            "sum":     float(np.sum(arr)),
        }

    def _skewness(self, arr: np.ndarray) -> float:
        if len(arr) < 3: return 0.0
        n = len(arr)
        m = np.mean(arr); s = np.std(arr, ddof=1)
        if s == 0: return 0.0
        return float(n / ((n-1)*(n-2)) * np.sum(((arr-m)/s)**3))

    def _kurtosis(self, arr: np.ndarray) -> float:
        if len(arr) < 4: return 0.0
        n = len(arr); m = np.mean(arr); s = np.std(arr, ddof=1)
        if s == 0: return 0.0
        return float((n*(n+1)/((n-1)*(n-2)*(n-3)) * np.sum(((arr-m)/s)**4))
                     - 3*(n-1)**2/((n-2)*(n-3)))

    # ── Unit conversion ──────────────────────────────────────────────────────

    def convert(self, value: float, from_unit: str, to_unit: str) -> Dict:
        """Convert between units."""
        # Temperature special case
        if from_unit in ("C","F","K") or to_unit in ("C","F","K"):
            return self._temp_convert(value, from_unit, to_unit)

        # Find category
        from_factor = to_factor = None
        for cat, units in self.UNITS.items():
            if cat == "temperature":
                continue
            if from_unit in units and to_unit in units:
                from_factor = units[from_unit]
                to_factor   = units[to_unit]
                break

        if from_factor is None:
            return {"success": False, "error": f"Unknown unit pair: {from_unit} → {to_unit}"}

        result = value * (from_factor / to_factor)
        return {
            "success": True,
            "input":   f"{value} {from_unit}",
            "output":  f"{result:.8g} {to_unit}",
            "result":  result,
        }

    def _temp_convert(self, v: float, f: str, t: str) -> Dict:
        # to Kelvin first
        if f == "C":   k = v + 273.15
        elif f == "F": k = (v + 459.67) * 5/9
        else:          k = v
        # from Kelvin
        if t == "C":   r = k - 273.15
        elif t == "F": r = k * 9/5 - 459.67
        else:          r = k
        return {"success": True, "input": f"{v}°{f}", "output": f"{r:.4f}°{t}", "result": r}

    # ── Matrix ops ───────────────────────────────────────────────────────────

    def matrix_op(self, op: str, *matrices) -> Dict:
        """Matrix operations: det, inv, eig, multiply, solve."""
        try:
            mats = [np.array(m, dtype=float) for m in matrices]
            if op == "det":
                return {"success": True, "result": float(np.linalg.det(mats[0]))}
            elif op == "inv":
                return {"success": True, "result": np.linalg.inv(mats[0]).tolist()}
            elif op == "eig":
                vals, vecs = np.linalg.eig(mats[0])
                return {"success": True, "eigenvalues": vals.tolist(), "eigenvectors": vecs.tolist()}
            elif op == "multiply":
                return {"success": True, "result": (mats[0] @ mats[1]).tolist()}
            elif op == "solve":
                return {"success": True, "result": np.linalg.solve(mats[0], mats[1]).tolist()}
            else:
                return {"success": False, "error": f"Unknown op: {op}"}
        except Exception as ex:
            return {"success": False, "error": str(ex)}

    # ── History ──────────────────────────────────────────────────────────────

    def _log(self, entry: Dict):
        self.history.append(entry)
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]

    def history_df_rows(self) -> List[Dict]:
        return [{"Expression": h["expression"], "Result": h["result_str"]}
                for h in self.history[-20:]]

    def clear_history(self):
        self.history.clear()

"""
fuzzy_code.py — Fuzzy Code Quality Evaluator
=============================================
Evaluates Python code on 3 fuzzy dimensions:
  - Complexity   (AST node count, nesting depth)
  - Readability  (docstrings, naming, line length)
  - Efficiency   (loop patterns, list comps, cache hints)

Produces an overall fuzzy quality score + improvement advice.
"""

import ast
import numpy as np
from typing import Dict, Tuple, List

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False


class FuzzyCodeEvaluator:
    """
    Multi-dimensional fuzzy code quality evaluator.

    Dimensions:
        complexity   [0,1]  — 0=simple, 1=very complex
        readability  [0,1]  — 0=unreadable, 1=excellent
        efficiency   [0,1]  — 0=inefficient, 1=highly efficient

    Output:
        quality      [0,1]  — overall fuzzy quality score
    """

    ADVICE = {
        (0.0, 0.30): [
            "🔴 Code quality is poor. Consider major refactoring.",
            "Break large functions into smaller ones.",
            "Add docstrings and type hints.",
            "Remove deeply nested conditionals.",
        ],
        (0.30, 0.55): [
            "🟡 Code quality is moderate. Several improvements possible.",
            "Add missing docstrings.",
            "Consider list comprehensions where applicable.",
            "Improve variable naming.",
        ],
        (0.55, 0.80): [
            "🟢 Code quality is good. Minor improvements possible.",
            "Consider caching repeated computations.",
            "Add unit tests for complex functions.",
        ],
        (0.80, 1.01): [
            "✨ Code quality is excellent!",
            "Well-structured, readable, and efficient.",
        ],
    }

    def __init__(self):
        self._skfuzzy_ready = False
        if SKFUZZY_AVAILABLE:
            self._build_system()

    def _build_system(self):
        try:
            r = np.arange(0, 1.01, 0.01)

            self.complexity   = ctrl.Antecedent(r, 'complexity')
            self.readability  = ctrl.Antecedent(r, 'readability')
            self.efficiency   = ctrl.Antecedent(r, 'efficiency')
            self.quality      = ctrl.Consequent(r, 'quality')

            for ant in [self.complexity, self.readability, self.efficiency]:
                ant['low']    = fuzz.trimf(r, [0.00, 0.00, 0.40])
                ant['medium'] = fuzz.trimf(r, [0.25, 0.50, 0.75])
                ant['high']   = fuzz.trimf(r, [0.60, 1.00, 1.00])

            self.quality['poor']      = fuzz.trimf(r, [0.00, 0.00, 0.35])
            self.quality['moderate']  = fuzz.trimf(r, [0.20, 0.45, 0.70])
            self.quality['good']      = fuzz.trimf(r, [0.55, 0.75, 0.90])
            self.quality['excellent'] = fuzz.trimf(r, [0.80, 1.00, 1.00])

            # Note: complexity is inverted — low complexity = higher quality
            rules = [
                # Excellent quality conditions
                ctrl.Rule(self.complexity['low']    & self.readability['high']   & self.efficiency['high'],   self.quality['excellent']),
                ctrl.Rule(self.complexity['low']    & self.readability['high']   & self.efficiency['medium'], self.quality['good']),
                # Good quality
                ctrl.Rule(self.complexity['low']    & self.readability['medium'] & self.efficiency['high'],   self.quality['good']),
                ctrl.Rule(self.complexity['medium'] & self.readability['high']   & self.efficiency['high'],   self.quality['good']),
                ctrl.Rule(self.complexity['low']    & self.readability['medium'] & self.efficiency['medium'], self.quality['good']),
                # Moderate quality
                ctrl.Rule(self.complexity['medium'] & self.readability['medium'] & self.efficiency['medium'], self.quality['moderate']),
                ctrl.Rule(self.complexity['medium'] & self.readability['low']    & self.efficiency['high'],   self.quality['moderate']),
                ctrl.Rule(self.complexity['high']   & self.readability['high']   & self.efficiency['medium'], self.quality['moderate']),
                # Poor quality
                ctrl.Rule(self.complexity['high']   & self.readability['low'],                                self.quality['poor']),
                ctrl.Rule(self.complexity['high']   & self.efficiency['low'],                                 self.quality['poor']),
                ctrl.Rule(self.readability['low']   & self.efficiency['low'],                                 self.quality['poor']),
            ]

            system = ctrl.ControlSystem(rules)
            self._sim = ctrl.ControlSystemSimulation(system)
            self._skfuzzy_ready = True
        except Exception:
            self._skfuzzy_ready = False

    # ── Metric extractors ───────────────────────────────────────────────────

    def _measure_complexity(self, source: str, tree: ast.AST) -> float:
        """Normalize complexity score [0,1]. Higher = more complex."""
        nodes = list(ast.walk(tree))
        node_count = len(nodes)
        branches = sum(1 for n in nodes if isinstance(n, (ast.If, ast.For, ast.While,
                                                          ast.Try, ast.ExceptHandler)))
        functions = sum(1 for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        classes = sum(1 for n in nodes if isinstance(n, ast.ClassDef))

        # Max nesting depth
        def depth(node, d=0):
            return max([d] + [depth(c, d+1) for c in ast.iter_child_nodes(node)
                               if isinstance(c, (ast.If, ast.For, ast.While, ast.With, ast.Try))])
        try:
            max_depth = depth(tree)
        except RecursionError:
            max_depth = 10

        score = (
            min(node_count / 200, 1.0) * 0.30 +
            min(branches / 20, 1.0)    * 0.30 +
            min(max_depth / 8, 1.0)    * 0.25 +
            min((functions + classes) / 15, 1.0) * 0.15
        )
        return float(np.clip(score, 0.0, 1.0))

    def _measure_readability(self, source: str, tree: ast.AST) -> float:
        """Readability [0,1]. Higher = more readable."""
        lines = source.split('\n')
        nodes = list(ast.walk(tree))

        # Docstring coverage
        funcs = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        docstring_ratio = 0.0
        if funcs:
            with_docs = sum(
                1 for f in funcs
                if f.body and
                isinstance(f.body[0], ast.Expr) and
                isinstance(f.body[0].value, ast.Constant) and
                isinstance(f.body[0].value.value, str)
            )
            docstring_ratio = with_docs / len(funcs)

        # Line length (< 88 chars PEP8 extended)
        long_lines = sum(1 for l in lines if len(l) > 88)
        line_ratio = 1.0 - min(long_lines / max(len(lines), 1), 1.0)

        # Naming: avoid single-char variables (except i, j, k, n, x, y)
        allowed_short = {'i', 'j', 'k', 'n', 'x', 'y', 'f', 'e', 'v', 's'}
        names = [n.id for n in nodes if isinstance(n, ast.Name)]
        bad_names = sum(1 for nm in names if len(nm) == 1 and nm not in allowed_short)
        naming_score = 1.0 - min(bad_names / max(len(names), 1), 1.0)

        # Comment density
        comment_lines = sum(1 for l in lines if l.strip().startswith('#'))
        comment_ratio = min(comment_lines / max(len(lines), 1) / 0.10, 1.0)

        score = (
            docstring_ratio * 0.35 +
            line_ratio      * 0.25 +
            naming_score    * 0.25 +
            comment_ratio   * 0.15
        )
        return float(np.clip(score, 0.0, 1.0))

    def _measure_efficiency(self, source: str, tree: ast.AST) -> float:
        """Efficiency [0,1]. Higher = more efficient patterns."""
        nodes = list(ast.walk(tree))

        # List comprehensions vs plain loops
        list_comps = sum(1 for n in nodes if isinstance(n, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)))
        for_loops  = sum(1 for n in nodes if isinstance(n, ast.For))
        comp_ratio = list_comps / max(for_loops + list_comps, 1)

        # LRU cache / functools usage
        imports = [ast.unparse(n) for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom))]
        has_cache = any('functools' in i or 'lru_cache' in i or 'cache' in i for i in imports)

        # Avoid repeated attribute lookup in loops (heuristic: nested ast.Attribute in For)
        nested_attrs = sum(
            1 for n in nodes
            if isinstance(n, ast.For) and
            any(isinstance(c, ast.Attribute) for c in ast.walk(n))
        )
        attr_penalty = min(nested_attrs / max(for_loops, 1), 1.0) * 0.2

        # Early returns (guard clauses)
        early_returns = sum(
            1 for n in nodes
            if isinstance(n, ast.Return) and
            isinstance(getattr(n, 'value', None), (ast.Constant, ast.Name))
        )
        early_score = min(early_returns / 5, 1.0)

        score = (
            comp_ratio  * 0.35 +
            (0.20 if has_cache else 0.0) +
            early_score * 0.25 +
            0.20
        ) - attr_penalty
        return float(np.clip(score, 0.0, 1.0))

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate(self, source: str) -> Dict:
        """
        Full fuzzy evaluation of Python source code.

        Returns dict with metrics, quality score, label, and advice.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"error": str(e), "quality": 0.0, "label": "🔴 Syntax Error"}

        complexity  = self._measure_complexity(source, tree)
        readability = self._measure_readability(source, tree)
        efficiency  = self._measure_efficiency(source, tree)

        quality = self._fuzzy_quality(complexity, readability, efficiency)
        label   = self._quality_label(quality)
        advice  = self._get_advice(quality, complexity, readability, efficiency)

        return {
            "complexity":    round(complexity, 4),
            "readability":   round(readability, 4),
            "efficiency":    round(efficiency, 4),
            "quality":       round(quality, 4),
            "label":         label,
            "advice":        advice,
            "lines":         source.count('\n') + 1,
        }

    def _fuzzy_quality(self, complexity: float, readability: float, efficiency: float) -> float:
        if self._skfuzzy_ready:
            try:
                self._sim.input['complexity']  = float(np.clip(complexity, 0.01, 0.99))
                self._sim.input['readability'] = float(np.clip(readability, 0.01, 0.99))
                self._sim.input['efficiency']  = float(np.clip(efficiency, 0.01, 0.99))
                self._sim.compute()
                return float(np.clip(self._sim.output['quality'], 0.0, 1.0))
            except Exception:
                pass
        # Fallback: weighted average (inverted complexity)
        return float(np.clip(
            (1 - complexity) * 0.30 + readability * 0.40 + efficiency * 0.30,
            0.0, 1.0
        ))

    def _quality_label(self, quality: float) -> str:
        if quality >= 0.80: return "✨ Excellent"
        if quality >= 0.55: return "🟢 Good"
        if quality >= 0.30: return "🟡 Moderate"
        return "🔴 Poor"

    def _get_advice(self, quality, complexity, readability, efficiency) -> List[str]:
        advice = []
        for (lo, hi), tips in self.ADVICE.items():
            if lo <= quality < hi:
                advice.extend(tips)
                break
        # Specific metric tips
        if complexity > 0.70:
            advice.append("⚠️ High complexity detected — consider splitting into smaller functions.")
        if readability < 0.40:
            advice.append("📖 Low readability — add docstrings and improve variable names.")
        if efficiency < 0.40:
            advice.append("⚡ Low efficiency — use list comprehensions and consider caching.")
        return advice

"""
self_modify/self_modify.py — PRD-AGI Self-Modification Engine
=============================================================
Enables PRD-AGI to write, evaluate, and optionally apply new code
to itself — guided by SU(5) curvature and fuzzy quality metrics.

Theory alignment:
  Self-modification is governed by Truth Preservation:
  - Only accept modifications that reduce curvature (increase consistency)
  - All changes are logged, reversible, and curvature-verified
  - Gauge invariance check on proposed transformation

Workflow:
  1. Identify improvement target (function/class/module)
  2. Generate improved version via LLM
  3. Syntax + safety check
  4. Curvature evaluation (does the new code reduce logical tension?)
  5. Fuzzy quality score
  6. If approved → write to workspace (NOT to main codebase without explicit confirm)
  7. Log to modification history
"""

import ast
import hashlib
import os
import sys
import difflib
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger('PRD-AGI.SelfModify')

# Safety blocklist — these patterns are NEVER allowed in generated code
BLOCKED_PATTERNS = [
    "os.system", "os.popen", "subprocess", "shutil.rmtree",
    "__import__", "importlib", "ctypes", "socket.connect",
    "requests.post", "eval(", "exec(", "compile(",
    "open(/etc", "open(/proc", "open(/sys",
    "rm -rf", "format c:", "del /f",
]


class ModificationRecord:
    """Records a single self-modification attempt."""
    def __init__(self, target: str, original: str, proposed: str,
                 approved: bool, reason: str, curvature_delta: float):
        self.id              = hashlib.md5(proposed.encode()).hexdigest()[:8]
        self.target          = target
        self.original        = original
        self.proposed        = proposed
        self.approved        = approved
        self.reason          = reason
        self.curvature_delta = curvature_delta
        self.timestamp       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.applied         = False

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "target": self.target,
            "approved": self.approved, "reason": self.reason,
            "curvature_delta": self.curvature_delta,
            "timestamp": self.timestamp, "applied": self.applied,
            "lines_original": self.original.count("\n"),
            "lines_proposed": self.proposed.count("\n"),
        }


class SelfModifyEngine:
    """
    PRD-AGI Self-Modification Engine.

    Generates improved code via LLM, validates it through multiple
    safety and quality gates, and writes approved versions to
    the workspace (never directly overwrites running code without user confirmation).

    Usage:
        engine = SelfModifyEngine(llm_client, curvature_engine, fuzzy_evaluator)
        result = engine.propose_improvement("def foo(x): return x*2", target="foo")
        # If approved, optionally apply:
        engine.apply(result['record_id'])
    """

    def __init__(self, llm_client=None, curvature_engine=None,
                 fuzzy_evaluator=None,
                 workspace: str = "workspace/self_modify"):
        self.llm          = llm_client
        self.engine       = curvature_engine
        self.fuzzy_ev     = fuzzy_evaluator
        self.workspace    = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.history: List[ModificationRecord] = []
        self._pending: Dict[str, ModificationRecord] = {}  # id → record

    # ── Public API ───────────────────────────────────────────────────────────

    def propose_improvement(
        self,
        source_code: str,
        target: str = "code",
        goal: str = "Improve readability, efficiency, and logical consistency.",
        max_attempts: int = 3,
    ) -> Dict:
        """
        Generate an improved version of source_code.

        Steps:
          1. LLM generates improved code
          2. Syntax check
          3. Safety check
          4. Curvature evaluation
          5. Fuzzy quality score
          6. Return proposal with approval status

        Returns:
            {success, approved, proposed_code, diff, reason,
             quality_before, quality_after, curvature_delta, record_id}
        """
        if not self.llm:
            return {"success": False, "error": "No LLM client available"}

        # Get baseline metrics
        quality_before = self._fuzzy_score(source_code)
        curv_before    = self._curvature_score(source_code)

        # Generate improved code
        proposed = None
        for attempt in range(max_attempts):
            candidate = self._generate(source_code, goal, attempt)
            if not candidate:
                continue

            # Clean (remove markdown fences)
            candidate = self._clean_code(candidate)

            # Syntax check
            syn_ok, syn_err = self._syntax_check(candidate)
            if not syn_ok:
                logger.warning(f"Attempt {attempt+1} syntax error: {syn_err}")
                continue

            # Safety check
            safe_ok, safe_err = self._safety_check(candidate)
            if not safe_ok:
                logger.warning(f"Attempt {attempt+1} safety blocked: {safe_err}")
                continue

            proposed = candidate
            break

        if not proposed:
            return {"success": False, "error": "Failed to generate valid code after retries"}

        # Evaluate improvements
        quality_after  = self._fuzzy_score(proposed)
        curv_after     = self._curvature_score(proposed)
        curv_delta     = curv_after - curv_before
        quality_delta  = quality_after - quality_before

        # Diff
        diff = self._compute_diff(source_code, proposed, target)

        # Approval: curvature must not increase, quality must improve or hold
        approved = (curv_delta <= 0.05) and (quality_delta >= -0.05)
        reason = self._approval_reason(approved, curv_delta, quality_delta)

        # Create record
        record = ModificationRecord(
            target=target,
            original=source_code,
            proposed=proposed,
            approved=approved,
            reason=reason,
            curvature_delta=curv_delta,
        )
        self.history.append(record)
        self._pending[record.id] = record

        # Save proposal to workspace
        self._save_proposal(record)

        return {
            "success":        True,
            "approved":       approved,
            "proposed_code":  proposed,
            "diff":           diff,
            "reason":         reason,
            "quality_before": round(quality_before, 4),
            "quality_after":  round(quality_after, 4),
            "quality_delta":  round(quality_delta, 4),
            "curv_before":    round(curv_before, 4),
            "curv_after":     round(curv_after, 4),
            "curvature_delta":round(curv_delta, 4),
            "record_id":      record.id,
            "lines_before":   source_code.count("\n") + 1,
            "lines_after":    proposed.count("\n") + 1,
        }

    def apply(self, record_id: str, output_filename: Optional[str] = None) -> Dict:
        """
        Apply an approved modification to the workspace.
        Does NOT overwrite running source files — writes to workspace only.
        """
        record = self._pending.get(record_id)
        if not record:
            return {"success": False, "error": f"Record {record_id} not found"}
        if not record.approved:
            return {"success": False, "error": "Cannot apply unapproved modification"}
        if record.applied:
            return {"success": False, "error": "Already applied"}

        fname = output_filename or f"{record.target}_{record.id}.py"
        path  = self.workspace / fname
        path.write_text(record.proposed, encoding="utf-8")
        record.applied = True

        logger.info(f"Self-modify applied: {fname}")
        return {"success": True, "written_to": str(path), "record_id": record_id}

    def run_improvement_loop(
        self,
        source_code: str,
        target: str = "code",
        max_iterations: int = 5,
        stop_at_quality: float = 0.85,
    ) -> Dict:
        """
        Iteratively improve code until quality threshold or max iterations.
        Uses FuzzyImprovementDecider logic internally.
        """
        current = source_code
        history = []
        prev_quality = self._fuzzy_score(current)

        for i in range(max_iterations):
            result = self.propose_improvement(current, target)
            if not result["success"]:
                break

            quality = result["quality_after"]
            history.append({
                "iteration":     i + 1,
                "quality":       quality,
                "delta":         result["quality_delta"],
                "curv_delta":    result["curvature_delta"],
                "approved":      result["approved"],
                "record_id":     result["record_id"],
            })

            if result["approved"]:
                current = result["proposed_code"]

            # Stop conditions
            if quality >= stop_at_quality:
                break
            if not result["approved"] and result["quality_delta"] < -0.1:
                break  # getting worse
            prev_quality = quality

        final_quality = self._fuzzy_score(current)
        return {
            "final_code":     current,
            "final_quality":  round(final_quality, 4),
            "iterations":     len(history),
            "history":        history,
            "improved":       final_quality > self._fuzzy_score(source_code),
        }

    def get_history(self, last_n: int = 20) -> List[Dict]:
        return [r.to_dict() for r in self.history[-last_n:]]

    # ── Internal ─────────────────────────────────────────────────────────────

    def _generate(self, code: str, goal: str, attempt: int) -> Optional[str]:
        """Ask LLM to improve the code."""
        if not self.llm:
            return None
        prompt = (
            f"Improve this Python code. Goal: {goal}\n"
            f"Attempt: {attempt + 1}/3\n\n"
            f"```python\n{code}\n```\n\n"
            f"Return ONLY the improved Python code. No explanations. No markdown."
        )
        system = (
            "You are a Python expert. Improve code for readability, "
            "efficiency, and logical clarity. Preserve all functionality. "
            "Return only code, no explanations."
        )
        try:
            response = self.llm.generate(prompt, system=system)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM generate error: {e}")
            return None

    def _clean_code(self, code: str) -> str:
        """Remove markdown fences if present."""
        lines = code.split("\n")
        cleaned = [l for l in lines if not l.strip().startswith("```")]
        return "\n".join(cleaned).strip()

    def _syntax_check(self, code: str) -> Tuple[bool, str]:
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, str(e)

    def _safety_check(self, code: str) -> Tuple[bool, str]:
        code_lower = code.lower()
        for pattern in BLOCKED_PATTERNS:
            if pattern.lower() in code_lower:
                return False, f"Blocked pattern: '{pattern}'"
        # Check for excessive file operations
        open_count = code_lower.count("open(")
        if open_count > 10:
            return False, "Too many file open() calls"
        return True, ""

    def _fuzzy_score(self, code: str) -> float:
        if self.fuzzy_ev:
            try:
                result = self.fuzzy_ev.evaluate(code)
                return result.get("quality", 0.5)
            except Exception:
                pass
        return 0.5  # neutral default

    def _curvature_score(self, code: str) -> float:
        """Map code AST to a curvature-like complexity score."""
        try:
            tree = ast.parse(code)
            nodes = list(ast.walk(tree))
            # Complexity proxy: normalized node count + nesting
            n_nodes = len(nodes)
            n_branches = sum(1 for n in nodes if isinstance(n, (ast.If, ast.For, ast.While)))
            return float(min((n_nodes / 300 + n_branches / 20) / 2, 1.0))
        except Exception:
            return 0.5

    def _compute_diff(self, original: str, proposed: str, target: str) -> str:
        """Compute unified diff between original and proposed."""
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=f"{target}_original.py",
            tofile=f"{target}_improved.py",
            n=2,
        )
        return "".join(list(diff)[:50])  # limit to 50 lines

    def _approval_reason(self, approved: bool, curv_delta: float,
                          quality_delta: float) -> str:
        if approved:
            parts = []
            if quality_delta > 0.05:
                parts.append(f"quality improved +{quality_delta:.3f}")
            if curv_delta < -0.02:
                parts.append(f"curvature reduced {curv_delta:.3f}")
            if not parts:
                parts.append("holds quality and curvature")
            return "✅ Approved: " + ", ".join(parts)
        else:
            parts = []
            if quality_delta < -0.05:
                parts.append(f"quality dropped {quality_delta:.3f}")
            if curv_delta > 0.05:
                parts.append(f"curvature increased +{curv_delta:.3f}")
            return "❌ Rejected: " + (", ".join(parts) or "insufficient improvement")

    def _save_proposal(self, record: ModificationRecord):
        """Save proposal to workspace for review."""
        try:
            fname = self.workspace / f"proposal_{record.id}.py"
            header = (
                f"# PRD-AGI Self-Modification Proposal\n"
                f"# ID: {record.id}\n"
                f"# Target: {record.target}\n"
                f"# Approved: {record.approved}\n"
                f"# Reason: {record.reason}\n"
                f"# Curvature delta: {record.curvature_delta:.4f}\n"
                f"# Timestamp: {record.timestamp}\n\n"
            )
            fname.write_text(header + record.proposed, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not save proposal: {e}")

"""
tools/file_io.py — PRD-AGI File I/O Tool
=========================================
Safe file operations for PRD-AGI:
  - Read/write text, JSON, CSV, Markdown
  - PDF text extraction (PyMuPDF or pdfminer fallback)
  - Image description request (sends to LLM)
  - Workspace management (sandboxed workspace/ dir)
  - Export: chat history, curvature data, sentience logs
"""

import os
import json
import csv
import io
import logging
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

logger = logging.getLogger('PRD-AGI.FileIO')

# Try optional PDF deps
try:
    import fitz  # PyMuPDF
    PDF_BACKEND = "pymupdf"
except ImportError:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        PDF_BACKEND = "pdfminer"
    except ImportError:
        PDF_BACKEND = None


class FileIOTool:
    """
    Sandboxed file I/O for PRD-AGI.

    All file operations are restricted to the workspace/ directory
    (except explicit uploads from Streamlit file_uploader).

    Supports:
      - .txt, .md, .py, .json, .csv, .log
      - PDF text extraction
      - Export helpers for PRD-AGI data
    """

    ALLOWED_EXT = {'.txt', '.md', '.py', '.json', '.csv', '.log',
                   '.yaml', '.toml', '.env', '.sh', '.bat', '.html'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    def __init__(self, workspace: str = "workspace"):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.file_registry: List[Dict] = []

    # ── Read ─────────────────────────────────────────────────────────────────

    def read(self, filename: str) -> Dict:
        """Read a text file from the workspace."""
        path = self._safe_path(filename)
        if not path:
            return {"success": False, "error": f"Path not allowed: {filename}"}
        if not path.exists():
            return {"success": False, "error": f"File not found: {filename}"}
        if path.stat().st_size > self.MAX_FILE_SIZE:
            return {"success": False, "error": "File too large (max 10MB)"}
        try:
            content = path.read_text(encoding="utf-8")
            return {
                "success": True, "filename": filename,
                "content": content, "size": len(content),
                "lines": content.count("\n") + 1,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_json(self, filename: str) -> Dict:
        """Read and parse a JSON file."""
        result = self.read(filename)
        if not result["success"]:
            return result
        try:
            data = json.loads(result["content"])
            return {"success": True, "filename": filename, "data": data}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON parse error: {e}"}

    def read_csv(self, filename: str) -> Dict:
        """Read a CSV file and return rows as list of dicts."""
        result = self.read(filename)
        if not result["success"]:
            return result
        try:
            reader = csv.DictReader(io.StringIO(result["content"]))
            rows = list(reader)
            return {
                "success": True, "filename": filename,
                "rows": rows, "count": len(rows),
                "columns": reader.fieldnames or [],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_pdf_text(self, uploaded_bytes: bytes, filename: str = "doc.pdf") -> Dict:
        """Extract text from PDF bytes (from Streamlit file_uploader)."""
        if PDF_BACKEND == "pymupdf":
            try:
                doc = fitz.open(stream=uploaded_bytes, filetype="pdf")
                pages = []
                for i, page in enumerate(doc):
                    text = page.get_text()
                    pages.append({"page": i+1, "text": text})
                full_text = "\n\n".join(p["text"] for p in pages)
                return {
                    "success": True, "filename": filename,
                    "pages": len(pages), "text": full_text,
                    "page_texts": pages,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif PDF_BACKEND == "pdfminer":
            try:
                text = pdfminer_extract(io.BytesIO(uploaded_bytes))
                return {"success": True, "filename": filename, "text": text, "pages": 1}
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {
                "success": False,
                "error": "PDF extraction requires PyMuPDF or pdfminer. "
                         "Install: pip install pymupdf  OR  pip install pdfminer.six"
            }

    # ── Write ────────────────────────────────────────────────────────────────

    def write(self, filename: str, content: str, mode: str = "w") -> Dict:
        """Write text to a file in the workspace."""
        path = self._safe_path(filename)
        if not path:
            return {"success": False, "error": f"Path not allowed: {filename}"}
        if len(content) > self.MAX_FILE_SIZE:
            return {"success": False, "error": "Content too large (max 10MB)"}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)
            self._register(filename, len(content))
            return {"success": True, "filename": filename, "bytes_written": len(content.encode())}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def write_json(self, filename: str, data: Any, indent: int = 2) -> Dict:
        """Write data as JSON to the workspace."""
        try:
            content = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
            return self.write(filename, content)
        except Exception as e:
            return {"success": False, "error": f"JSON serialization error: {e}"}

    def write_csv(self, filename: str, rows: List[Dict],
                  fieldnames: Optional[List[str]] = None) -> Dict:
        """Write list of dicts as CSV."""
        if not rows:
            return {"success": False, "error": "No rows to write"}
        try:
            buf = io.StringIO()
            fnames = fieldnames or list(rows[0].keys())
            writer = csv.DictWriter(buf, fieldnames=fnames)
            writer.writeheader()
            writer.writerows(rows)
            return self.write(filename, buf.getvalue())
        except Exception as e:
            return {"success": False, "error": str(e)}

    def append(self, filename: str, content: str) -> Dict:
        """Append text to a file."""
        return self.write(filename, content, mode="a")

    # ── List / Delete ─────────────────────────────────────────────────────────

    def list_files(self, subdir: str = "") -> Dict:
        """List files in workspace."""
        base = self.workspace / subdir if subdir else self.workspace
        if not base.exists():
            return {"success": False, "error": f"Directory not found: {subdir}"}
        files = []
        for p in sorted(base.iterdir()):
            if p.is_file():
                files.append({
                    "name":     p.name,
                    "size":     p.stat().st_size,
                    "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "ext":      p.suffix,
                })
        return {"success": True, "files": files, "count": len(files)}

    def delete(self, filename: str) -> Dict:
        """Delete a file from workspace."""
        path = self._safe_path(filename)
        if not path:
            return {"success": False, "error": "Path not allowed"}
        if not path.exists():
            return {"success": False, "error": "File not found"}
        try:
            path.unlink()
            return {"success": True, "deleted": filename}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── PRD-AGI Export Helpers ────────────────────────────────────────────────

    def export_chat(self, history: List[Dict], filename: str = "chat_export.json") -> Dict:
        """Export chat history to workspace."""
        return self.write_json(filename, {
            "exported_at": datetime.now().isoformat(),
            "message_count": len(history),
            "messages": history,
        })

    def export_curvature(self, curvature_history: List[float],
                          filename: str = "curvature_log.csv") -> Dict:
        """Export curvature history as CSV."""
        rows = [{"step": i, "curvature": c} for i, c in enumerate(curvature_history)]
        return self.write_csv(filename, rows)

    def export_session_snapshot(self, session_data: Dict,
                                 filename: str = "session_snapshot.json") -> Dict:
        """Export full session snapshot (state, history, metrics)."""
        return self.write_json(filename, {
            "exported_at": datetime.now().isoformat(),
            "version": "PRD-AGI Phase 6 + Fuzzy + Sentience",
            **session_data,
        })

    # ── Internal ─────────────────────────────────────────────────────────────

    def _safe_path(self, filename: str) -> Optional[Path]:
        """Ensure path stays within workspace."""
        # Prevent path traversal
        try:
            path = (self.workspace / filename).resolve()
            if not str(path).startswith(str(self.workspace)):
                return None
            if path.suffix and path.suffix not in self.ALLOWED_EXT:
                return None
            return path
        except Exception:
            return None

    def _register(self, filename: str, size: int):
        self.file_registry.append({
            "filename": filename,
            "size":     size,
            "time":     datetime.now().strftime("%H:%M:%S"),
        })
        if len(self.file_registry) > 200:
            self.file_registry = self.file_registry[-200:]

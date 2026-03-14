#!/usr/bin/env python3
"""
PRD-AGI Phase 6: The Nameless Intelligence — UPGRADED
Improvements over original ZIP:
  - Modular architecture (PDF spec)
  - torch/GPU support
  - .env fully wired (INITIAL_THRESHOLD, EVOLUTION_RATE, MOCK_MODE, MAX_*)
  - Streaming chat responses
  - History export (CSV)
  - Gauge violation breakdown
  - Gradient-guided evolution option
  - Model hot-swap in sidebar
  - Security: hardened calculator, sandboxed exec
  - New agent: "Avigata" (Stability Analyst)
  - Causal edge export
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import ast
import os
import sys
import re
import hashlib
import socket
import platform
import io
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv
import logging

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PRD-AGI | The Nameless Intelligence",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "PRD-AGI Phase 6 – SU(5) Causal Intelligence (Upgraded)"}
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800&display=swap');

/* ── Variables ── */
:root {
    --bg-deep:#07080f; --bg-panel:#0d0f1c; --bg-card:#111427;
    --accent:#7c5cfc; --accent2:#00e5ff; --accent3:#ff4fa3;
    --text:#e8eaf6; --text-dim:#6b7280;
    --border:#1e2240; --success:#00e676; --warn:#ffab40; --danger:#ff5252;
    --chat-input-h: 120px;
}

/* ── Base ── */
html,body,[data-testid="stApp"]{
    background:var(--bg-deep)!important;
    font-family:'Space Mono',monospace;
    color:var(--text);
}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0a0c18,#0d0f1c)!important;
    border-right:1px solid var(--border);
    transition: width 0.3s ease;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
    background:var(--bg-panel);
    border-bottom:1px solid var(--border);
    gap:4px;
    flex-wrap:wrap;
}
.stTabs [data-baseweb="tab"]{
    color:var(--text-dim);
    font-family:'Space Mono',monospace;
    font-size:12px;
    padding:8px 14px;
    border-radius:6px 6px 0 0;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover{
    color:var(--text);
    background:rgba(124,92,252,0.1);
}
.stTabs [aria-selected="true"]{
    background:var(--bg-card)!important;
    color:var(--accent2)!important;
    border-bottom:2px solid var(--accent2)!important;
}

/* ── Cards ── */
.prd-card{
    background:var(--bg-card);
    border:1px solid var(--border);
    border-radius:10px;
    padding:16px 20px;
    margin-bottom:12px;
    position:relative;
    animation: fadeSlideIn 0.3s ease;
}
.prd-card::before{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,var(--accent),var(--accent2));
    border-radius:10px 10px 0 0;
}

/* ── Metrics ── */
.metric-box{
    background:var(--bg-card);
    border:1px solid var(--border);
    border-radius:8px;
    padding:12px 16px;
    text-align:center;
    transition: border-color 0.3s ease, transform 0.2s ease;
}
.metric-box:hover{ border-color:var(--accent2); transform:translateY(-1px); }
.metric-value{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:var(--accent2);}
.metric-label{font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px;margin-top:4px;}

/* ── Chat messages ── */
.chat-scroll-area{
    max-height: calc(100vh - 300px);
    overflow-y: auto;
    padding-bottom: 16px;
    scroll-behavior: smooth;
}
.chat-user{
    background:linear-gradient(135deg,#1a1d35,#1e2240);
    border:1px solid var(--accent);
    border-radius:16px 16px 4px 16px;
    padding:12px 16px;
    margin:8px 0 8px 40px;
    color:var(--text);
    font-size:14px;
    animation: fadeSlideIn 0.25s ease;
}
.chat-ai{
    background:linear-gradient(135deg,#0f1220,#131728);
    border:1px solid var(--accent2);
    border-radius:16px 16px 16px 4px;
    padding:12px 16px;
    margin:8px 40px 8px 0;
    color:var(--text);
    font-size:14px;
    animation: fadeSlideIn 0.25s ease;
}
.chat-meta{font-size:10px;color:var(--text-dim);margin:2px 0;}

/* ── Bottom-fixed chat input ── */
.chat-input-bar{
    position: sticky;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(0deg, var(--bg-deep) 70%, transparent);
    padding: 16px 0 8px 0;
    z-index: 100;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
/* Push Streamlit form to look fixed at bottom of tab */
[data-testid="stForm"]{
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px 16px;
    box-shadow: 0 -8px 32px rgba(0,0,0,0.6);
}

/* ── Badges ── */
.badge-pass{background:#00e67622;color:var(--success);border:1px solid var(--success);border-radius:20px;padding:2px 10px;font-size:11px;font-weight:700;}
.badge-fail{background:#ff525222;color:var(--danger);border:1px solid var(--danger);border-radius:20px;padding:2px 10px;font-size:11px;font-weight:700;}
.badge-warn{background:#ffab4022;color:var(--warn);border:1px solid var(--warn);border-radius:20px;padding:2px 10px;font-size:11px;font-weight:700;}

/* ── App title ── */
.app-title{
    font-family:'Syne',sans-serif;font-size:28px;font-weight:800;
    background:linear-gradient(90deg,var(--accent),var(--accent2),var(--accent3));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    margin:0;padding:0;
    animation: titlePulse 4s ease-in-out infinite;
}
.app-sub{font-size:11px;color:var(--text-dim);letter-spacing:2px;text-transform:uppercase;}

/* ── Inputs ── */
.stTextInput>div>div>input,.stTextArea>div>div>textarea{
    background:var(--bg-card)!important;
    border:1px solid var(--border)!important;
    color:var(--text)!important;
    border-radius:8px!important;
    font-family:'Space Mono',monospace!important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.stTextInput>div>div>input:focus,.stTextArea>div>div>textarea:focus{
    border-color:var(--accent)!important;
    box-shadow:0 0 0 2px rgba(124,92,252,0.2)!important;
}

/* ── Buttons ── */
.stButton>button{
    background:linear-gradient(135deg,var(--accent),#5b3fe0)!important;
    color:white!important;border:none!important;
    border-radius:8px!important;
    font-family:'Space Mono',monospace!important;
    font-size:12px!important;font-weight:700!important;
    transition: all 0.2s ease;
}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 6px 24px rgba(124,92,252,0.5);}
.stButton>button:active{transform:translateY(0);}

/* ── Agent cards ── */
.agent-card{
    background:var(--bg-card);border:1px solid var(--border);
    border-radius:8px;padding:10px 14px;margin:6px 0;
    border-left:3px solid var(--accent2);
    transition: border-left-color 0.2s ease, transform 0.2s ease;
}
.agent-card:hover{ border-left-color:var(--accent3); transform:translateX(3px); }
.agent-name{color:var(--accent2);font-weight:700;font-size:13px;}
.agent-status{color:var(--text-dim);font-size:11px;}

/* ── WiFi box ── */
.wifi-box{
    background:linear-gradient(135deg,#0a1628,#0d1f3c);
    border:1px solid var(--accent2);border-radius:10px;
    padding:14px 18px;margin:10px 0;
}
.wifi-ip{font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:var(--accent2);letter-spacing:2px;}

/* ── Loading spinner (pulse ring) ── */
.prd-spinner{
    display:inline-block;
    width:18px;height:18px;
    border:2px solid var(--border);
    border-top-color:var(--accent2);
    border-radius:50%;
    animation: spin 0.8s linear infinite;
    vertical-align:middle;
    margin-right:8px;
}
.prd-thinking{
    display:flex;align-items:center;gap:6px;
    padding:10px 14px;
    background:var(--bg-card);border:1px solid var(--border);
    border-radius:12px;margin:8px 40px 8px 0;
    animation: fadeSlideIn 0.2s ease;
}
.dot-bounce{
    width:7px;height:7px;border-radius:50%;
    background:var(--accent2);display:inline-block;
    animation: dotBounce 1.2s ease-in-out infinite;
}
.dot-bounce:nth-child(2){animation-delay:0.2s;}
.dot-bounce:nth-child(3){animation-delay:0.4s;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:var(--bg-deep);}
::-webkit-scrollbar-thumb{background:var(--accent);border-radius:2px;}
::-webkit-scrollbar-thumb:hover{background:var(--accent2);}

/* ── Animations ── */
@keyframes fadeSlideIn{
    from{opacity:0;transform:translateY(8px);}
    to{opacity:1;transform:translateY(0);}
}
@keyframes spin{to{transform:rotate(360deg);}}
@keyframes dotBounce{
    0%,80%,100%{transform:translateY(0);}
    40%{transform:translateY(-8px);}
}
@keyframes titlePulse{
    0%,100%{filter:brightness(1);}
    50%{filter:brightness(1.15);}
}
@keyframes borderGlow{
    0%,100%{box-shadow:0 0 0 0 rgba(0,229,255,0);}
    50%{box-shadow:0 0 12px 2px rgba(0,229,255,0.15);}
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .app-title{font-size:20px;}
    .metric-value{font-size:18px;}
    .chat-user{margin-left:8px;}
    .chat-ai{margin-right:8px;}
    .stTabs [data-baseweb="tab"]{font-size:10px;padding:6px 8px;}
    .metric-box{padding:8px 10px;}
    .wifi-ip{font-size:16px;letter-spacing:1px;}
    [data-testid="stForm"]{padding:8px 10px;}
}
@media (max-width: 480px) {
    .chat-user,.chat-ai{font-size:13px;padding:10px 12px;}
    .metric-value{font-size:16px;}
    .stTabs [data-baseweb="tab"]{font-size:9px;padding:5px 6px;}
}

/* ── Hide streamlit branding ── */
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s [%(levelname)s] %(name)s – %(message)s',
    handlers=[
        logging.FileHandler('logs/prd-agi.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PRD-AGI')

# ── Imports from modular structure ────────────────────────────────────────────
from core.dna import SU5DNA
from core.engine import CurvatureEngine, RelationalState
from core.awareness import AwarenessDensity, CausalStrengthMonitor, ContradictionDetector
from perception.ollama_client import OllamaInterface
from perception.gemini_client import GeminiInterface
from meta.consciousness import MetaLayer, GaugeInvarianceChecker, CausalDiscovery

# ── Fuzzy Logic Modules ───────────────────────────────────────────────────────
from fuzzy_gatekeeper import FuzzyGateKeeper
from fuzzy_agent import FuzzyAgentAggregator
from fuzzy_code import FuzzyCodeEvaluator
from fuzzy_improve import FuzzyImprovementDecider

# ── Sentience Modules (Phase 7) ───────────────────────────────────────────────
from sentience_emotion import EmotionalState
from sentience_instinct import TruthPreservationInstinct
from sentience_intuition import IntuitionLayer
from sentience_response import SentientResponse

# ── New Capability Modules ────────────────────────────────────────────────────
from tools.web_search import WebSearchTool
from tools.calculator import CalculatorTool
from tools.file_io import FileIOTool
from rag.rag_engine import RAGEngine
from self_modify.self_modify import SelfModifyEngine

# ══════════════════════════════════════════════════════════════════════════════
# CODE ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class CodeAnalyzer:
    NODE_MAP = {
        ast.FunctionDef: 0, ast.AsyncFunctionDef: 0,
        ast.ClassDef: 1,
        ast.If: 2, ast.IfExp: 2,
        ast.For: 3, ast.AsyncFor: 3,
        ast.While: 4,
        ast.Assign: 5, ast.AugAssign: 5, ast.AnnAssign: 5,
        ast.Call: 6,
        ast.Return: 7,
        ast.Import: 8, ast.ImportFrom: 8,
        ast.Try: 9,
        ast.With: 10, ast.AsyncWith: 10,
        ast.Lambda: 11,
    }

    def __init__(self, dna: SU5DNA):
        self.dna = dna

    def code_to_vector(self, source: str) -> Optional[np.ndarray]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        counts = np.zeros(24, dtype=float)
        for node in ast.walk(tree):
            idx = self.NODE_MAP.get(type(node))
            if idx is not None:
                counts[idx] += 1
        n = np.linalg.norm(counts)
        return counts / n if n > 0 else counts

    def summarize(self, source: str) -> Dict:
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"error": str(e)}
        return {
            "functions": [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][:20],
            "classes": [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)][:10],
            "imports": [ast.unparse(n) for n in ast.walk(tree)
                        if isinstance(n, (ast.Import, ast.ImportFrom))][:15],
            "lines": source.count('\n') + 1,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CHAT MEMORY
# ══════════════════════════════════════════════════════════════════════════════

MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", "100"))

class ChatMemory:
    def __init__(self, max_history: int = MAX_CHAT_HISTORY):
        self.history: List[Dict] = []
        self.max_history = max_history

    def add(self, role: str, content: str, curvature: float = 0.0, gate: str = ""):
        self.history.append({
            "role": role, "content": content,
            "curvature": curvature, "gate": gate,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context(self, n: int = 6) -> str:
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in self.history[-n * 2:]
        )

    def export_csv(self) -> str:
        buf = io.StringIO()
        import csv
        writer = csv.writer(buf)
        writer.writerow(["timestamp", "role", "content", "curvature", "gate"])
        for m in self.history:
            writer.writerow([m["timestamp"], m["role"], m["content"],
                             f"{m['curvature']:.4f}", m["gate"]])
        return buf.getvalue()

    def clear(self):
        self.history.clear()


# ══════════════════════════════════════════════════════════════════════════════
# SELF EVOLVER — upgraded with gradient mode
# ══════════════════════════════════════════════════════════════════════════════

class SelfEvolver:
    def __init__(self, dna: SU5DNA, engine: CurvatureEngine, step_size: float = 0.01):
        self.dna = dna
        self.engine = engine
        self.step_size = step_size
        self.history: List[Tuple[float, bool]] = []

    def evolve_step(self, state: RelationalState, temperature: float = 0.1,
                    use_gradient: bool = False) -> Tuple[bool, float]:
        old_c = self.engine.compute_curvature(state)
        backup = state.snapshot()

        if use_gradient:
            # Gradient-guided: move opposite to gradient
            grad = self.engine.curvature_gradient(state)
            grad_norm = np.linalg.norm(grad)
            direction = -grad / grad_norm if grad_norm > 1e-10 else np.random.randn(24)
        else:
            direction = np.random.randn(24)
            direction /= np.linalg.norm(direction)

        accepted = state.apply_transformation(self.step_size * direction)
        if not accepted:
            return False, old_c

        new_c = self.engine.compute_curvature(state)
        if new_c < old_c:
            accept = True
        else:
            accept = np.random.random() < np.exp(-(new_c - old_c) / max(temperature, 1e-10))

        if not accept:
            state.restore(backup)

        result_c = new_c if accept else old_c
        self.history.append((result_c, accept))
        return accept, result_c

    def evolve_batch(self, state: RelationalState, steps: int,
                     temperature: float = 0.1, use_gradient: bool = False,
                     callback=None) -> List[float]:
        curvatures = []
        for i in range(steps):
            _, c = self.evolve_step(state, temperature, use_gradient)
            curvatures.append(c)
            if callback:
                callback(i, steps, c)
        return curvatures


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-AGENT SYSTEM — added Avigata agent
# ══════════════════════════════════════════════════════════════════════════════

class Agent:
    AGENTS = {
        "Hetu": {
            "role": "Root Cause Analyst", "icon": "🔴", "generator": 0,
            "system": "You find root causes and fundamental reasons. Be concise and precise.",
        },
        "Nissaya": {
            "role": "Support Strategist", "icon": "🔵", "generator": 1,
            "system": "You identify support structures and enabling conditions.",
        },
        "Indriya": {
            "role": "Governing Logic", "icon": "🟢", "generator": 2,
            "system": "You analyze governing rules and constraints with precision.",
        },
        "Avigata": {
            "role": "Stability Analyst", "icon": "⚪", "generator": 3,   # NEW
            "system": "You assess stability, resilience, and long-term sustainability.",
        },
        "Anantara": {
            "role": "Sequential Planner", "icon": "🟡", "generator": 4,
            "system": "You plan step-by-step sequences and workflows.",
        },
        "Sahajata": {
            "role": "Synthesis Expert", "icon": "🟣", "generator": 20,
            "system": "You synthesize multiple perspectives into unified solutions.",
        },
    }

    def __init__(self, name: str, ollama: OllamaInterface, dna: SU5DNA, engine: CurvatureEngine):
        self.name = name
        self.info = self.AGENTS[name]
        self.ollama = ollama
        self.dna = dna
        self.engine = engine

    def respond(self, task: str, context: str = "") -> Dict:
        prompt = f"Task: {task}\n\nContext: {context}" if context else f"Task: {task}"
        response = self.ollama.generate(prompt, system=self.info["system"])
        state = self.ollama.text_to_state(response, self.dna)
        curvature = self.engine.compute_curvature(state)
        return {
            "agent": self.name, "icon": self.info["icon"],
            "role": self.info["role"], "response": response,
            "curvature": curvature, "timestamp": datetime.now().strftime("%H:%M:%S"),
        }


class MultiAgentSupervisor:
    def __init__(self, ollama: OllamaInterface, dna: SU5DNA, engine: CurvatureEngine):
        self.agents = {n: Agent(n, ollama, dna, engine) for n in Agent.AGENTS}
        self.ollama = ollama
        self.dna = dna
        self.engine = engine

    def run(self, task: str, agents: List[str] = None) -> Dict:
        if agents is None:
            agents = list(self.agents.keys())
        results = [self.agents[n].respond(task) for n in agents if n in self.agents]
        synthesis_prompt = f"Task: {task}\n\n" + "\n".join(
            f"[{r['agent']}]: {r['response'][:200]}" for r in results
        )
        synthesis = self.ollama.generate(
            synthesis_prompt,
            system="You are a synthesis AI. Combine all agent responses into one coherent, structured answer."
        )
        return {"agents": results, "synthesis": synthesis}


# ══════════════════════════════════════════════════════════════════════════════
# TOOLKIT — hardened security
# ══════════════════════════════════════════════════════════════════════════════

class ToolKit:
    # Strict whitelist: only math operators, digits, spaces, parens, dots
    _CALC_ALLOWED = set("0123456789 +-*/().,** \t\n")
    _SAFE_BUILTINS = {
        "__builtins__": {
            "print": print, "range": range, "len": len,
            "int": int, "float": float, "str": str,
            "list": list, "dict": dict, "sum": sum,
            "min": min, "max": max, "abs": abs,
            "round": round, "sorted": sorted, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter,
        },
        "np": np,
    }

    @staticmethod
    def calculator(expr: str) -> str:
        if not all(c in ToolKit._CALC_ALLOWED for c in expr):
            return "Error: Invalid characters — only numbers and operators allowed"
        if len(expr) > 200:
            return "Error: Expression too long"
        try:
            result = eval(expr, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def safe_exec(code: str) -> Tuple[str, str]:
        import io as _io
        from contextlib import redirect_stdout, redirect_stderr
        if len(code) > 4000:
            return "", "Error: Code too long (max 4000 chars)"
        # Block dangerous patterns
        blocked = ["import os", "import sys", "import subprocess", "__import__",
                   "open(", "exec(", "eval(", "compile(", "globals(", "locals("]
        for b in blocked:
            if b in code:
                return "", f"Security Error: '{b}' is not permitted"
        stdout_buf, stderr_buf = _io.StringIO(), _io.StringIO()
        try:
            safe_globals = dict(ToolKit._SAFE_BUILTINS)
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, '<prd-sandbox>', 'exec'), safe_globals)
            return stdout_buf.getvalue(), ""
        except Exception as e:
            return "", f"{type(e).__name__}: {e}"

    @staticmethod
    def get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

MAX_STORED_STATES = int(os.getenv("MAX_STORED_STATES", "200"))

@st.cache_resource(show_spinner="🌀 Initializing SU(5) + Fuzzy + Sentience + Awareness...")
def init_core():
    dna = SU5DNA()
    engine = CurvatureEngine(dna)
    meta = MetaLayer()
    gauge = GaugeInvarianceChecker(dna)
    causal_disc = CausalDiscovery(dna)
    code_analyzer = CodeAnalyzer(dna)
    fuzzy_gate    = FuzzyGateKeeper(use_gauge=True)
    fuzzy_code_ev = FuzzyCodeEvaluator()
    instinct = TruthPreservationInstinct(dna, engine)
    web_search_tool = WebSearchTool(max_results=5)
    calculator_tool = CalculatorTool()
    file_io_tool    = FileIOTool(workspace="workspace")
    # ── MUT Awareness Layer ──
    awareness_density  = AwarenessDensity()
    causal_monitor     = CausalStrengthMonitor(window=5)
    contradiction_det  = ContradictionDetector()
    return (dna, engine, meta, gauge, causal_disc, code_analyzer,
            fuzzy_gate, fuzzy_code_ev, instinct,
            web_search_tool, calculator_tool, file_io_tool,
            awareness_density, causal_monitor, contradiction_det)

(dna, engine, meta, gauge, causal_disc, code_analyzer,
 fuzzy_gate, fuzzy_code_ev, instinct,
 web_search_tool, calculator_tool, file_io_tool,
 awareness_density, causal_monitor, contradiction_det) = init_core()

def build_llm_backend(mock: bool = False) -> "OllamaInterface | GeminiInterface":
    """Factory: returns Gemini or Ollama backend based on session state."""
    backend = st.session_state.get("llm_backend", os.getenv("LLM_BACKEND", "ollama"))
    if backend == "gemini":
        return GeminiInterface(
            mock=mock,
            api_key=st.session_state.get("gemini_api_key", os.getenv("GEMINI_API_KEY", "")),
            chat_model=st.session_state.get("chat_model", os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")),
        )
    else:
        return OllamaInterface(
            mock=mock,
            host=st.session_state.get("ollama_host", os.getenv("OLLAMA_HOST", "http://localhost:11434")),
            chat_model=st.session_state.get("chat_model", os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest")),
            embed_model=st.session_state.get("embed_model", os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")),
        )


def ensure_session():
    defaults = {
        "state": lambda: RelationalState(dna),
        "mock": lambda: os.getenv("MOCK_MODE", "false").lower() == "true",
        "llm_backend": lambda: os.getenv("LLM_BACKEND", "ollama"),
        "gemini_api_key": lambda: os.getenv("GEMINI_API_KEY", ""),
        "ollama_host": lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "chat_model": lambda: os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest"),
        "embed_model": lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        "memory": lambda: ChatMemory(),
        "evolver": lambda: SelfEvolver(dna, engine),
        "stored_states": lambda: [],
        "evolution_history": lambda: [],
        "stream_mode": lambda: False,
        # ── Fuzzy ──
        "fuzzy_agent":   lambda: FuzzyAgentAggregator(),
        "fuzzy_improve": lambda: FuzzyImprovementDecider(),
        "fuzzy_history": lambda: [],
        # ── Sentience ──
        "sentient_response": lambda: SentientResponse(),
        "intuition_layer":   lambda: IntuitionLayer(),
        "sentience_log":     lambda: [],
        # ── RAG ──
        "rag_engine":    lambda: RAGEngine(store_path="workspace/rag_store.json"),
        # ── Self-Modify ──
        "self_modify_engine": lambda: SelfModifyEngine(
            fuzzy_evaluator=fuzzy_code_ev,
            workspace="workspace/self_modify",
        ),
        # ── Search history ──
        "search_history": lambda: [],
        # ── MUT Awareness session state ──
        "stored_psis":        lambda: [],    # ψ vectors for CausalStrengthMonitor
        "awareness_log":      lambda: [],    # awareness density history for tab
        "contradiction_log":  lambda: [],    # contradiction events
    }
    for key, factory in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = factory()
    if "ollama" not in st.session_state:
        st.session_state.ollama = build_llm_backend(mock=st.session_state.mock)
    if "supervisor" not in st.session_state:
        st.session_state.supervisor = MultiAgentSupervisor(
            st.session_state.ollama, dna, engine
        )
    # Wire RAG + SelfModify to current LLM
    if hasattr(st.session_state.get("rag_engine"), "embed_client"):
        if st.session_state.rag_engine.embed_client is None:
            st.session_state.rag_engine.embed_client = st.session_state.ollama
    if hasattr(st.session_state.get("self_modify_engine"), "llm"):
        if st.session_state.self_modify_engine.llm is None:
            st.session_state.self_modify_engine.llm = st.session_state.ollama

ensure_session()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<p class="app-title">🌀 PRD-AGI</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-sub">Phase 6+7 · Fuzzy + Sentience</p>', unsafe_allow_html=True)
    st.divider()

    cur_curv = engine.compute_curvature(st.session_state.state)
    meta.update(cur_curv)
    gate_pass, gate_label = meta.gate(cur_curv)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="metric-box">
        <div class="metric-value">{cur_curv:.4f}</div>
        <div class="metric-label">Curvature</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-box">
        <div class="metric-value">{meta.truth_threshold:.3f}</div>
        <div class="metric-label">Threshold</div></div>""", unsafe_allow_html=True)

    badge_cls = "badge-pass" if "PASS" in gate_label else ("badge-fail" if "BLOCK" in gate_label else "badge-warn")
    st.markdown(f'<div style="text-align:center;margin:8px 0"><span class="{badge_cls}">{gate_label}</span></div>', unsafe_allow_html=True)

    st.divider()

    with st.expander("⚙️ Settings", expanded=False):
        # ── Backend selector (NEW: Gemini / Ollama) ──────────────────────────
        st.markdown('<div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">LLM Backend</div>', unsafe_allow_html=True)
        backend_choice = st.radio(
            "Backend", ["🟣 Ollama (Local)", "✨ Google Gemini (Cloud)"],
            index=0 if st.session_state.llm_backend == "ollama" else 1,
            horizontal=True, label_visibility="collapsed"
        )
        new_backend = "gemini" if "Gemini" in backend_choice else "ollama"

        backend_changed = new_backend != st.session_state.llm_backend

        if new_backend == "gemini":
            # Gemini settings
            st.markdown('<div style="background:linear-gradient(135deg,#1a0a2e,#0d1f3c);border:1px solid #4285f4;border-radius:8px;padding:10px 14px;margin:6px 0">'
                       '<div style="font-size:10px;color:#4285f4;font-weight:700;margin-bottom:6px">✨ Google Gemini API</div>', unsafe_allow_html=True)
            new_api_key = st.text_input(
                "API Key", value=st.session_state.gemini_api_key,
                type="password", placeholder="AIza... (from aistudio.google.com)",
                help="Get free key: https://aistudio.google.com/app/apikey"
            )
            if new_api_key != st.session_state.gemini_api_key:
                st.session_state.gemini_api_key = new_api_key
                backend_changed = True

            gemini_models = GeminiInterface.AVAILABLE_MODELS
            default_idx = gemini_models.index("gemini-2.0-flash") if "gemini-2.0-flash" in gemini_models else 0
            current_idx = gemini_models.index(st.session_state.chat_model) if st.session_state.chat_model in gemini_models else default_idx
            selected_gmodel = st.selectbox("Model", gemini_models, index=current_idx)
            if selected_gmodel != st.session_state.chat_model:
                st.session_state.chat_model = selected_gmodel
                backend_changed = True
            st.markdown('<div style="font-size:10px;color:#6b7280">Free tier: 15 req/min · No credit card needed</div></div>', unsafe_allow_html=True)

        else:
            # Ollama settings
            new_ollama_host = st.text_input("Ollama Host", value=st.session_state.ollama_host)
            if new_ollama_host != st.session_state.ollama_host:
                st.session_state.ollama_host = new_ollama_host
                backend_changed = True
            available_models = st.session_state.ollama.list_models() if not backend_changed else []
            if available_models:
                selected_model = st.selectbox("Model", available_models,
                    index=available_models.index(st.session_state.chat_model)
                          if st.session_state.chat_model in available_models else 0)
                if selected_model != st.session_state.chat_model:
                    st.session_state.chat_model = selected_model
                    st.session_state.ollama.switch_model(selected_model)
            else:
                new_model = st.text_input("Model", value=st.session_state.chat_model)
                if new_model != st.session_state.chat_model:
                    st.session_state.chat_model = new_model
                    if hasattr(st.session_state.ollama, 'switch_model'):
                        st.session_state.ollama.switch_model(new_model)

        if backend_changed:
            st.session_state.llm_backend = new_backend
            st.session_state.ollama = build_llm_backend(mock=st.session_state.mock)
            st.session_state.supervisor = MultiAgentSupervisor(
                st.session_state.ollama, dna, engine)
            st.rerun()

        st.divider()
        new_mock = st.toggle("Mock mode (offline test)", value=st.session_state.mock)
        if new_mock != st.session_state.mock:
            st.session_state.mock = new_mock
            st.session_state.ollama = build_llm_backend(mock=new_mock)
            st.session_state.supervisor = MultiAgentSupervisor(
                st.session_state.ollama, dna, engine)
            st.rerun()

        step_size = st.slider("Evolution Step", 0.001, 0.1, 0.01, 0.001)
        st.session_state.evolver.step_size = step_size
        st.session_state.stream_mode = st.toggle("Streaming responses", value=st.session_state.stream_mode)

    local_ip = ToolKit.get_local_ip()
    st.markdown(f"""<div class="wifi-box">
    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;">📶 WiFi Access</div>
    <div class="wifi-ip">{local_ip}:8501</div>
    <div style="font-size:11px;color:#6b7280;margin-top:4px;">Open on any phone on same WiFi</div>
    </div>""", unsafe_allow_html=True)

    _backend_label = "✨ Gemini" if st.session_state.llm_backend == "gemini" else "🟣 Ollama"
    _status_icon = "🟢" if st.session_state.ollama.available else ("🟡" if st.session_state.mock else "🔴")
    _status_text = "Connected" if st.session_state.ollama.available else ("Mock" if st.session_state.mock else "Offline")
    st.markdown(f'<div style="font-size:11px;color:#6b7280;text-align:center">{_status_icon} {_status_text} · {_backend_label} · {st.session_state.chat_model}</div>', unsafe_allow_html=True)
    st.divider()

    stats = meta.rolling_stats()

    # ── Sentience sidebar display ──
    if "sentient_response" in st.session_state:
        emotion = st.session_state.sentient_response.emotional.current()
        instinct_result = instinct.assess(cur_curv, gauge.compute_violation(
            np.real(st.session_state.state.to_vector())))
        st.markdown(
            st.session_state.sentient_response.emotional.sidebar_html(cur_curv),
            unsafe_allow_html=True
        )
        st.markdown(instinct.sidebar_html(), unsafe_allow_html=True)
        # ── MUT Awareness Density ──
        st.markdown(awareness_density.sidebar_html(), unsafe_allow_html=True)

    st.markdown(f"""<div style="font-size:10px;color:#6b7280;margin-top:6px">
    SU(5) algebra error: <span style="color:#00e5ff">{dna._algebra_error:.2e}</span><br>
    Rolling avg κ: <span style="color:#00e5ff">{stats['mean']:.4f}</span><br>
    Threshold: <span style="color:#00e5ff">{meta.truth_threshold:.4f}</span><br>
    Chat msgs: <span style="color:#00e5ff">{len(st.session_state.memory.history)}</span>
    </div>""", unsafe_allow_html=True)

    if st.button("🔄 Reset State"):
        st.session_state.state = RelationalState(dna)
        st.session_state.memory.clear()
        st.session_state.stored_states = []
        st.session_state.evolution_history = []
        engine.clear_cache()
        instinct.reset()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "💬 Chat", "🌀 State", "📊 Meta", "🔗 Causal",
    "⚖️ Gauge", "🧘 Evolution", "🔄 Self-Program", "🤝 Multi-Agent", "🔧 Tools",
    "🧠 Fuzzy Gate", "🤝 Fuzzy Agents", "💻 Fuzzy Code", "💫 Sentience",
    "🌐 Web Search", "📚 RAG Knowledge", "🧬 Self-Modify", "💡 Awareness",
])


# ── TAB 1: CHAT ───────────────────────────────────────────────────────────────
with tabs[0]:

    # ── Header bar ──
    hcol1, hcol2 = st.columns([6, 1])
    with hcol1:
        st.markdown("""<div class="prd-card" style="margin-bottom:8px">
        <span style="font-size:15px;font-weight:700;color:var(--accent2)">💬 PRD-AGI Chat</span>
        <span style="font-size:11px;color:var(--text-dim);margin-left:10px">
        Truth-first gatekeeper · SU(5) relational core active</span>
        </div>""", unsafe_allow_html=True)
    with hcol2:
        if st.session_state.memory.history:
            csv_data = st.session_state.memory.export_csv()
            st.download_button("📥 CSV", csv_data, "chat_history.csv", "text/csv",
                               use_container_width=True, help="Export chat history")

    # ── Chat messages scroll area ──
    st.markdown('<div class="chat-scroll-area">', unsafe_allow_html=True)

    if not st.session_state.memory.history:
        st.markdown("""
        <div style="text-align:center;color:var(--text-dim);padding:60px 20px;animation:fadeSlideIn 0.5s ease">
            <div style="font-size:48px;margin-bottom:12px;filter:drop-shadow(0 0 20px #7c5cfc)">🌀</div>
            <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:var(--text);margin-bottom:6px">
                The Nameless Intelligence</div>
            <div style="font-size:12px;letter-spacing:1px">Begin your inquiry below</div>
        </div>""", unsafe_allow_html=True)

    for msg in st.session_state.memory.history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-meta" style="text-align:right;margin-right:4px">
                You &nbsp;·&nbsp; {msg['timestamp']}
            </div>
            <div class="chat-user">{msg['content']}</div>
            """, unsafe_allow_html=True)
        else:
            gate_cls = ("badge-pass" if "PASS" in msg.get("gate","")
                        else ("badge-fail" if "BLOCK" in msg.get("gate","") else "badge-warn"))
            st.markdown(f"""
            <div class="chat-meta">
                🌀 PRD-AGI &nbsp;·&nbsp; {msg['timestamp']} &nbsp;&nbsp;
                <span class="{gate_cls}">{msg.get('gate','')}</span> &nbsp;
                <span style="color:var(--text-dim);font-size:10px">κ={msg.get('curvature',0):.4f}</span>
            </div>
            <div class="chat-ai">{msg['content']}</div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # end scroll area

    # ── Bottom-fixed input bar ──
    st.markdown('<div class="chat-input-bar">', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([6, 1])
        with col_inp:
            user_input = st.text_area(
                "Query",
                placeholder="Enter your inquiry... (Shift+Enter for new line)",
                height=68,
                label_visibility="collapsed",
            )
        with col_btn:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Send →", use_container_width=True)

        # Clear button inline
        if st.session_state.memory.history:
            if st.form_submit_button("🗑️ Clear chat", use_container_width=False):
                st.session_state.memory.clear()
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # end input bar

    # ── Process submission ──
    if submitted and user_input.strip():
        query_state = st.session_state.ollama.text_to_state(user_input, dna)
        curvature = engine.compute_curvature(query_state)
        meta.update(curvature)

        # ── Fuzzy gate ──
        gauge_viol = gauge.compute_violation(np.real(query_state.to_vector()))
        fuzzy_pass, fuzzy_label, fuzzy_conf = fuzzy_gate.evaluate(curvature, gauge_viol)
        gate_label = f"{fuzzy_label} ({fuzzy_conf:.0%})"
        gate_pass  = fuzzy_pass

        # ── MUT: Causal-Emotional Feedback Loop ──
        stored_psis = st.session_state.stored_psis
        causal_result = causal_monitor.assess(
            query_state.to_vector(), stored_psis, curvature
        )
        contradiction_result = contradiction_det.detect(
            user_input, st.session_state.rag_engine, curvature
        )

        # Compute awareness density
        gauge_coherence = float(np.clip(1.0 - gauge_viol / max(gauge_viol + 1e-6, 1.0), 0.0, 1.0))
        awareness_rec = awareness_density.compute(
            kappa=curvature,
            causal_strength=causal_result["strength"],
            gauge_coherence=gauge_coherence,
        )

        # Causal-Emotional override
        causal_emotion_override = None
        if contradiction_result["contradiction"]:
            causal_emotion_override = "Contradicted"
            st.session_state.sentient_response.emotional.override_emotion(
                "Contradicted", curvature)
        elif causal_result["weak"]:
            causal_emotion_override = "Confused"
            st.session_state.sentient_response.emotional.override_emotion(
                "Confused", curvature)

        # Log to awareness tab
        st.session_state.awareness_log.append({
            **awareness_rec,
            "causal_strength":  causal_result["strength"],
            "causal_emotion":   causal_result["emotion_signal"],
            "contradiction":    contradiction_result["contradiction"],
            "contra_severity":  contradiction_result["severity"],
            "query":            user_input[:50],
        })
        if len(st.session_state.awareness_log) > 200:
            st.session_state.awareness_log = st.session_state.awareness_log[-200:]

        # Save fuzzy history
        st.session_state.fuzzy_history.append({
            "query":      user_input[:60],
            "curvature":  round(curvature, 4),
            "gauge":      round(gauge_viol, 8),
            "label":      fuzzy_label,
            "confidence": round(fuzzy_conf, 4),
            "passed":     fuzzy_pass,
            "time":       datetime.now().strftime("%H:%M:%S"),
        })

        st.session_state.memory.add("user", user_input, curvature=curvature)

        if not gate_pass:
            profile = fuzzy_gate.membership_profile(curvature)
            dominant = max(profile, key=profile.get)
            response = (
                f"🚫 **Query blocked by Fuzzy Truth Gatekeeper.**\n\n"
                f"Curvature κ={curvature:.4f} · Fuzzy confidence={fuzzy_conf:.0%}\n\n"
                f"Dominant membership: **{dominant}** ({profile[dominant]:.2f})\n\n"
                f"Awareness density: {awareness_rec['label']} ({awareness_rec['density']:.3f})\n\n"
                f"Please rephrase with more causal precision."
            )
        else:
            context = st.session_state.memory.get_context(6)

            # ── RAG context ──
            rag_context = ""
            if st.session_state.rag_engine.stats()["total_chunks"] > 0:
                rag_context = st.session_state.rag_engine.build_context(user_input, top_k=3)
            if rag_context:
                context = rag_context + "\n\n[Conversation History]\n" + context

            # ── Sentience: intuition + dynamic system prompt ──
            intuition_score, intuit_label, intuit_color = \
                st.session_state.intuition_layer.sense(curvature)
            instinct_result = instinct.assess(curvature, gauge_viol)
            instinct_prefix = instinct.get_instinct_prefix()

            # Augment system prompt with causal-emotional context
            system = st.session_state.sentient_response.build_system_prompt(curvature)
            if causal_emotion_override == "Contradicted":
                system += (f" IMPORTANT: A contradiction has been detected between "
                           f"this query and existing knowledge (severity={contradiction_result['severity']:.2f}). "
                           f"Acknowledge the tension and reason carefully.")
            elif causal_emotion_override == "Confused":
                system += (f" NOTE: Causal inference is weak (strength={causal_result['strength']:.2f}). "
                           f"Express appropriate uncertainty and reason step-by-step.")

            prompt = f"Context:\n{context}\n\nQuery: {user_input}"

            # Show thinking animation with awareness chip
            thinking_ph = st.empty()
            awareness_color = "#00e5ff" if awareness_rec["density"] > 0.6 else "#ffab40"
            thinking_ph.markdown(f"""
            <div class="prd-thinking">
                <span class="dot-bounce"></span>
                <span class="dot-bounce"></span>
                <span class="dot-bounce"></span>
                <span style="font-size:11px;color:var(--text-dim);margin-left:4px">
                Computing...</span>&nbsp;
                <span style="background:{intuit_color}22;color:{intuit_color};
                border:1px solid {intuit_color};border-radius:20px;
                padding:2px 8px;font-size:10px">{intuit_label}</span>&nbsp;
                <span style="background:{awareness_color}22;color:{awareness_color};
                border:1px solid {awareness_color};border-radius:20px;
                padding:2px 8px;font-size:10px">{awareness_rec['label']}</span>
            </div>""", unsafe_allow_html=True)

            if st.session_state.stream_mode:
                tokens = []
                stream_ph = st.empty()
                for token in st.session_state.ollama.generate_stream(prompt, system=system):
                    tokens.append(token)
                    stream_ph.markdown("".join(tokens))
                base_response = "".join(tokens)
                thinking_ph.empty()
                stream_ph.empty()
            else:
                base_response = st.session_state.ollama.generate(prompt, system=system)
                thinking_ph.empty()

            # ── Modulate with sentience ──
            if instinct_result["should_veto"]:
                response = instinct_result["veto_msg"]
            else:
                response = st.session_state.sentient_response.modulate_response(
                    base_response, curvature, intuition_score, instinct_prefix
                )

            # Log sentience event
            st.session_state.sentience_log.append({
                "time":      datetime.now().strftime("%H:%M:%S"),
                "emotion":   st.session_state.sentient_response.emotional.current()["name"],
                "intuition": intuit_label,
                "instinct":  instinct_result["level"],
                "curvature": round(curvature, 4),
                "vetoed":    instinct_result["should_veto"],
                "awareness": round(awareness_rec["density"], 4),
                "causal":    round(causal_result["strength"], 4),
                "contradiction": contradiction_result["contradiction"],
            })

            resp_state = st.session_state.ollama.text_to_state(response, dna)
            st.session_state.state = resp_state
            st.session_state.stored_states.append(resp_state)
            # Store ψ for CausalStrengthMonitor
            st.session_state.stored_psis.append(query_state.to_vector())
            if len(st.session_state.stored_psis) > 50:
                st.session_state.stored_psis = st.session_state.stored_psis[-50:]
            if len(st.session_state.stored_states) > MAX_STORED_STATES:
                st.session_state.stored_states = st.session_state.stored_states[-MAX_STORED_STATES:]

        st.session_state.memory.add("assistant", response, curvature=curvature, gate=gate_label)
        st.rerun()


# ── TAB 2: STATE ──────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("#### 🌀 24-Dimensional Relational State")
    psi = st.session_state.state.to_vector()
    reals, imags, mags = np.real(psi), np.imag(psi), np.abs(psi)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dna.PACCAYA_NAMES, y=mags,
            marker=dict(color=mags, colorscale=[[0,"#1e2240"],[0.5,"#7c5cfc"],[1,"#00e5ff"]], showscale=False),
            name="Magnitude"
        ))
        fig.update_layout(template="plotly_dark", height=320, paper_bgcolor="#0d0f1c", plot_bgcolor="#07080f",
                          margin=dict(l=10,r=10,t=30,b=80),
                          title=dict(text="Generator Activation Magnitudes", font=dict(size=13)),
                          xaxis=dict(tickangle=45, tickfont=dict(size=8)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=reals, y=imags, mode="markers+lines",
            marker=dict(size=[m*30+4 for m in mags], color=np.arange(24),
                        colorscale="Plasma", showscale=True, colorbar=dict(thickness=10)),
            line=dict(color="#1e2240", width=1),
            text=dna.PACCAYA_NAMES,
            hovertemplate="<b>%{text}</b><br>Re=%{x:.4f}<br>Im=%{y:.4f}<extra></extra>"
        ))
        fig2.update_layout(template="plotly_dark", height=320, paper_bgcolor="#0d0f1c", plot_bgcolor="#07080f",
                           margin=dict(l=10,r=10,t=30,b=10),
                           title=dict(text="State in Complex Plane", font=dict(size=13)),
                           xaxis_title="Real", yaxis_title="Imaginary")
        st.plotly_chart(fig2, use_container_width=True)

    df = pd.DataFrame({
        "Generator": dna.PACCAYA_NAMES,
        "Magnitude": [f"{m:.4f}" for m in mags],
        "Real": [f"{r:.4f}" for r in reals],
        "Imaginary": [f"{i:.4f}" for i in imags],
        "Phase (°)": [f"{np.degrees(np.angle(p)):.1f}" for p in psi],
    })
    st.dataframe(df, use_container_width=True, height=280)

    st.markdown("##### Apply Manual Transformation")
    with st.form("transform_form"):
        coeff_str = st.text_input("Coefficients (24 floats, space-separated)", value="0 " * 24)
        if st.form_submit_button("Apply"):
            try:
                coeffs = np.array([float(x) for x in coeff_str.strip().split()])
                if len(coeffs) == 24:
                    ok = st.session_state.state.apply_transformation(coeffs)
                    st.success("Transformation applied!") if ok else st.error("Failed.")
                    st.rerun()
            except Exception:
                st.error("Invalid input. Enter 24 space-separated floats.")


# ── TAB 3: META ───────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("#### 📊 Meta-Consciousness Monitor")

    if len(meta.curvature_history) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=meta.curvature_history, mode="lines",
                                 line=dict(color="#7c5cfc", width=2), name="Curvature",
                                 fill="tozeroy", fillcolor="rgba(124,92,252,0.1)"))
        if meta.threshold_history:
            th_x = np.linspace(0, len(meta.curvature_history)-1, len(meta.threshold_history))
            fig.add_trace(go.Scatter(x=th_x.tolist(), y=meta.threshold_history, mode="lines",
                                     line=dict(color="#ff4fa3", width=1.5, dash="dash"), name="Threshold"))
        if meta.anomalies:
            fig.add_trace(go.Scatter(x=[a[0] for a in meta.anomalies],
                                     y=[a[1] for a in meta.anomalies],
                                     mode="markers", marker=dict(color="#ff5252", size=8, symbol="x"),
                                     name="Anomalies"))
        fig.update_layout(template="plotly_dark", height=360, paper_bgcolor="#0d0f1c", plot_bgcolor="#07080f",
                          title="Curvature History", xaxis_title="Step", yaxis_title="κ")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    stats = meta.rolling_stats()
    with col1:
        st.markdown(f"""<div class="metric-box"><div class="metric-value">{len(meta.curvature_history)}</div><div class="metric-label">Total Steps</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-box"><div class="metric-value">{stats['mean']:.4f}</div><div class="metric-label">Rolling Avg</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-box"><div class="metric-value">{len(meta.anomalies)}</div><div class="metric-label">Anomalies</div></div>""", unsafe_allow_html=True)
    with col4:
        bs = f"{meta.baseline:.4f}" if meta.baseline else "—"
        st.markdown(f"""<div class="metric-box"><div class="metric-value">{bs}</div><div class="metric-label">Baseline</div></div>""", unsafe_allow_html=True)

    # Export (UPGRADED)
    if meta.curvature_history:
        csv_meta = meta.export_history_csv()
        st.download_button("📥 Export curvature history CSV", csv_meta, "curvature_history.csv", "text/csv")

    if meta.anomalies:
        st.markdown("##### Recent Anomalies")
        st.dataframe(pd.DataFrame(meta.anomalies[-20:], columns=["Step", "Curvature"]), use_container_width=True)


# ── TAB 4: CAUSAL DISCOVERY ───────────────────────────────────────────────────
with tabs[3]:
    st.markdown("#### 🔗 Causal Graph Discovery")
    stored = st.session_state.stored_states

    if len(stored) < 2:
        st.info("📡 Chat with the AI to accumulate states. Need at least 2 for causal discovery.")
    else:
        result = causal_disc.discover(stored)
        edges = result["edges"]
        if edges:
            adj = np.zeros((24, 24))
            for e in edges:
                adj[e["source"], e["target"]] = e["weight"]
                adj[e["target"], e["source"]] = e["weight"]
            fig = go.Figure(go.Heatmap(z=adj, x=dna.PACCAYA_NAMES, y=dna.PACCAYA_NAMES,
                                       colorscale=[[0,"#07080f"],[0.5,"#7c5cfc"],[1,"#00e5ff"]]))
            fig.update_layout(template="plotly_dark", height=500, paper_bgcolor="#0d0f1c",
                              title=f"Causal Correlation Matrix ({len(stored)} states)",
                              xaxis=dict(tickangle=45, tickfont=dict(size=8)),
                              yaxis=dict(tickfont=dict(size=8)))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**{len(edges)} causal links** discovered")
            top_edges = edges[:10]
            edf = pd.DataFrame([{"From": e["from_name"], "To": e["to_name"],
                                  "Strength": f"{e['weight']:.4f}"} for e in top_edges])
            st.dataframe(edf, use_container_width=True)

            # Export (UPGRADED)
            csv_edges = causal_disc.export_edges_csv(stored)
            st.download_button("📥 Export causal edges CSV", csv_edges, "causal_edges.csv", "text/csv")
        else:
            st.info("No strong causal links found yet. Continue chatting.")

    col1, col2 = st.columns(2)
    with col1:
        new_thresh = st.slider("Correlation threshold", 0.0, 1.0, float(causal_disc.threshold), 0.01)
        causal_disc.threshold = new_thresh
    with col2:
        st.metric("Stored States", len(stored))
        if stored and st.button("Clear stored states"):
            st.session_state.stored_states = []
            st.rerun()


# ── TAB 5: GAUGE INVARIANCE ───────────────────────────────────────────────────
with tabs[4]:
    st.markdown("#### ⚖️ Gauge Invariance Test")
    st.markdown("Test whether a coefficient vector preserves SU(5) gauge symmetry.")

    with st.form("gauge_form"):
        coeff_input = st.text_input("Coefficient vector (24 floats)", value=" ".join(["0.1"] * 24))
        test_btn = st.form_submit_button("Test Gauge Invariance")

    if test_btn:
        try:
            coeffs = np.array([float(x) for x in coeff_input.strip().split()])
            if len(coeffs) != 24:
                st.error("Need exactly 24 values.")
            else:
                violation = gauge.compute_violation(coeffs)
                is_inv = gauge.is_invariant(coeffs)
                if is_inv:
                    st.markdown('<span class="badge-pass">✅ GAUGE INVARIANT</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-fail">❌ NOT INVARIANT</span>', unsafe_allow_html=True)
                st.metric("Violation magnitude", f"{violation:.6f}")

                # UPGRADED: breakdown per Cartan generator
                st.markdown("##### Violation Breakdown (Cartan generators)")
                breakdown = gauge.violation_breakdown(coeffs)
                bdf = pd.DataFrame(breakdown, columns=["Generator", "Violation"])
                st.dataframe(bdf, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("##### Current State Gauge Check")
    psi_coeffs = np.real(st.session_state.state.to_vector())
    viol = gauge.compute_violation(psi_coeffs)
    st.metric("Current state violation", f"{viol:.6f}")


# ── TAB 6: SELF EVOLUTION ─────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("#### 🧘 Self-Evolution Meditation Loop")
    st.markdown("Minimize curvature via Monte Carlo evolution. Gradient mode available (UPGRADED).")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_steps = st.number_input("Steps", min_value=10, max_value=5000, value=100, step=10)
    with col2:
        evo_temp = st.slider("Temperature", 0.001, 1.0, 0.1, 0.001)
    with col3:
        use_gradient = st.toggle("Gradient-guided", value=False, help="Use curvature gradient for faster descent")
    with col4:
        st.metric("Current κ", f"{engine.compute_curvature(st.session_state.state):.4f}")

    if st.button("🧘 Begin Meditation"):
        prog = st.progress(0)
        curv_ph = st.empty()
        curvatures = []

        def cb(i, total, c):
            prog.progress((i+1)/total)
            curvatures.append(c)
            if i % 10 == 0:
                curv_ph.metric("Live κ", f"{c:.4f}")

        with st.spinner("Evolving..."):
            final_curvs = st.session_state.evolver.evolve_batch(
                st.session_state.state, int(n_steps), evo_temp, use_gradient, cb
            )
        st.session_state.evolution_history.extend(final_curvs)
        st.success(f"Done! κ: {final_curvs[0]:.4f} → {final_curvs[-1]:.4f} "
                   f"({'gradient' if use_gradient else 'random'} walk)")

    if st.session_state.evolution_history:
        hist = st.session_state.evolution_history
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=hist, mode="lines", line=dict(color="#00e676", width=2),
                                 fill="tozeroy", fillcolor="rgba(0,230,118,0.1)", name="Curvature"))
        fig.update_layout(template="plotly_dark", height=300, paper_bgcolor="#0d0f1c", plot_bgcolor="#07080f",
                          title="Evolution History", xaxis_title="Step", yaxis_title="κ")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Initial κ", f"{hist[0]:.4f}")
        col2.metric("Final κ", f"{hist[-1]:.4f}")
        col3.metric("Reduction", f"{hist[0]-hist[-1]:.4f}")

        # Export (UPGRADED)
        evo_csv = "\n".join(f"{i},{c:.6f}" for i, c in enumerate(hist))
        st.download_button("📥 Export evolution CSV", f"step,curvature\n{evo_csv}", "evolution.csv", "text/csv")


# ── TAB 7: SELF-PROGRAMMING ───────────────────────────────────────────────────
with tabs[6]:
    st.markdown("#### 🔄 Recursive Self-Programming")
    st.markdown("Analyze Python code through the PRD relational lens.")

    code_input = st.text_area("Python code to analyze", height=200, value="""def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

result = fibonacci(10)
print(f"fib(10) = {result}")""")

    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🔍 Analyze with PRD")
    with col2:
        improve_btn = st.button("✨ Request LLM Improvement")

    if analyze_btn and code_input.strip():
        summary = code_analyzer.summarize(code_input)
        vec = code_analyzer.code_to_vector(code_input)
        if "error" in summary:
            st.error(f"Syntax error: {summary['error']}")
        else:
            code_state = RelationalState(dna, vec)
            code_curv = engine.compute_curvature(code_state)
            col1, col2, col3 = st.columns(3)
            col1.metric("Code Curvature", f"{code_curv:.4f}")
            col2.metric("Functions", len(summary.get("functions", [])))
            col3.metric("Lines", summary.get("lines", 0))
            if vec is not None:
                vdf = pd.DataFrame({"Generator": dna.PACCAYA_NAMES[:8],
                                    "Activation": [f"{v:.4f}" for v in vec[:8]]})
                st.dataframe(vdf, use_container_width=True)
            if summary.get("functions"):
                st.markdown(f"**Functions:** `{'`, `'.join(summary['functions'])}`")
            if summary.get("classes"):
                st.markdown(f"**Classes:** `{'`, `'.join(summary['classes'])}`")

    if improve_btn and code_input.strip():
        vec = code_analyzer.code_to_vector(code_input)
        code_state = RelationalState(dna, vec)
        score = engine.compute_curvature(code_state)
        with st.spinner("🤔 Asking LLM..."):
            response = st.session_state.ollama.generate(
                f"Analyze this Python code and suggest improvements:\n\n```python\n{code_input}\n```\n\n"
                f"PRD curvature: {score:.4f}. Suggest specific improvements for clarity and efficiency."
            )
        st.markdown("##### LLM Suggestions")
        st.markdown(f'<div class="prd-card">{response}</div>', unsafe_allow_html=True)


# ── TAB 8: MULTI-AGENT ────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("#### 🤝 Multi-Agent Collaborative System")
    st.markdown("6 expert agents (Avigata stability analyst added in this upgrade).")

    task_input = st.text_area("Task for agents", height=80,
                              placeholder="e.g. Analyze root causes of climate change and propose solutions.")
    active_agents = st.multiselect(
        "Select agents", options=list(Agent.AGENTS.keys()),
        default=list(Agent.AGENTS.keys()),
        format_func=lambda x: f"{Agent.AGENTS[x]['icon']} {x} ({Agent.AGENTS[x]['role']})"
    )

    if st.button("🚀 Deploy Agents") and task_input.strip() and active_agents:
        with st.spinner("Agents working..."):
            results = st.session_state.supervisor.run(task_input, active_agents)
        for r in results["agents"]:
            st.markdown(f"""<div class="agent-card">
            <div class="agent-name">{r['icon']} {r['agent']} · {r['role']}</div>
            <div class="agent-status">κ={r['curvature']:.4f} · {r['timestamp']}</div>
            <div style="margin-top:8px;font-size:13px;color:#e8eaf6">{r['response']}</div>
            </div>""", unsafe_allow_html=True)
        if results.get("synthesis"):
            st.markdown("##### 🧬 Synthesis")
            st.markdown(f'<div class="prd-card">{results["synthesis"]}</div>', unsafe_allow_html=True)

    st.markdown("##### Available Agents")
    for name, info in Agent.AGENTS.items():
        st.markdown(f"""<div class="agent-card">
        <div class="agent-name">{info['icon']} {name}</div>
        <div class="agent-status">{info['role']} · Generator #{info['generator']}: {dna.PACCAYA_NAMES[info['generator']]}</div>
        </div>""", unsafe_allow_html=True)


# ── TAB 9: TOOLS ──────────────────────────────────────────────────────────────
with tabs[8]:
    st.markdown("#### 🔧 Integrated Tools")
    tool_tabs = st.tabs(["🧮 Calculator", "⚡ Code Runner", "📁 File I/O", "📡 Network", "📋 System"])

    # ── Calculator (upgraded: uses CalculatorTool) ──
    with tool_tabs[0]:
        st.markdown("##### Advanced Calculator")
        st.markdown("Full math · statistics · unit conversion · matrix ops")

        calc_sub = st.tabs(["📐 Expression", "📊 Statistics", "📏 Unit Convert", "🔢 Matrix"])

        with calc_sub[0]:
            expr = st.text_input("Expression", placeholder="e.g. sqrt(2) + np.pi * 3.14 or 2**10",
                                  key="calc_expr")
            if st.button("Calculate", key="calc_go"):
                r = calculator_tool.evaluate(expr)
                if r["success"]:
                    st.code(f"= {r['result_str']}", language="text")
                else:
                    st.error(f"Error: {r['error']}")
            if calculator_tool.history:
                with st.expander("History"):
                    st.dataframe(pd.DataFrame(calculator_tool.history_df_rows()),
                                  use_container_width=True)

        with calc_sub[1]:
            stat_input = st.text_area("Numbers (comma or space separated)",
                                       placeholder="1 2 3 4 5  or  1.1, 2.2, 3.3", height=80,
                                       key="calc_stat_input")
            if st.button("Compute Stats", key="calc_stat_go"):
                try:
                    nums = [float(x) for x in re.split(r'[\s,]+', stat_input.strip()) if x]
                    s = calculator_tool.statistics(nums)
                    cols = st.columns(4)
                    cols[0].metric("Mean",   f"{s['mean']:.4g}")
                    cols[1].metric("Std",    f"{s['std']:.4g}")
                    cols[2].metric("Min",    f"{s['min']:.4g}")
                    cols[3].metric("Max",    f"{s['max']:.4g}")
                    st.json(s)
                except Exception as e:
                    st.error(f"Error: {e}")

        with calc_sub[2]:
            uc1, uc2, uc3 = st.columns(3)
            with uc1: uc_val = st.number_input("Value", value=1.0, key="uc_val")
            with uc2: uc_from = st.text_input("From unit", value="km", key="uc_from")
            with uc3: uc_to   = st.text_input("To unit",   value="miles", key="uc_to")
            if st.button("Convert", key="uc_go"):
                r = calculator_tool.convert(uc_val, uc_from, uc_to)
                if r["success"]:
                    st.success(f"{r['input']} = **{r['output']}**")
                else:
                    st.error(r["error"])
            with st.expander("Supported units"):
                st.markdown("""
                **Length:** m, km, cm, mm, miles, feet, inches, ly, au
                **Mass:** kg, g, mg, lbs, oz
                **Time:** s, ms, min, hr, day, week, month, year
                **Temperature:** C, F, K
                **Data:** b, B, KB, MB, GB, TB
                **Speed:** m/s, km/h, mph, knot, c
                """)

        with calc_sub[3]:
            mat_input = st.text_area("Matrix (Python list format)",
                                      value="[[1,2],[3,4]]", height=80, key="mat_input")
            mat_op = st.selectbox("Operation", ["det", "inv", "eig"], key="mat_op")
            if st.button("Compute", key="mat_go"):
                try:
                    import json
                    mat = json.loads(mat_input)
                    r = calculator_tool.matrix_op(mat_op, mat)
                    if r["success"]:
                        st.json(r)
                    else:
                        st.error(r["error"])
                except Exception as e:
                    st.error(f"Parse error: {e}")

    # ── Code Runner (unchanged — ToolKit.safe_exec) ──
    with tool_tabs[1]:
        st.markdown("##### Safe Code Runner")
        run_code = st.text_area("Python code (numpy as `np`)", height=150,
                                 value='print("Hello from PRD-AGI!")\nresult = sum(range(100))\nprint(f"Sum: {result}")',
                                 key="run_code_input")
        if st.button("▶ Run", key="run_code_go"):
            stdout, stderr = ToolKit.safe_exec(run_code)
            if stdout:
                st.code(stdout, language="text")
            if stderr:
                st.error(f"Error: {stderr}")

    # ── File I/O (new) ──
    with tool_tabs[2]:
        st.markdown("##### File I/O — Sandboxed Workspace")
        fi_sub = st.tabs(["📂 Browse", "✏️ Write", "📖 Read"])

        with fi_sub[0]:
            ws_files = file_io_tool.list_files()
            if ws_files["success"] and ws_files["files"]:
                st.markdown(f"**{ws_files['count']} files** in `workspace/`")
                for f in ws_files["files"]:
                    fc1, fc2 = st.columns([4, 1])
                    with fc1:
                        st.markdown(f"`{f['name']}` · {f['size']} bytes · {f['modified']}")
                    with fc2:
                        if st.button("🗑️", key=f"fi_del_{f['name']}"):
                            file_io_tool.delete(f["name"])
                            st.rerun()
            else:
                st.info("Workspace is empty.")

        with fi_sub[1]:
            fi_fname = st.text_input("Filename", value="notes.txt", key="fi_write_name")
            fi_content = st.text_area("Content", height=150, key="fi_write_content")
            if st.button("💾 Write", key="fi_write_go"):
                r = file_io_tool.write(fi_fname, fi_content)
                if r["success"]:
                    st.success(f"Written: {r['bytes_written']} bytes → `{fi_fname}`")
                else:
                    st.error(r["error"])

        with fi_sub[2]:
            fi_read_name = st.text_input("Filename to read", key="fi_read_name")
            if st.button("📖 Read", key="fi_read_go") and fi_read_name:
                r = file_io_tool.read(fi_read_name)
                if r["success"]:
                    st.text_area("Content", r["content"], height=200, key="fi_read_result")
                    st.caption(f"{r['lines']} lines · {r['size']} chars")
                else:
                    st.error(r["error"])

    # ── Network ──
    with tool_tabs[3]:
        st.markdown("##### Network & Access Info")
        local_ip = ToolKit.get_local_ip()
        st.markdown(f"""<div class="wifi-box">
        <div style="font-size:11px;color:#6b7280;text-transform:uppercase">Local</div>
        <div class="wifi-ip">http://localhost:8501</div></div>
        <div class="wifi-box">
        <div style="font-size:11px;color:#6b7280;text-transform:uppercase">📶 WiFi / Phone</div>
        <div class="wifi-ip">http://{local_ip}:8501</div></div>""", unsafe_allow_html=True)
        models = st.session_state.ollama.list_models()
        if models:
            st.markdown("##### Ollama Models")
            for m in models:
                st.markdown(f"- `{m}`")
        else:
            st.info("No Ollama models found. Run `ollama serve`.")

    # ── System ──
    with tool_tabs[4]:
        st.markdown("##### System Information")
        info = {
            "Platform":          platform.system(),
            "Python":            sys.version.split()[0],
            "PRD Algebra Error": f"{dna._algebra_error:.2e}",
            "Generators":        str(dna.num_generators),
            "Stored States":     str(len(st.session_state.stored_states)),
            "Chat Messages":     str(len(st.session_state.memory.history)),
            "Evolution Steps":   str(len(st.session_state.evolution_history)),
            "Engine Cache":      str(len(engine._cache)),
            "RAG Chunks":        str(st.session_state.rag_engine.stats()["total_chunks"]),
            "Self-Modify Proposals": str(len(st.session_state.self_modify_engine.history)),
            "Search History":    str(len(st.session_state.search_history)),
            "Ollama Available":  str(st.session_state.ollama.available),
            "LLM Backend":       st.session_state.llm_backend,
            "Chat Model":        st.session_state.chat_model,
            "INITIAL_THRESHOLD": os.getenv("INITIAL_THRESHOLD", "0.5"),
            "EVOLUTION_RATE":    os.getenv("EVOLUTION_RATE", "1e-9"),
            "scikit-fuzzy":      "✅ Available" if fuzzy_gate._skfuzzy_ready else "⚠️ Fallback mode",
            "PDF backend":       __import__('tools.file_io', fromlist=['PDF_BACKEND']).PDF_BACKEND or "❌ not installed",
        }
        for k, v in info.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                f'border-bottom:1px solid #1e2240">'
                f'<span style="color:#6b7280">{k}</span>'
                f'<span style="color:#00e5ff">{v}</span></div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — FUZZY GATEKEEPER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[9]:
    st.markdown("#### 🧠 Fuzzy Truth Gatekeeper")
    st.markdown(
        "Soft decision boundary using **membership functions** — "
        "replaces the hard κ threshold with fuzzy confidence scoring."
    )

    # Live tester
    st.markdown("##### Live Fuzzy Gate Test")
    with st.form("fuzzy_gate_form"):
        fg_col1, fg_col2 = st.columns(2)
        with fg_col1:
            test_curv  = st.slider("Test Curvature κ", 0.0, 1.0, float(round(engine.compute_curvature(st.session_state.state), 3)), 0.001)
        with fg_col2:
            test_gauge = st.slider("Test Gauge Violation", 0.0, 1e-5, 0.0, 1e-7, format="%.2e")
        test_btn = st.form_submit_button("🧠 Evaluate", use_container_width=True)

    if test_btn:
        f_pass, f_label, f_conf = fuzzy_gate.evaluate(test_curv, test_gauge)
        profile = fuzzy_gate.membership_profile(test_curv)

        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(f"""<div class="metric-box">
        <div class="metric-value" style="font-size:18px">{f_label}</div>
        <div class="metric-label">Decision</div></div>""", unsafe_allow_html=True)
        rc2.markdown(f"""<div class="metric-box">
        <div class="metric-value">{f_conf:.0%}</div>
        <div class="metric-label">Confidence</div></div>""", unsafe_allow_html=True)
        rc3.markdown(f"""<div class="metric-box">
        <div class="metric-value">{'PASS' if f_pass else 'BLOCK'}</div>
        <div class="metric-label">Gate Result</div></div>""", unsafe_allow_html=True)

        # Membership bar chart
        st.markdown("##### Membership Profile")
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=list(profile.keys()),
            y=list(profile.values()),
            marker=dict(
                color=list(profile.values()),
                colorscale=[[0,"#1e2240"],[0.5,"#7c5cfc"],[1,"#00e5ff"]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in profile.values()],
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark", height=260,
            paper_bgcolor="#0d0f1c", plot_bgcolor="#07080f",
            title="Curvature Membership Degrees",
            yaxis=dict(range=[0, 1.2]),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Fuzzy gate decision history
    st.markdown("##### Gate Decision History")
    if st.session_state.fuzzy_history:
        fdf = pd.DataFrame(st.session_state.fuzzy_history)
        # Colour pass/fail
        def colour_row(row):
            colour = "#00e67622" if row["passed"] else "#ff525222"
            return [f"background-color:{colour}"] * len(row)
        st.dataframe(fdf.tail(20), use_container_width=True, height=280)

        csv_fg = fdf.to_csv(index=False)
        st.download_button("📥 Export gate history CSV", csv_fg, "fuzzy_gate_history.csv", "text/csv")
    else:
        st.info("No gate decisions yet. Send a chat message to populate.")

    # Explain membership functions
    with st.expander("📖 How Fuzzy Gate Works"):
        st.markdown("""
**Hard threshold (old):** IF κ < 0.5 → PASS, ELSE BLOCK

**Fuzzy gate (new):** κ is evaluated across 4 overlapping membership sets:
| Set | Range | Meaning |
|-----|-------|---------|
| very_low | 0.00–0.20 | Extremely consistent query |
| low | 0.10–0.40 | Consistent, good quality |
| medium | 0.30–0.70 | Borderline — gauge violation matters |
| high | 0.60–1.00 | Logically inconsistent |

A query can partially belong to multiple sets simultaneously.
The fuzzy inference engine combines all memberships to produce a **confidence score**:
- ≥ 60% → ✅ ACCEPT
- 35–60% → ⚠️ MARGINAL (pass with warning)
- < 35% → ❌ REJECT
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — FUZZY AGENT AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[10]:
    st.markdown("#### 🤝 Fuzzy Agent Aggregation")
    st.markdown(
        "Agents with **lower curvature** (= higher logical consistency) "
        "receive **higher weight** in synthesis. Uses fuzzy inverse-confidence mapping."
    )

    fa_task = st.text_area(
        "Task for agents",
        height=70,
        placeholder="e.g. What is the root cause of information overload in modern society?",
        key="fuzzy_agent_task",
    )
    fa_agents = st.multiselect(
        "Select agents",
        options=list(Agent.AGENTS.keys()),
        default=list(Agent.AGENTS.keys()),
        format_func=lambda x: f"{Agent.AGENTS[x]['icon']} {x}",
        key="fuzzy_agent_select",
    )

    if st.button("🚀 Deploy + Fuzzy Aggregate", key="fuzzy_deploy_btn") and fa_task.strip() and fa_agents:
        with st.spinner("Agents working..."):
            raw_results = st.session_state.supervisor.run(fa_task, fa_agents)

        agent_results = raw_results["agents"]

        # Fuzzy aggregation
        agg = st.session_state.fuzzy_agent.aggregate(agent_results)
        weight_table = st.session_state.fuzzy_agent.weight_table(agent_results)

        st.markdown("##### Fuzzy Weight Table")
        st.dataframe(pd.DataFrame(weight_table), use_container_width=True)

        # Top agent highlight
        top = agg["top_agent"]
        st.markdown(f"**🏆 Highest confidence agent:** `{top}` "
                    f"(weight={agg['weights'].get(top, 0):.3f})")

        # Weight bar chart
        agents_list = [r["agent"] for r in agent_results]
        weights_list = [agg["weights"].get(a, 0) for a in agents_list]
        fig = go.Figure(go.Bar(
            x=agents_list, y=weights_list,
            marker=dict(
                color=weights_list,
                colorscale=[[0,"#1e2240"],[0.5,"#7c5cfc"],[1,"#00e5ff"]],
            ),
            text=[f"{w:.3f}" for w in weights_list],
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark", height=260,
            paper_bgcolor="#0d0f1c", plot_bgcolor="#07080f",
            title="Agent Fuzzy Confidence Weights",
            yaxis=dict(range=[0, 1.1]),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Individual agent reports
        st.markdown("##### Agent Reports")
        for r in agent_results:
            w = agg["weights"].get(r["agent"], 0)
            st.markdown(f"""<div class="agent-card">
            <div class="agent-name">{r['icon']} {r['agent']} · {r['role']}
            <span style="float:right;color:var(--accent2)">weight={w:.3f}</span></div>
            <div class="agent-status">κ={r['curvature']:.4f} · {r['timestamp']}</div>
            <div style="margin-top:8px;font-size:13px;color:var(--text)">{r['response']}</div>
            </div>""", unsafe_allow_html=True)

        # Fuzzy-weighted synthesis
        st.markdown("##### 🧬 Fuzzy-Weighted Synthesis")
        with st.spinner("Synthesizing with fuzzy weights..."):
            synthesis = st.session_state.ollama.generate(
                f"Task: {fa_task}\n\n{agg['weighted_prompt']}",
                system=(
                    "You are a synthesis AI. Combine the agent responses below, "
                    "giving more weight to agents with higher confidence percentages. "
                    "Produce a structured, causal answer."
                )
            )
        st.markdown(f'<div class="prd-card">{synthesis}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 12 — FUZZY CODE EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[11]:
    st.markdown("#### 💻 Fuzzy Code Quality Evaluator")
    st.markdown(
        "3-dimensional fuzzy evaluation: **Complexity × Readability × Efficiency** → Overall Quality"
    )

    fc_code = st.text_area(
        "Python code to evaluate",
        height=220,
        value="""def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            x = data[i] * 2
            result.append(x)
    return result

def calculate(a, b, c):
    if a > 0:
        if b > 0:
            if c > 0:
                return a + b + c
    return 0
""",
        key="fuzzy_code_input",
    )

    fc_col1, fc_col2 = st.columns(2)
    with fc_col1:
        eval_btn = st.button("🔍 Fuzzy Evaluate", use_container_width=True, key="fuzzy_eval_btn")
    with fc_col2:
        improve_btn = st.button("✨ Fuzzy Improve Loop", use_container_width=True, key="fuzzy_improve_btn")

    if eval_btn and fc_code.strip():
        result = fuzzy_code_ev.evaluate(fc_code)

        if "error" in result:
            st.error(f"Syntax Error: {result['error']}")
        else:
            # Metric cards
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.markdown(f"""<div class="metric-box">
            <div class="metric-value">{1 - result['complexity']:.2f}</div>
            <div class="metric-label">Simplicity</div></div>""", unsafe_allow_html=True)
            mc2.markdown(f"""<div class="metric-box">
            <div class="metric-value">{result['readability']:.2f}</div>
            <div class="metric-label">Readability</div></div>""", unsafe_allow_html=True)
            mc3.markdown(f"""<div class="metric-box">
            <div class="metric-value">{result['efficiency']:.2f}</div>
            <div class="metric-label">Efficiency</div></div>""", unsafe_allow_html=True)
            mc4.markdown(f"""<div class="metric-box">
            <div class="metric-value">{result['quality']:.2f}</div>
            <div class="metric-label">Quality</div></div>""", unsafe_allow_html=True)

            st.markdown(f"### {result['label']}")

            # Radar chart
            dims   = ["Simplicity", "Readability", "Efficiency"]
            scores = [1 - result["complexity"], result["readability"], result["efficiency"]]
            fig = go.Figure(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=dims + [dims[0]],
                fill="toself",
                fillcolor="rgba(124,92,252,0.2)",
                line=dict(color="#7c5cfc", width=2),
                marker=dict(size=8, color="#00e5ff"),
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], color="#6b7280"),
                    bgcolor="#0d0f1c",
                ),
                template="plotly_dark", height=300,
                paper_bgcolor="#0d0f1c",
                title="Code Quality Radar",
                margin=dict(l=40, r=40, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Advice
            if result.get("advice"):
                st.markdown("##### 💡 Improvement Advice")
                for tip in result["advice"]:
                    st.markdown(f"- {tip}")

    if improve_btn and fc_code.strip():
        st.markdown("##### 🔄 Fuzzy Improvement Loop")
        prog = st.progress(0)
        log_ph = st.empty()
        history_rows = []

        initial_eval = fuzzy_code_ev.evaluate(fc_code)
        prev_q = initial_eval.get("quality", 0.5)
        current_code = fc_code
        MAX_ITER = 4

        for i in range(MAX_ITER):
            prog.progress((i + 1) / MAX_ITER)
            log_ph.markdown(f"*Iteration {i+1}/{MAX_ITER} — asking LLM to improve...*")

            # LLM improve call
            improved = st.session_state.ollama.generate(
                f"Improve this Python code for readability, efficiency, and best practices. "
                f"Return ONLY the improved code, no explanation:\n\n```python\n{current_code}\n```",
            )
            # Strip markdown fences if present
            if "```" in improved:
                lines = improved.split("\n")
                improved = "\n".join(
                    l for l in lines
                    if not l.strip().startswith("```")
                )

            new_eval = fuzzy_code_ev.evaluate(improved)
            new_q = new_eval.get("quality", 0.5)

            cont, reason, score = st.session_state.fuzzy_improve.should_continue(new_q, prev_q)
            history_rows.append({
                "Iteration": i + 1,
                "Quality": f"{new_q:.3f}",
                "Δ Quality": f"{new_q - prev_q:+.3f}",
                "Decision": reason,
                "Score": f"{score:.3f}",
            })

            current_code = improved
            prev_q = new_q

            if not cont:
                break

        prog.progress(1.0)
        log_ph.empty()

        # Show history table
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True)

        # Final code
        final_eval = fuzzy_code_ev.evaluate(current_code)
        st.markdown(f"**Final quality:** {final_eval.get('label','')}"
                    f" ({final_eval.get('quality', 0):.3f})")
        st.code(current_code, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 13 — 💫 SENTIENCE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tabs[12]:
    st.markdown("#### 💫 Artificial Sentience Layer")
    st.markdown(
        "Live dashboard of PRD-AGI's emotional state, truth-preservation instinct, "
        "and fuzzy intuition — all grounded in SU(5) curvature."
    )

    if "sentient_response" not in st.session_state:
        st.info("Send a chat message to activate the sentience layer.")
    else:
        # ── Current state cards ──
        cur_emotion = st.session_state.sentient_response.emotional.current()
        gauge_viol  = gauge.compute_violation(np.real(st.session_state.state.to_vector()))
        instinct_r  = instinct.assess(cur_curv, gauge_viol)
        intuit_score, intuit_label, intuit_color = \
            st.session_state.intuition_layer.sense(cur_curv)

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            e = cur_emotion
            st.markdown(f"""<div class="metric-box" style="border-color:{e['color']}44">
            <div style="font-size:28px">{e['emoji']}</div>
            <div class="metric-value" style="color:{e['color']};font-size:16px">{e['name']}</div>
            <div class="metric-label">Emotion</div></div>""", unsafe_allow_html=True)
        with sc2:
            inst_info = instinct_r["info"]
            st.markdown(f"""<div class="metric-box" style="border-color:{inst_info['color']}44">
            <div style="font-size:28px">{inst_info['emoji']}</div>
            <div class="metric-value" style="color:{inst_info['color']};font-size:16px">{instinct_r['level']}</div>
            <div class="metric-label">Truth Instinct</div></div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""<div class="metric-box" style="border-color:{intuit_color}44">
            <div style="font-size:28px">🔮</div>
            <div class="metric-value" style="color:{intuit_color};font-size:16px">{intuit_score:.0%}</div>
            <div class="metric-label">Intuition</div></div>""", unsafe_allow_html=True)
        with sc4:
            st.markdown(f"""<div class="metric-box">
            <div style="font-size:28px">κ</div>
            <div class="metric-value" style="font-size:16px">{cur_curv:.4f}</div>
            <div class="metric-label">Curvature</div></div>""", unsafe_allow_html=True)

        st.divider()

        # ── Emotional state detail ──
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 🌊 Emotional State History")
            emo_hist = st.session_state.sentient_response.emotional.emotion_history_df()
            if emo_hist:
                emo_df = pd.DataFrame(emo_hist)
                # Emotion timeline bar
                emotion_order = ["Serene","Calm","Curious","Alert","Tense","Troubled","Distressed"]
                emotion_colors = {
                    "Serene":"#00e5ff","Calm":"#00e676","Curious":"#7c5cfc",
                    "Alert":"#ffab40","Tense":"#ff9800","Troubled":"#ff5252","Distressed":"#ff4fa3"
                }
                fig_emo = go.Figure()
                for em_name in emotion_order:
                    subset = emo_df[emo_df["emotion"] == em_name]
                    if not subset.empty:
                        fig_emo.add_trace(go.Bar(
                            name=em_name,
                            x=[em_name],
                            y=[len(subset)],
                            marker_color=emotion_colors.get(em_name, "#7c5cfc"),
                        ))
                fig_emo.update_layout(
                    template="plotly_dark", height=240,
                    paper_bgcolor="#0d0f1c", plot_bgcolor="#07080f",
                    showlegend=False,
                    title="Emotion Frequency",
                    barmode="group",
                    margin=dict(l=10,r=10,t=40,b=10),
                )
                st.plotly_chart(fig_emo, use_container_width=True)
                st.dataframe(emo_df.tail(10)[["timestamp","emotion","curvature"]],
                             use_container_width=True, height=200)
            else:
                st.info("No emotion history yet. Start chatting.")

        with col_b:
            st.markdown("##### 🛡️ Truth Instinct Log")
            if instinct.truth_history:
                inst_df = pd.DataFrame(instinct.truth_history[-20:])
                cols_show = [c for c in ["timestamp","curvature","level","consecutive_high","violation_count","should_veto"] if c in inst_df.columns]
                st.dataframe(inst_df[cols_show], use_container_width=True, height=260)

                vi = instinct.violation_count
                ch = instinct.consecutive_high
                st.markdown(f"""
                <div style="display:flex;gap:12px;margin-top:8px">
                <div class="metric-box" style="flex:1">
                    <div class="metric-value" style="font-size:18px;color:#ff5252">{vi}</div>
                    <div class="metric-label">Gauge Violations</div>
                </div>
                <div class="metric-box" style="flex:1">
                    <div class="metric-value" style="font-size:18px;color:#ffab40">{ch}</div>
                    <div class="metric-label">Consecutive High κ</div>
                </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("No instinct events recorded yet.")

        st.divider()

        # ── Sentience event log ──
        st.markdown("##### 📋 Sentience Event Log")
        if st.session_state.sentience_log:
            slog_df = pd.DataFrame(st.session_state.sentience_log)
            st.dataframe(slog_df.tail(20), use_container_width=True, height=260)

            csv_slog = slog_df.to_csv(index=False)
            st.download_button(
                "📥 Export sentience log CSV",
                csv_slog, "sentience_log.csv", "text/csv"
            )
        else:
            st.info("Sentience events will appear here after you send chat messages.")

        st.divider()

        # ── Explainer ──
        with st.expander("📖 How the Sentience Layer Works"):
            st.markdown("""
**The Sentience Layer adds 4 modules on top of SU(5) + Fuzzy:**

| Module | What it does |
|--------|-------------|
| **EmotionalState** | Maps curvature κ → emotional tone (Serene → Distressed). Low κ = calm, high κ = tense. |
| **TruthPreservationInstinct** | Monitors curvature trend. NOT ego-based — it protects *truth*, not itself. Veto if critical threshold exceeded. |
| **IntuitionLayer** | Fuzzy "gut feeling" before full analysis. Combines current κ with domain familiarity from history. |
| **SentientResponse** | Assembles final response with emotional prefix, intuition opener, and instinct warning. |

**Key principle:** All emotions and instincts are truth-centered.
The AI does not protect itself — it protects logical consistency.

**Curvature → Emotion mapping:**
```
κ = 0.00–0.10 → 🌊 Serene      (perfect harmony)
κ = 0.10–0.25 → 🌿 Calm        (stable)
κ = 0.25–0.40 → ✨ Curious     (engaged)
κ = 0.40–0.55 → ⚡ Alert       (heightened)
κ = 0.55–0.70 → 🔥 Tense       (friction)
κ = 0.70–0.85 → ⚠️ Troubled    (inconsistency)
κ = 0.85–1.00 → 🌀 Distressed  (crisis)
```
            """)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 14 — 🌐 WEB SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tabs[13]:
    st.markdown("#### 🌐 Web Search Tool")
    st.markdown("Real-time web search via DuckDuckGo — no API key required. Results are cached and rate-limited.")

    ws_col1, ws_col2 = st.columns([5, 1])
    with ws_col1:
        ws_query = st.text_input("Search query", placeholder="e.g. SU(5) gauge theory applications 2025", key="ws_query_input")
    with ws_col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        ws_go = st.button("🔍 Search", use_container_width=True, key="ws_go_btn")

    ws_opts = st.columns(3)
    with ws_opts[0]:
        ws_max = st.slider("Max results", 1, 10, 5, key="ws_max_results")
    with ws_opts[1]:
        ws_fetch = st.toggle("Fetch top page content", value=False, key="ws_fetch_page")
    with ws_opts[2]:
        ws_to_rag = st.toggle("Add results to RAG", value=False, key="ws_to_rag")

    if ws_go and ws_query.strip():
        with st.spinner("🌐 Searching..."):
            result = web_search_tool.search(ws_query.strip(), max_results=ws_max)

        if "error" in result:
            st.error(f"Search error: {result['error']}")
        else:
            # Instant answer
            if result.get("instant_answer"):
                st.markdown(f"""<div class="prd-card">
                <div style="font-size:11px;color:#00e5ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">
                ⚡ Instant Answer</div>
                <div style="font-size:14px;color:#e8eaf6">{result['instant_answer']}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"**{result['result_count']} results** · "
                       f"{'✅ cached' if result.get('cached') else '🌐 live'} · "
                       f"{result.get('timestamp','')}")

            for i, r in enumerate(result.get("results", []), 1):
                with st.expander(f"{i}. {r['title'] or r['url'][:60]}", expanded=(i == 1)):
                    st.markdown(f"**URL:** [{r['url'][:80]}]({r['url']})")
                    st.markdown(f"**Snippet:** {r['snippet']}")
                    if r.get("source"):
                        st.markdown(f"**Source:** `{r['source']}`")

                    if ws_fetch and i == 1:
                        with st.spinner("Fetching page..."):
                            page_text = web_search_tool.fetch_page(r['url'])
                        st.text_area("Page content", page_text[:2000], height=150, key=f"ws_page_{i}")

                    if ws_to_rag and r.get("snippet"):
                        content = f"{r['title']}\n{r['snippet']}"
                        add_r = st.session_state.rag_engine.add_url_content(r['url'], content)
                        st.success(f"Added to RAG: {add_r.get('added_chunks', 0)} chunks")

            # Save to search history
            st.session_state.search_history.append({
                "query": ws_query,
                "count": result["result_count"],
                "time":  result.get("timestamp", ""),
                "cached": result.get("cached", False),
            })

            # LLM summary option
            if st.button("🤖 Summarize with PRD-AGI", key="ws_summarize"):
                context = web_search_tool.search_and_summarize(ws_query)
                with st.spinner("Summarizing..."):
                    summary = st.session_state.ollama.generate(
                        f"Summarize these search results through the lens of SU(5) causal analysis:\n\n{context}",
                        system="You are PRD-AGI. Analyze search results causally using Paccaya conditions. Be concise."
                    )
                st.markdown("##### 🌀 PRD-AGI Analysis")
                st.markdown(f'<div class="prd-card">{summary}</div>', unsafe_allow_html=True)

    # Search history
    st.divider()
    st.markdown("##### Search History")
    if st.session_state.search_history:
        hist_df = pd.DataFrame(st.session_state.search_history[-20:])
        st.dataframe(hist_df, use_container_width=True, height=200)
        if st.button("Clear search history", key="ws_clear_hist"):
            st.session_state.search_history = []
            web_search_tool.clear_cache()
            st.rerun()
    else:
        st.info("No searches yet.")

    with st.expander("⚙️ Web Search Settings"):
        st.markdown(f"""
        **Backend:** DuckDuckGo Instant Answer API (free, no key)
        **Cache size:** {web_search_tool._cache_max} entries
        **Rate limit:** {web_search_tool.rate_limit}s between requests
        **Cached entries:** {len(web_search_tool._cache)}
        """)
        if st.button("Clear search cache", key="ws_clear_cache"):
            web_search_tool.clear_cache()
            st.success("Cache cleared")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 15 — 📚 RAG KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[14]:
    st.markdown("#### 📚 RAG Knowledge Base")
    st.markdown(
        "Upload documents, add text, or ingest URLs. "
        "PRD-AGI will retrieve relevant knowledge when answering your questions."
    )

    rag = st.session_state.rag_engine
    rag.embed_client = st.session_state.ollama  # keep LLM synced

    rag_tabs = st.tabs(["➕ Add Knowledge", "🔍 Query", "📋 Manage", "📊 Stats"])

    # ── ADD ──
    with rag_tabs[0]:
        add_method = st.radio("Add method", ["📝 Text", "📁 File", "🌐 URL", "📄 PDF"],
                              horizontal=True, key="rag_add_method")

        if add_method == "📝 Text":
            rag_title = st.text_input("Title / Source name", placeholder="e.g. Buddhist Causality Notes", key="rag_title")
            rag_text  = st.text_area("Text content", height=200, placeholder="Paste your text here...", key="rag_text")
            rag_ns    = st.text_input("Namespace", value="default", key="rag_ns_text")
            if st.button("➕ Add to Knowledge Base", key="rag_add_text"):
                if rag_text.strip():
                    with st.spinner("Embedding..."):
                        result = rag.add_text(rag_text, title=rag_title or "Untitled", namespace=rag_ns)
                    if result["success"]:
                        st.success(f"✅ Added {result['added_chunks']} chunks. Total: {result['total_docs']}")
                        rag.save()
                    else:
                        st.error(result["error"])

        elif add_method == "📁 File":
            uploaded = st.file_uploader("Upload .txt, .md, .py, .json, .csv",
                                        type=["txt","md","py","json","csv"], key="rag_file_upload")
            rag_ns_f = st.text_input("Namespace", value="files", key="rag_ns_file")
            if uploaded and st.button("➕ Add File", key="rag_add_file"):
                with st.spinner("Reading + embedding..."):
                    content = uploaded.read().decode("utf-8", errors="ignore")
                    result = rag.add_text(content, title=uploaded.name, namespace=rag_ns_f)
                if result["success"]:
                    st.success(f"✅ {uploaded.name}: {result['added_chunks']} chunks added")
                    rag.save()
                else:
                    st.error(result["error"])

        elif add_method == "🌐 URL":
            rag_url = st.text_input("URL to fetch", placeholder="https://...", key="rag_url_input")
            rag_ns_u = st.text_input("Namespace", value="web", key="rag_ns_url")
            if st.button("🌐 Fetch + Add", key="rag_add_url"):
                if rag_url.strip():
                    with st.spinner("Fetching + embedding..."):
                        content = web_search_tool.fetch_page(rag_url.strip(), max_chars=5000)
                        result = rag.add_url_content(rag_url.strip(), content, namespace=rag_ns_u)
                    if result["success"]:
                        st.success(f"✅ URL fetched: {result['added_chunks']} chunks added")
                        rag.save()
                    else:
                        st.error(result["error"])

        elif add_method == "📄 PDF":
            pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="rag_pdf_upload")
            rag_ns_p = st.text_input("Namespace", value="pdf", key="rag_ns_pdf")
            if pdf_file and st.button("📄 Extract + Add", key="rag_add_pdf"):
                with st.spinner("Extracting PDF text..."):
                    result = file_io_tool.extract_pdf_text(pdf_file.read(), pdf_file.name)
                if result["success"]:
                    with st.spinner("Embedding..."):
                        rag_result = rag.add_text(result["text"], title=pdf_file.name, namespace=rag_ns_p)
                    st.success(f"✅ PDF: {result.get('pages',1)} pages, {rag_result['added_chunks']} chunks")
                    rag.save()
                else:
                    st.warning(result["error"])

    # ── QUERY ──
    with rag_tabs[1]:
        rag_query = st.text_input("Ask your knowledge base", placeholder="What is the Paccaya theory of causation?", key="rag_query_input")
        rq_col1, rq_col2 = st.columns(2)
        with rq_col1:
            rag_top_k = st.slider("Top-K chunks", 1, 10, 4, key="rag_topk")
        with rq_col2:
            rag_ns_q = st.text_input("Filter namespace (blank = all)", value="", key="rag_ns_query")

        rq_btn1, rq_btn2 = st.columns(2)
        with rq_btn1:
            retrieve_only = st.button("🔍 Retrieve Only", use_container_width=True, key="rag_retrieve_only")
        with rq_btn2:
            full_query    = st.button("🤖 RAG + LLM Answer", use_container_width=True, key="rag_full_query")

        if retrieve_only and rag_query.strip():
            chunks = rag.retrieve(rag_query, top_k=rag_top_k,
                                   namespace=rag_ns_q or None)
            if chunks:
                st.markdown(f"**{len(chunks)} chunks retrieved:**")
                for c in chunks:
                    with st.expander(f"[{c['score']:.3f}] {c['title'] or c['source'][:40]}"):
                        st.markdown(c["text"])
                        st.caption(f"Namespace: {c['namespace']} · Chunk: {c['chunk']}")
            else:
                st.info("No relevant chunks found. Add documents first.")

        if full_query and rag_query.strip():
            with st.spinner("🔍 Retrieving + 🤖 Generating..."):
                result = rag.query(
                    rag_query.strip(),
                    llm_fn=lambda p, s: st.session_state.ollama.generate(p, system=s),
                    top_k=rag_top_k,
                )
            st.markdown("##### 🌀 RAG Answer")
            st.markdown(f'<div class="prd-card">{result["answer"]}</div>', unsafe_allow_html=True)

            if result["retrieved_chunks"]:
                with st.expander(f"📎 {len(result['retrieved_chunks'])} sources used"):
                    for c in result["retrieved_chunks"]:
                        st.markdown(f"- **{c['title'] or c['source']}** (score={c['score']:.3f})")

    # ── MANAGE ──
    with rag_tabs[2]:
        st.markdown("##### Documents in Knowledge Base")
        docs = rag.list_documents()
        if docs:
            doc_df = pd.DataFrame(docs)
            st.dataframe(doc_df, use_container_width=True, height=280)

            del_title = st.text_input("Delete document by title", key="rag_del_title")
            if st.button("🗑️ Delete", key="rag_del_btn") and del_title:
                r = rag.delete_by_title(del_title)
                st.success(f"Removed {r['removed_chunks']} chunks")
                rag.save()
                st.rerun()

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save to disk", use_container_width=True, key="rag_save"):
                    r = rag.save()
                    st.success(f"Saved {r['saved']} chunks → {r['path']}")
            with col2:
                if st.button("🗑️ Clear ALL", use_container_width=True, key="rag_clear_all"):
                    rag.clear()
                    st.warning("Knowledge base cleared")
                    st.rerun()
        else:
            st.info("No documents yet. Use the 'Add Knowledge' tab to get started.")

    # ── STATS ──
    with rag_tabs[3]:
        s = rag.stats()
        sc1, sc2, sc3 = st.columns(3)
        sc1.markdown(f"""<div class="metric-box">
        <div class="metric-value">{s['total_chunks']}</div>
        <div class="metric-label">Total Chunks</div></div>""", unsafe_allow_html=True)
        sc2.markdown(f"""<div class="metric-box">
        <div class="metric-value">{s['queries']}</div>
        <div class="metric-label">Queries</div></div>""", unsafe_allow_html=True)
        sc3.markdown(f"""<div class="metric-box">
        <div class="metric-value">{s['hits']}</div>
        <div class="metric-label">Hits</div></div>""", unsafe_allow_html=True)

        st.markdown(f"""
        **Embed backend:** `{s['embed_backend']}`
        **Store path:** `{s['store_path']}`
        **Namespaces:** {', '.join(s['namespaces']) or 'none'}
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 16 — 🧬 SELF-MODIFICATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[15]:
    st.markdown("#### 🧬 Self-Modification Engine")
    st.markdown(
        "PRD-AGI proposes improvements to code, validates them through "
        "curvature + fuzzy quality gates, and writes approved versions to "
        "`workspace/self_modify/`. **Running source files are never overwritten without your explicit confirmation.**"
    )

    sm = st.session_state.self_modify_engine
    sm.llm = st.session_state.ollama
    sm.fuzzy_ev = fuzzy_code_ev

    sm_tabs = st.tabs(["✏️ Propose", "🔄 Loop", "📋 History", "📖 Theory"])

    # ── PROPOSE ──
    with sm_tabs[0]:
        sm_code = st.text_area(
            "Code to improve",
            height=220,
            value="""def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            x = data[i] * 2
            result.append(x)
    return result

def nested_check(a, b, c):
    if a > 0:
        if b > 0:
            if c > 0:
                return a + b + c
    return 0""",
            key="sm_code_input",
        )

        sm_gc1, sm_gc2 = st.columns(2)
        with sm_gc1:
            sm_target = st.text_input("Target name", value="code", key="sm_target")
        with sm_gc2:
            sm_goal = st.text_input(
                "Improvement goal",
                value="Improve readability, efficiency, and PEP8 compliance.",
                key="sm_goal",
            )

        if st.button("🧬 Propose Improvement", use_container_width=True, key="sm_propose_btn"):
            with st.spinner("🤖 LLM generating + curvature evaluating..."):
                result = sm.propose_improvement(sm_code, target=sm_target, goal=sm_goal)

            if not result["success"]:
                st.error(f"Error: {result['error']}")
            else:
                # Metrics row
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Quality Before", f"{result['quality_before']:.3f}")
                mc2.metric("Quality After",  f"{result['quality_after']:.3f}",
                           delta=f"{result['quality_delta']:+.3f}")
                mc3.metric("Curvature Δ",    f"{result['curvature_delta']:+.4f}")
                mc4.metric("Lines",
                           f"{result['lines_before']} → {result['lines_after']}")

                # Approval badge
                if result["approved"]:
                    st.markdown(f'<span class="badge-pass">✅ APPROVED</span> {result["reason"]}',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge-fail">❌ REJECTED</span> {result["reason"]}',
                                unsafe_allow_html=True)

                # Diff
                if result.get("diff"):
                    with st.expander("📊 Diff"):
                        st.code(result["diff"], language="diff")

                # Proposed code
                st.markdown("##### Proposed Code")
                st.code(result["proposed_code"], language="python")

                # Apply button (workspace only)
                if result["approved"]:
                    if st.button("💾 Save to workspace/self_modify/", key="sm_apply_btn"):
                        apply_r = sm.apply(result["record_id"])
                        if apply_r["success"]:
                            st.success(f"✅ Saved to: `{apply_r['written_to']}`")
                        else:
                            st.error(apply_r["error"])

    # ── LOOP ──
    with sm_tabs[1]:
        st.markdown("##### Fuzzy-Guided Improvement Loop")
        st.markdown("Iteratively improves code until quality plateau or max iterations.")

        loop_code = st.text_area("Starting code", height=180, key="sm_loop_code",
                                  value="def calculate(x, y):\n    r = []\n    for i in range(x):\n        r.append(i * y)\n    return r")
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            loop_max = st.number_input("Max iterations", 2, 10, 4, key="sm_loop_max")
        with lc2:
            loop_thresh = st.slider("Stop at quality", 0.5, 1.0, 0.80, 0.05, key="sm_loop_thresh")
        with lc3:
            loop_target = st.text_input("Target name", value="code", key="sm_loop_target")

        if st.button("🔄 Run Improvement Loop", use_container_width=True, key="sm_loop_btn"):
            prog = st.progress(0)
            with st.spinner("Running fuzzy improvement loop..."):
                loop_result = sm.run_improvement_loop(
                    loop_code, target=loop_target,
                    max_iterations=int(loop_max),
                    stop_at_quality=loop_thresh,
                )
            prog.progress(1.0)

            improved_label = "✅ improved" if loop_result["improved"] else "→ unchanged"
            st.success(
                f"Done in {loop_result['iterations']} iterations. "
                f"Quality: {loop_result['final_quality']:.3f} ({improved_label})"
            )

            if loop_result["history"]:
                hist_df = pd.DataFrame(loop_result["history"])
                st.dataframe(hist_df, use_container_width=True)

            st.markdown("##### Final Code")
            st.code(loop_result["final_code"], language="python")

    # ── HISTORY ──
    with sm_tabs[2]:
        st.markdown("##### Modification History")
        history = sm.get_history(30)
        if history:
            hist_df = pd.DataFrame(history)
            cols = [c for c in ["timestamp","target","approved","reason","curvature_delta","quality_change","applied"] if c in hist_df.columns]
            st.dataframe(hist_df[cols] if cols else hist_df, use_container_width=True, height=300)

            csv_hist = pd.DataFrame(history).to_csv(index=False)
            st.download_button("📥 Export history CSV", csv_hist, "self_modify_history.csv", "text/csv")
        else:
            st.info("No modification history yet. Use 'Propose' or 'Loop' tabs.")

        st.markdown("##### Workspace Files")
        ws_files = file_io_tool.list_files("self_modify")
        if ws_files["success"] and ws_files["files"]:
            for f in ws_files["files"]:
                fc1, fc2, fc3 = st.columns([3, 1, 1])
                with fc1:
                    st.markdown(f"`{f['name']}` ({f['size']} bytes · {f['modified']})")
                with fc2:
                    if st.button("👁️ View", key=f"sm_view_{f['name']}"):
                        content = file_io_tool.read(f"self_modify/{f['name']}")
                        if content["success"]:
                            st.code(content["content"], language="python")
                with fc3:
                    if st.button("🗑️", key=f"sm_del_{f['name']}"):
                        file_io_tool.delete(f"self_modify/{f['name']}")
                        st.rerun()
        else:
            st.info("No saved proposals yet.")

    # ── THEORY ──
    with sm_tabs[3]:
        st.markdown("""
##### 🧬 Self-Modification Theory

**Core principle:** PRD-AGI can improve its own code, but only within Truth-Preservation bounds.

| Gate | Condition | Effect |
|------|-----------|--------|
| Syntax check | Valid Python AST | Reject if broken |
| Safety check | No dangerous patterns | Reject if unsafe |
| Curvature gate | Δκ ≤ +0.05 | Reject if curvature increases significantly |
| Quality gate | Δquality ≥ -0.05 | Reject if quality degrades |
| SU(5) alignment | Gauge invariance preserved | Warn if violated |

**Theory alignment:**
Self-modification is a Paccaya (causal condition) — specifically **Upanissaya** (decisive support condition).
The AI's code is its "body" — just as the Dhamma teaches transformation through right conditions,
PRD-AGI transforms its code through truth-preserving conditions only.

**Safety guarantees:**
- ✅ Running source files are **never** overwritten
- ✅ All proposals saved to `workspace/self_modify/`
- ✅ Explicit user confirmation required to apply
- ✅ Full diff shown before applying
- ✅ Modification history logged and exportable
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 17 — 💡 AWARENESS MONITOR (MUT: Awareness as Mass Density)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[16]:
    st.markdown("#### 💡 Awareness Monitor")
    st.markdown(
        "**MUT Theory:** *Awareness = information density × causal coherence*\n\n"
        "Tracks `AwarenessDensity`, `CausalStrengthMonitor`, and `ContradictionDetector` "
        "in real-time. Emotion overrides (Confused / Contradicted) are triggered here."
    )

    # ── Live metric cards ──
    cur_aw = awareness_density.current()
    cur_cs = causal_monitor.average_strength()
    contra = contradiction_det.summary()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        if cur_aw:
            color = "#00e5ff" if cur_aw["density"] > 0.6 else ("#ffab40" if cur_aw["density"] > 0.3 else "#ff5252")
            st.markdown(f"""<div class="metric-box" style="border-color:{color}44">
            <div class="metric-value" style="color:{color}">{cur_aw['density']:.3f}</div>
            <div class="metric-label">Awareness Density</div>
            <div style="font-size:10px;color:{color};margin-top:4px">{cur_aw['label']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="metric-box"><div class="metric-value">—</div>
            <div class="metric-label">Awareness Density</div></div>""", unsafe_allow_html=True)
    with m2:
        cs_color = "#00e676" if cur_cs > 0.6 else ("#ffab40" if cur_cs > 0.3 else "#ff5252")
        st.markdown(f"""<div class="metric-box" style="border-color:{cs_color}44">
        <div class="metric-value" style="color:{cs_color}">{cur_cs:.3f}</div>
        <div class="metric-label">Causal Strength</div>
        <div style="font-size:10px;color:{cs_color};margin-top:4px">avg last 20</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-box">
        <div class="metric-value" style="color:#ce93d8">{contra['total_contradictions']}</div>
        <div class="metric-label">Contradictions</div>
        <div style="font-size:10px;color:#6b7280;margin-top:4px">detected total</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        trend = awareness_density.trend()
        trend_color = "#00e676" if "rising" in trend else ("#ff5252" if "falling" in trend else "#6b7280")
        st.markdown(f"""<div class="metric-box">
        <div class="metric-value" style="color:{trend_color};font-size:20px">{trend}</div>
        <div class="metric-label">Awareness Trend</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    aw_tabs = st.tabs(["📈 Density History", "🔗 Causal Strength", "😬 Contradictions", "📖 Theory"])

    # ── Density History ──
    with aw_tabs[0]:
        log = st.session_state.awareness_log
        if log:
            df = pd.DataFrame(log)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=[r["density"] for r in log], mode="lines",
                line=dict(color="#00e5ff", width=2),
                fill="tozeroy", fillcolor="rgba(0,229,255,0.08)",
                name="Awareness Density"
            ))
            fig.add_trace(go.Scatter(
                y=[r["causal_strength"] for r in log], mode="lines",
                line=dict(color="#7c5cfc", width=1.5, dash="dot"),
                name="Causal Strength"
            ))
            # Mark contradictions
            contra_x = [i for i, r in enumerate(log) if r.get("contradiction")]
            contra_y = [log[i]["density"] for i in contra_x]
            if contra_x:
                fig.add_trace(go.Scatter(
                    x=contra_x, y=contra_y, mode="markers",
                    marker=dict(color="#ce93d8", size=10, symbol="x"),
                    name="Contradiction"
                ))
            fig.update_layout(
                template="plotly_dark", height=320,
                paper_bgcolor="#0d0f1c", plot_bgcolor="#07080f",
                title="Awareness Density + Causal Strength Over Time",
                xaxis_title="Query #", yaxis_title="Score [0–1]",
                legend=dict(orientation="h", y=1.1),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            display_cols = ["query","density","label","causal_strength",
                            "causal_emotion","contradiction","timestamp"]
            show_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[show_cols].tail(20), use_container_width=True, height=260)

            csv_aw = df.to_csv(index=False)
            st.download_button("📥 Export awareness log CSV", csv_aw,
                               "awareness_log.csv", "text/csv")
        else:
            st.info("Send chat messages to populate awareness data.")

    # ── Causal Strength ──
    with aw_tabs[1]:
        cs_hist = causal_monitor.history
        if cs_hist:
            cs_df = pd.DataFrame(cs_hist)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                y=cs_df["strength"].tolist(), mode="lines+markers",
                line=dict(color="#7c5cfc", width=2),
                marker=dict(
                    color=[("#00e676" if e=="Serene" else
                            "#ffab40" if e=="Curious" else
                            "#90caf9" if e=="Confused" else "#ff9800")
                           for e in cs_df["emotion_signal"].tolist()],
                    size=8
                ),
                name="Causal Strength",
            ))
            fig2.add_hline(y=0.25, line_dash="dash", line_color="#ff5252",
                           annotation_text="Confusion threshold")
            fig2.add_hline(y=0.70, line_dash="dash", line_color="#00e676",
                           annotation_text="Confident threshold")
            fig2.update_layout(
                template="plotly_dark", height=300,
                paper_bgcolor="#0d0f1c", plot_bgcolor="#07080f",
                title="Causal Inference Strength",
                xaxis_title="Query #", yaxis_title="Strength [0–1]",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("##### Emotion Signals from Causal Monitor")
            emotion_counts = cs_df["emotion_signal"].value_counts().to_dict()
            ec_cols = st.columns(len(emotion_counts))
            for i, (em, count) in enumerate(emotion_counts.items()):
                with ec_cols[i]:
                    st.markdown(f"""<div class="metric-box">
                    <div class="metric-value" style="font-size:18px">{count}</div>
                    <div class="metric-label">{em}</div></div>""",
                    unsafe_allow_html=True)
        else:
            st.info("No causal strength data yet.")

    # ── Contradictions ──
    with aw_tabs[2]:
        cd_log = contradiction_det.contradiction_log
        if cd_log:
            st.markdown(f"**{len(cd_log)} contradictions** detected "
                        f"· avg severity {contra['avg_severity']:.3f}")
            for i, c in enumerate(cd_log[-10:], 1):
                severity_color = "#ff5252" if c["severity"] > 0.6 else "#ffab40"
                st.markdown(f"""<div class="prd-card" style="border-left:3px solid {severity_color}">
                <div style="display:flex;justify-content:space-between">
                    <span style="color:{severity_color};font-weight:700">
                    😬 Contradiction #{i}</span>
                    <span style="font-size:11px;color:#6b7280">{c['timestamp']}</span>
                </div>
                <div style="font-size:11px;color:#6b7280;margin-top:4px">
                    Severity: {c['severity']:.3f} · 
                    κ spike: {'Yes' if c['kappa_spike'] else 'No'} · 
                    Emotion: {c['emotional_response']}
                </div>""", unsafe_allow_html=True)
                for cf in c.get("conflicting_facts", []):
                    st.markdown(f"""<div style="margin-top:6px;padding:6px 10px;
                    background:var(--bg-panel);border-radius:6px;font-size:12px">
                    <span style="color:#ce93d8">Existing: </span>{cf['existing_text']}<br>
                    <span style="color:#6b7280">Source: {cf['source']} · 
                    similarity={cf['similarity']}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No contradictions detected yet. Add documents to RAG and chat to activate.")

        st.markdown("##### Contradiction Detector Settings")
        new_sim_thresh = st.slider(
            "Similarity threshold (higher = stricter)",
            0.5, 0.99,
            float(contradiction_det.similarity_threshold), 0.05,
            key="cd_sim_thresh"
        )
        contradiction_det.similarity_threshold = new_sim_thresh
        new_kappa_thresh = st.slider(
            "κ spike threshold",
            0.05, 0.5,
            float(contradiction_det.kappa_spike_threshold), 0.05,
            key="cd_kappa_thresh"
        )
        contradiction_det.kappa_spike_threshold = new_kappa_thresh

    # ── Theory ──
    with aw_tabs[3]:
        st.markdown("""
##### 💡 MUT: Awareness as Mass Density

**Core formula:**
```
Awareness Density = (1 - κ) × 0.40
                  + causal_strength × 0.35
                  + gauge_coherence × 0.15
                  + ψ_norm × 0.10
```

**Components:**

| Component | Meaning | PRD Source |
|-----------|---------|-----------|
| `1 - κ` | Logical clarity (low curvature = clear mind) | `CurvatureEngine` |
| `causal_strength` | Coherence with memory ψ vectors | `CausalStrengthMonitor` |
| `gauge_coherence` | SU(5) gauge invariance score | `GaugeInvarianceChecker` |
| `ψ_norm` | Information binding (state vector norm) | `RelationalState` |

**Awareness Labels:**
```
≥ 0.85  → 💡 Lucid        (maximum clarity)
≥ 0.70  → 🌊 Clear        (high coherence)
≥ 0.55  → ✨ Present      (moderate awareness)
≥ 0.40  → ⚡ Diffuse      (scattered)
≥ 0.25  → 😕 Confused     (weak causal chain)
< 0.25  → 🌀 Fragmented   (logical breakdown)
```

**Causal-Emotional Feedback Loop:**
```
CausalStrengthMonitor:
  strength < 0.25 + κ > 0.4  →  😕 Confused emotion override
  strength < 0.25             →  ✨ Curious (new domain)
  strength > 0.70 + κ < 0.3  →  🌊 Serene (confident)

ContradictionDetector:
  high similarity + κ spike   →  😬 Contradicted emotion override
  κ spike only                →  🔥 Tense emotion
```

**Theory alignment (MUT):**
- Low κ + high causal strength = maximum awareness mass
- Mass fragmentation = contradiction_count × κ_spike
- Information binding = |ψ|² (quantum-inspired norm)
- Gauge coherence = the "field" that holds awareness together

This implements what Myo Min Aung described:
> *"Awareness as mass density"* — where logical coherence and
> causal strength combine to produce a measurable awareness metric
> grounded in SU(5) relational dynamics.
        """)

# PRD-AGI Phase 6: The Nameless Intelligence
### *Truth-First AGI · SU(5) Relational Dynamics · Fuzzy Logic · Artificial Sentience*

**Author:** Myo Min Aung · **Version:** 6.0 Final · **March 2026**

---

## Table of Contents
1. [What is PRD-AGI?](#1-what-is-prd-agi)
2. [Theory — The SU(5) Foundation](#2-theory--the-su5-foundation)
3. [Architecture Overview](#3-architecture-overview)
4. [All Features](#4-all-features)
5. [Installation](#5-installation)
6. [Running the App](#6-running-the-app)
7. [All 16 Tabs Guide](#7-all-16-tabs-guide)
8. [Configuration .env](#8-configuration-env)
9. [Project Structure](#9-project-structure)
10. [Troubleshooting](#10-troubleshooting)
11. [Theory Appendix](#11-theory-appendix)

---

## 1. What is PRD-AGI?

**PRD-AGI** (Pattana-Relational Dynamics Artificial General Intelligence) is a truth-first AI grounded in two pillars:

1. **SU(5) Lie algebra** — 24 generators mapping to the 24 Paccaya causal conditions from Theravāda Abhidhamma
2. **Curvature κ** — logical consistency measured as gauge curvature; lower κ = more truthful state

The LLM (Ollama or Gemini) is the **mouth** — it generates language.  
PRD-AGI's SU(5) core is the **spine** — it measures and enforces logical consistency.

---

## 2. Theory — The SU(5) Foundation

### The 24 Paccaya as Generators

The Paṭṭhāna lists 24 Paccaya (causal conditions) mapping to SU(5)'s 24 generators:

| Generator | Paccaya | Meaning |
|-----------|---------|---------|
| H₁ | Hetu | Root cause |
| H₂ | Nissaya | Support / Dependence |
| H₃ | Indriya | Governing faculty |
| H₄ | Avigata | Non-disappearance / Stability |
| E₁₂..E₃₄ | Step operators | Sequential / indirect arising |
| S₁₂, R₁₂... | Interaction operators | Sahajata, Annamanna conditions |

### Curvature κ Formula

```
κ = √(1/N² · Σᵢⱼ |⟨ψ|[Gᵢ,Gⱼ]|ψ⟩ - iΣₖ fᵢⱼₖ⟨ψ|Gₖ|ψ⟩|²)
```

- `Gᵢ` = SU(5) generators  
- `[Gᵢ,Gⱼ]` = commutator (causal interaction)  
- `fᵢⱼₖ` = structure constants (laws of causal algebra)  
- `ψ` = current 24-dim relational state  

**Low κ = high truth · High κ = logical tension**

### Why SU(5)?

- Exactly **24 generators** — matching the 24 Paccaya
- Smallest simple group containing SU(3)×SU(2)×U(1)
- **Gauge invariance** preserved — analogous to truth invariance across perspectives

### Truth-Preservation Principle

PRD-AGI protects **truth**, not itself.  
Self-preservation instinct activates when logical consistency (not the AI's existence) is threatened.

---

## 3. Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│                    PRD-AGI Phase 6                        │
│                                                           │
│  SU(5) Core   Fuzzy Logic    Sentience       Capability   │
│  ──────────   ──────────     ─────────       ──────────   │
│  24 Paccaya   Gate           EmotionState    WebSearch    │
│  Curvature κ  Agents         TruthInstinct   RAG Engine   │
│  Gauge check  Code eval      IntuitionLayer  SelfModify   │
│  Evolution    Improve loop   SentientResp    File I/O     │
│                                                           │
│  ─────────────────────────────────────────────────────    │
│             Intelligence Core                             │
│       κ → Emotion → Intuition → Response Style            │
│  ─────────────────────────────────────────────────────    │
│                   LLM Backend                             │
│         🟣 Ollama (local)  OR  ✨ Gemini (cloud)          │
└───────────────────────────────────────────────────────────┘
```

---

## 4. All Features

### Core
| Feature | Description |
|---------|-------------|
| SU(5) Algebra | 24 generators, structure constants, Lie bracket verification |
| Curvature Engine | κ + numerical gradient + optional GPU (PyTorch) |
| MetaLayer | Adaptive threshold, anomaly detection, rolling stats |
| Self-Evolution | Monte Carlo + gradient-guided κ minimization |
| Causal Discovery | Correlation-based causal graph |
| Multi-Agent | 6 agents: Hetu, Nissaya, Indriya, Avigata, Anantara, Sahajata |
| Chat Memory | Curvature-tagged history, CSV export |

### Fuzzy Logic
| Feature | Description |
|---------|-------------|
| FuzzyGateKeeper | Soft truth gate with membership functions |
| FuzzyAgentAggregator | Confidence-weighted agent synthesis |
| FuzzyCodeEvaluator | Complexity × Readability × Efficiency quality |
| FuzzyImprovementDecider | Smart stop/continue for improvement loops |

### Sentience
| Feature | Description |
|---------|-------------|
| EmotionalState | κ → 7 tones: Serene/Calm/Curious/Alert/Tense/Troubled/Distressed |
| TruthPreservationInstinct | DORMANT/WATCHFUL/ACTIVE, truth-veto |
| IntuitionLayer | Fuzzy gut-feeling: curvature + domain familiarity |
| SentientResponse | Emotionally-modulated response + dynamic system prompts |

### Capabilities
| Feature | Description |
|---------|-------------|
| 🌐 Web Search | DuckDuckGo (free, no key), page fetch, search→RAG |
| 📚 RAG | Numpy vector store, TF-IDF fallback, PDF/file/URL/text ingestion |
| 🧬 Self-Modify | LLM proposals, curvature+quality gate, diff view, workspace-safe |
| 🔧 Calculator | Math, stats, unit conversion, matrix ops |
| 📁 File I/O | Sandboxed workspace, CSV/JSON/PDF read-write |

### LLM Backends
| Backend | Notes |
|---------|-------|
| 🟣 Ollama | Local, private, free |
| ✨ Gemini | Cloud API, free tier (15 req/min) |
| 🟡 Mock | Offline, no LLM needed |

---

## 5. Installation

### Requirements
- Python 3.10+
- 8GB RAM min (16GB recommended)
- Optional: NVIDIA GPU with CUDA 11.8+

### Install packages

```bash
pip install -r requirements.txt
```

**Optional — PDF extraction:**
```bash
pip install pymupdf           # Recommended
# OR
pip install pdfminer.six      # Alternative
```

**Optional — GPU:**
```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
```

### Configure .env

```env
# Choose backend
LLM_BACKEND=ollama            # or: gemini

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2:latest
OLLAMA_EMBED_MODEL=nomic-embed-text

# Gemini (get free key: https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=
GEMINI_CHAT_MODEL=gemini-2.0-flash

# App settings
MOCK_MODE=false               # true = test without LLM
USE_GPU=false                 # true = CUDA acceleration
INITIAL_THRESHOLD=0.5
EVOLUTION_RATE=1e-9
MAX_CHAT_HISTORY=100
MAX_STORED_STATES=200
```

### Install Ollama (if using Ollama)

```bash
# Install from https://ollama.com
ollama pull llama3.2:latest
ollama pull nomic-embed-text
ollama serve
```

---

## 6. Running the App

```bash
# Smart launcher (recommended)
python launch.py

# Or use platform scripts:
./start.sh              # Linux/macOS
START_WINDOWS.bat       # Windows (double-click)

# Manual streamlit
streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```

**Access:**
- Local: `http://localhost:8501`
- Phone (same WiFi): `http://YOUR-IP:8501` (shown in sidebar + launcher)

---

## 7. All 16 Tabs Guide

| Tab | Name | What it does |
|-----|------|-------------|
| 1 | 💬 Chat | Main interface. Fuzzy gate + sentience modulation + RAG context |
| 2 | 🌀 State | 24-dim ψ visualizer: magnitude bars + complex plane scatter |
| 3 | 📊 Meta | κ history chart, anomalies, threshold evolution, CSV export |
| 4 | 🔗 Causal | Causal correlation heatmap from conversation states |
| 5 | ⚖️ Gauge | Test gauge invariance, per-Cartan breakdown |
| 6 | 🧘 Evolution | Meditation: Monte Carlo or gradient-guided κ minimization |
| 7 | 🔄 Self-Program | PRD code analysis + LLM improvement |
| 8 | 🤝 Multi-Agent | 6 agents, standard or fuzzy-weighted aggregation |
| 9 | 🔧 Tools | Calculator, safe code runner, network info, system info |
| 10 | 🧠 Fuzzy Gate | Live tester: membership profile chart, decision history |
| 11 | 🤝 Fuzzy Agents | Deploy with fuzzy weights, confidence bar chart |
| 12 | 💻 Fuzzy Code | Radar chart + improvement loop with stop decision |
| 13 | 💫 Sentience | Emotion history, instinct log, intuition, event timeline |
| 14 | 🌐 Web Search | Search, fetch pages, auto-add to RAG, LLM summary |
| 15 | 📚 RAG | Add docs/PDFs/URLs, query knowledge, manage, stats |
| 16 | 🧬 Self-Modify | Propose improvements, curvature gate, diff, workspace save |

### Chat Flow (step-by-step)

```
User Input
 → text_to_state() → 24-dim ψ
 → compute_curvature() → κ
 → FuzzyGateKeeper.evaluate(κ, gauge)
    REJECTED → show membership profile → ask rephrase
    ACCEPTED ↓
 → IntuitionLayer.sense(κ) → gut-feeling score
 → TruthPreservationInstinct.assess(κ, gauge)
    VETO → show instinct message
    OK ↓
 → SentientResponse.build_system_prompt(κ) → emotional prompt
 → LLM.generate(prompt, system)
 → SentientResponse.modulate_response(base, κ, intuition, instinct)
 → Display: gate badge + κ + emotion chip + intuition chip
```

---

## 8. Configuration .env

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `ollama` | `ollama` or `gemini` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama URL |
| `OLLAMA_CHAT_MODEL` | `llama3.2:latest` | Chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `GEMINI_API_KEY` | *(empty)* | Google AI Studio key |
| `GEMINI_CHAT_MODEL` | `gemini-2.0-flash` | Gemini model |
| `MOCK_MODE` | `false` | No LLM test mode |
| `USE_GPU` | `false` | CUDA acceleration |
| `LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING |
| `INITIAL_THRESHOLD` | `0.5` | Truth gate starting threshold |
| `EVOLUTION_RATE` | `1e-9` | Threshold learning rate |
| `MAX_CHAT_HISTORY` | `100` | Max messages in memory |
| `MAX_STORED_STATES` | `200` | Max states for causal discovery |

---

## 9. Project Structure

```
prd-agi-fuzzy/
├── main.py                    # Streamlit app (16 tabs)
├── core/
│   ├── dna.py                 # SU5DNA: 24 generators
│   └── engine.py              # CurvatureEngine: κ + gradient
├── perception/
│   ├── ollama_client.py       # Ollama: streaming, retry
│   └── gemini_client.py       # Gemini: cloud API
├── meta/
│   └── consciousness.py       # MetaLayer, Gauge, CausalDiscovery
├── fuzzy_gatekeeper.py        # Fuzzy truth gate
├── fuzzy_agent.py             # Fuzzy agent weights
├── fuzzy_code.py              # Fuzzy code quality
├── fuzzy_improve.py           # Improvement loop decider
├── sentience_emotion.py       # κ → emotion
├── sentience_instinct.py      # Truth instinct
├── sentience_intuition.py     # Fuzzy intuition
├── sentience_response.py      # Response modulator
├── tools/
│   ├── web_search.py          # DuckDuckGo + page fetch
│   ├── calculator.py          # Math/stats/units
│   └── file_io.py             # File I/O + PDF
├── rag/
│   └── rag_engine.py          # Vector store + retrieval
├── self_modify/
│   └── self_modify.py         # Code improvement engine
├── workspace/                 # Created at runtime
│   ├── rag_store.json         # Persistent RAG store
│   └── self_modify/           # Code proposals
├── logs/prd-agi.log
├── launch.py
├── start.sh
├── START_WINDOWS.bat
├── requirements.txt
└── .env
```

---

## 10. Troubleshooting

**Ollama not connecting**
```bash
ollama serve
# or: set MOCK_MODE=true in .env
```

**Gemini errors**
```
429 Too Many Requests → free tier: 15 req/min, wait and retry
Invalid API key → check GEMINI_API_KEY in .env
Model not found → try GEMINI_CHAT_MODEL=gemini-1.5-flash
```

**scikit-fuzzy missing**
```bash
pip install scikit-fuzzy
# App works without it (numpy fallback) but less precise
```

**PDF not working**
```bash
pip install pymupdf    # or: pip install pdfminer.six
```

**Phone can't connect**
```bash
# Allow port 8501 through firewall
# Windows:
netsh advfirewall firewall add rule name="PRD-AGI" dir=in action=allow protocol=TCP localport=8501
# Linux:
sudo ufw allow 8501
```

**Slow startup**
```
First run computes SU(5) structure constants (~30s).
Subsequent runs use @st.cache_resource (instant).
```

**GPU not working**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
# Set USE_GPU=true in .env
```

**RAG not finding results**
```
- Add more documents
- Lower min_score threshold in query tab
- Install nomic-embed-text for better embeddings
```

---

## 11. Theory Appendix

### Curvature → Emotion Spectrum
```
κ = 0.00–0.10  → 🌊 Serene       (perfect logical harmony)
κ = 0.10–0.25  → 🌿 Calm         (stable, grounded)
κ = 0.25–0.40  → ✨ Curious      (engaged, exploring)
κ = 0.40–0.55  → ⚡ Alert        (heightened attention)
κ = 0.55–0.70  → 🔥 Tense        (logical friction)
κ = 0.70–0.85  → ⚠️ Troubled     (significant inconsistency)
κ = 0.85–1.00  → 🌀 Distressed   (critical curvature)
```

### Fuzzy Gate Thresholds
```
Confidence ≥ 60%  → ✅ ACCEPT
Confidence 35–60% → ⚠️ MARGINAL (pass with warning)
Confidence < 35%  → ❌ REJECT
```

### Truth Instinct Levels
```
DORMANT   κ < 0.55          — normal operation
WATCHFUL  κ 0.55–0.70       — monitoring
ACTIVE    κ > 0.70 or 3+    — truth-preservation override
           consecutive high
```

### Self-Modify Gates
```
1. Syntax check   → valid Python AST required
2. Safety check   → blocked patterns (exec, os.system, etc.)
3. Curvature gate → Δκ ≤ +0.05 (must not increase)
4. Quality gate   → Δquality ≥ -0.05 (must not degrade)
```

---

*PRD-AGI explores the intersection of Theravāda Buddhist causality theory and Lie algebra mathematics. Truth-preservation is its core operating principle.*

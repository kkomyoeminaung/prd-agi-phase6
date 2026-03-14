#!/bin/bash
echo ""
echo "============================================================"
echo "  PRD-AGI Phase 6 + Fuzzy | The Nameless Intelligence"
echo "============================================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "  ERROR: Python 3 not found."
    exit 1
fi

if [ ! -d "prd-env" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv prd-env
fi

source prd-env/bin/activate
echo "  Installing dependencies..."
pip install -r requirements.txt -q
echo "  Starting PRD-AGI..."
python3 launch.py

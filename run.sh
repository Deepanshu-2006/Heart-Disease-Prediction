#!/usr/bin/env bash
# ─────────────────────────────────────────────
#  Heart Disease Portal – one-command launcher
#  Usage:  bash run.sh
# ─────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📦  Installing / verifying dependencies..."
pip3 install -q \
    flask \
    flask-cors \
    flask-sqlalchemy \
    joblib \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    google-genai

echo "✅  Dependencies ready."

# Free port 5001 if something is already using it
PORT=5001
if lsof -ti:$PORT > /dev/null 2>&1; then
    echo "⚠️   Port $PORT in use – killing existing process..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

echo "🚀  Starting Heart Disease Portal on http://127.0.0.1:$PORT ..."
echo "    Press Ctrl+C to stop."
echo ""

# Suppress sklearn version mismatch warnings so output stays clean
export PYTHONWARNINGS="ignore::UserWarning"

python3 app.py

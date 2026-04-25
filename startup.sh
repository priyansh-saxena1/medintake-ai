#!/bin/bash
set -e

MODEL="${MODEL_NAME:-qwen2.5:0.5b}"
OLLAMA_URL="http://localhost:11434"

echo "======================================"
echo " Clinical Intake Agent - Startup"
echo "======================================"

# ── Step 1: Start Ollama in the background ──────────────────────────────────
echo "[startup] Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# ── Step 2: Wait until Ollama is responsive ─────────────────────────────────
echo "[startup] Waiting for Ollama to be ready..."
MAX_WAIT=30
WAITED=0
until curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "[startup] ERROR: Ollama did not start within ${MAX_WAIT}s. Aborting."
        exit 1
    fi
done
echo "[startup] Ollama is ready! (waited ${WAITED}s)"

# ── Step 3: Pull / verify model ─────────────────────────────────────────────
echo "[startup] Pulling model '${MODEL}' (skipped if already cached)..."
ollama pull "${MODEL}"
echo "[startup] Model '${MODEL}' is ready."

# ── Step 4: Start FastAPI application ────────────────────────────────────────
echo "[startup] Launching FastAPI on port 7860..."
exec uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1

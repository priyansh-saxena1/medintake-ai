# ─── Stage: Base ──────────────────────────────────────────────────────────────
# Hugging Face Spaces uses port 7860 by default.
# We install Ollama (llama.cpp under the hood) for fast CPU inference.
FROM python:3.11-slim

# System dependencies for Ollama install script + curl
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    bash \
    && rm -rf /var/lib/apt/lists/*

# ─── Install Ollama ───────────────────────────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | bash

WORKDIR /app

# ─── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Copy source code ─────────────────────────────────────────────────────────
COPY app/ ./app/
COPY tests/ ./tests/
COPY startup.sh .

RUN chmod +x startup.sh

# ─── Environment ──────────────────────────────────────────────────────────────
# Set MOCK_LLM=false to use Ollama. Override at runtime if needed for testing.
ENV MOCK_LLM=false
ENV MODEL_NAME=qwen2.5:0.5b
ENV OLLAMA_HOST=http://localhost:11434

EXPOSE 7860

# startup.sh: boots Ollama, pulls model, starts FastAPI
CMD ["./startup.sh"]
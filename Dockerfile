FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG MOCK_LLM=false
ENV MOCK_LLM=${MOCK_LLM}

RUN if [ "$MOCK_LLM" != "true" ]; then \
    pip install llama-cpp-python --no-cache-dir && \
    mkdir -p /models && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('bartowski/Qwen2.5-0.5B-Instruct-GGUF', 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf', local_dir='/models')"; \
fi

COPY app/ ./app/
COPY tests/ ./tests/

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

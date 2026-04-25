FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# CPU-only torch (~220MB vs 2.4GB CUDA wheel)
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights at build time (baked into image)
# Swap model name here if you want a bigger one
ARG MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('${MODEL_NAME}'); \
    AutoModelForCausalLM.from_pretrained('${MODEL_NAME}')"

ENV MOCK_LLM=false
ENV MODEL_NAME=${MODEL_NAME}

COPY app/ ./app/
COPY tests/ ./tests/

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
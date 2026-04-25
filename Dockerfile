FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default to mock mode for HF Spaces (no GPU available)
ENV MOCK_LLM=true

COPY app/ ./app/
COPY tests/ ./tests/

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
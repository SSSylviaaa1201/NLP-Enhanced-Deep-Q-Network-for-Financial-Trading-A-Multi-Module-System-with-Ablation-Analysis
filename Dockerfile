FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download FinBERT model
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ProsusAI/finbert')" || true
RUN pip install sentence-transformers --quiet && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["bash", "-c", "python main.py --ablate && streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"]

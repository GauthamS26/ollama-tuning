# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Install system dependencies + Ollama CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY essay_bot.py finetune_llama_lora.py prepare_finetune_data.py ./
COPY training_data/ ./training_data/

# ── Default entry point ────────────────────────────────────────────────────────
# Override with `docker run ... python finetune_llama_lora.py` etc.
CMD ["python", "--help"]

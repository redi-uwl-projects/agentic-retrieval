# Hybrid Agentic Retrieval — local console pipeline
# Base image with CUDA 12.1 + cuDNN 9 + PyTorch 2.4.0
# Contains torch/torchvision/torchaudio
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

LABEL project="hybrid-agentic-retrieval" \
      description="LLM-Assisted Agentic Retrieval vs BM25/Dense Baselines"

# Install system dependencies
# build-essential: because some pip packages (e.g. pytrec_eval) build a C extension
# git: because some HF dataset loading scripts fetch via git
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python deps first. Then this layer is cached and does not need to be rebuilt across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY run.py .
COPY api_server.py .

# Cache locations in order not to download from the start across every run.
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    IR_DATASETS_HOME=/app/.cache/ir_datasets \
    GENSIM_DATA_DIR=/app/.cache/gensim-data \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

RUN mkdir -p /app/.cache/huggingface /app/.cache/ir_datasets /app/.cache/gensim-data /app/outputs

# Default entrypoint runs the console pipeline.
# The API server (api_server.py) needs the custom --entrypoint
# agentic-api service -- and listens on port 8000.
EXPOSE 8000
ENTRYPOINT ["python", "run.py"]
CMD ["--help"]

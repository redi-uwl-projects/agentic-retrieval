# Agentic Retrieval Python Implementation

## Implemented techniques

1. **Keyword Search** — baseline
2. **BM25** — baseline
3. **TD-IF** — baseline
4. **Word Embeddings** — baseline
5. **Dense Search** — baseline
6. **Agentic Search** — LLM plan → retrieve → fuse/re-rank → judge → reformulate → retry, with iterative query self-correction

## Building the docker image
`docker compose build`

## Run a batch test
`docker compose run --rm agentic-retrieval --dataset-source beir --dataset-name fiqa --max-queries 100 --output-dir /app/outputs/fiqa --llm-backend ollama --ollama-model qwen2.5:7b --llm-name Qwen/Qwen2.5-7B-Instruct --ollama-host http://host.docker.internal:11434 --enable-decomposition`

## Run an API endpoint test
`docker compose run --rm --service-ports agentic-api --dataset-source beir --dataset-name nq --max-queries 100 --output-dir /app/outputs/nq --llm-backend ollama --ollama-model qwen2.5:7b --llm-name Qwen/Qwen2.5-7B-Instruct --ollama-host http://host.docker.internal:11434 --enable-decomposition`

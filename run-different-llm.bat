docker compose run --rm agentic-retrieval --dataset-source beir --dataset-name scifact --max-queries 200 --output-dir /app/outputs/scifact_deepseek_r1_8b --llm-backend ollama --ollama-model deepseek-r1:8b --llm-name deepseek-ai/deepseek-r1:8b --ollama-host http://host.docker.internal:11434 --enable-decomposition

docker compose run --rm agentic-retrieval --dataset-source beir --dataset-name scifact --max-queries 200 --output-dir /app/outputs/scifact_llama3_1_8b --llm-backend ollama --ollama-model llama3.1:8b --llm-name  meta-llama/llama3.1:8b --ollama-host http://host.docker.internal:11434 --enable-decomposition

docker compose run --rm agentic-retrieval --dataset-source beir --dataset-name scifact --max-queries 200 --output-dir /app/outputs/scifact_llama3_2_3b --llm-backend ollama --ollama-model llama3.2 --llm-name  meta-llama/llama3.2 --ollama-host http://host.docker.internal:11434 --enable-decomposition

docker compose run --rm agentic-retrieval --dataset-source beir --dataset-name scifact --max-queries 200 --output-dir /app/outputs/scifact_qwen_2_5_7_b --llm-backend ollama --ollama-model qwen2.5:7b --llm-name Qwen/Qwen2.5-7B-Instruct --ollama-host http://host.docker.internal:11434 --enable-decomposition

docker compose run --rm agentic-retrieval --dataset-source beir --dataset-name nfcorpus --max-queries 200 --output-dir /app/outputs/nfcorpus_deepseek_r1_8b --llm-backend ollama --ollama-model deepseek-r1:8b --llm-name deepseek-ai/deepseek-r1:8b --ollama-host http://host.docker.internal:11434 --enable-decomposition

docker compose run --rm agentic-retrieval --dataset-source beir --dataset-name nfcorpus --max-queries 200 --output-dir /app/outputs/nfcorpus_llama3_1_8b --llm-backend ollama --ollama-model llama3.1:8b --llm-name  meta-llama/llama3.1:8b --ollama-host http://host.docker.internal:11434 --enable-decomposition

docker compose run --rm agentic-retrieval --dataset-source beir --dataset-name nfcorpus --max-queries 200 --output-dir /app/outputs/nfcorpus_llama3_2_3b --llm-backend ollama --ollama-model llama3.2 --llm-name  meta-llama/llama3.2 --ollama-host http://host.docker.internal:11434 --enable-decomposition

docker compose run --rm agentic-retrieval --dataset-source beir --dataset-name nfcorpus --max-queries 200 --output-dir /app/outputs/nfcorpus_qwen_2_5_7_b --llm-backend ollama --ollama-model qwen2.5:7b --llm-name Qwen/Qwen2.5-7B-Instruct --ollama-host http://host.docker.internal:11434 --enable-decomposition


pause
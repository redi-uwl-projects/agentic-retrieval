#!/usr/bin/env python
"""
Hybrid Agentic Retrieval — API server.

Loads the corpus, BM25/Dense/CrossEncoder index, and the LLM exactly once
at startup, then serves single queries over of the HTTP web server.

Using with python command:
    python api_server.py --dataset-source bright --dataset-name psychology \\
        --corpus-subset-size 20000 --llm-name Qwen/Qwen2.5-3B-Instruct \\
        --host 0.0.0.0 --port 8000

Using with docker:
    docker run --rm --gpus all -p 8000:8000 \\
        --entrypoint python agentic-retrieval:latest api_server.py \\
        --dataset-source bright --dataset-name psychology

Using the API:
    - open http://localhost:8000/ in a browser for the live query UI
    - POST http://localhost:8000/query  {"query": "...", "top_k": 5}
    - GET  http://localhost:8000/health to check if the server is running
"""
import logging
import uvicorn

from src.agent_factory import build_agent
from src.api import STATE, app
from src.config import build_arg_parser, get_use_system_prompt, parse_config_from_args
from src.data_loading import build_eval_subset, load_dataset_auto
from src.retrieval import (
    BM25Index,
    CrossEncoderReranker,
    DenseIndex,
    KeywordIndex,
    TfidfIndex,
    WordEmbeddingIndex,
)
from src.utils import detect_device, set_seed, setup_logging

logger = logging.getLogger("agentic_retrieval.api_server")


def load_state(cfg) -> None:
    device = detect_device(cfg.device)
    logger.info("Device: %s", device)
    if device == "cpu":
        logger.warning("Running on CPU -- each query's agentic condition will be noticeably slower.")

    if cfg.hf_token:
        from huggingface_hub import login

        login(token=cfg.hf_token)
        logger.info("Logged in to Hugging Face.")

    use_system_prompt = get_use_system_prompt(cfg.llm_name)
    logger.info("Dataset: %s/%s", cfg.dataset_source, cfg.dataset_name)
    logger.info("LLM:     %s (use_system_prompt=%s)", cfg.llm_name, use_system_prompt)

    logger.info("Loading dataset...")
    corpus, queries, qrels, dataset_label = load_dataset_auto(cfg.dataset_source, cfg.dataset_name, cfg.split)

    logger.info("Building corpus subset...")
    eval_corpus, eval_queries, eval_qrels = build_eval_subset(
        corpus, queries, qrels, cfg.use_subset, cfg.corpus_subset_size, cfg.max_queries, cfg.random_seed
    )
    doc_ids = list(eval_corpus.keys())
    doc_texts = [eval_corpus[d] for d in doc_ids]

    logger.info("Building BM25 index over %d docs...", len(doc_ids))
    bm25_index = BM25Index(doc_ids, doc_texts)

    logger.info("Building dense (FAISS) index...")
    dense_index = DenseIndex(doc_ids, doc_texts, cfg.dense_model_name, device)

    logger.info("Building keyword index...")
    keyword_index = KeywordIndex(doc_ids, doc_texts)

    logger.info("Building TF-IDF index...")
    tfidf_index = TfidfIndex(doc_ids, doc_texts)

    logger.info("Building word embedding index...")
    glove_index = WordEmbeddingIndex(doc_ids, doc_texts)

    logger.info("Loading cross-encoder re-ranker...")
    cross_encoder = CrossEncoderReranker(cfg.cross_encoder_name, device, eval_corpus)

    logger.info("Loading LLM backend and building the agent...")
    agent = build_agent(cfg, use_system_prompt, device, bm25_index, dense_index, cross_encoder, eval_corpus)

    STATE.update(
        {
            "cfg": cfg,
            "dataset_label": dataset_label,
            "llm_name": cfg.llm_name,
            "eval_corpus": eval_corpus,
            "eval_queries": eval_queries,
            "eval_qrels": eval_qrels,
            "bm25_index": bm25_index,
            "dense_index": dense_index,
            "keyword_index": keyword_index,
            "tfidf_index": tfidf_index,
            "glove_index": glove_index,
            "cross_encoder": cross_encoder,
            "agent": agent,
        }
    )
    logger.info(
        "Ready. Dataset=%s | corpus=%d docs | LLM=%s | max_refine=%d",
        dataset_label,
        len(eval_corpus),
        cfg.llm_name,
        cfg.max_refine,
    )


def main() -> None:
    parser = build_arg_parser()
    server = parser.add_argument_group("API server")
    server.add_argument("--host", default="0.0.0.0", help="Bind address inside the container.")
    server.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    cfg = parse_config_from_args(args)
    setup_logging(cfg.log_level)
    set_seed(cfg.random_seed)

    load_state(cfg)

    logger.info("Starting API server on http://%s:%d  (UI at '/', POST queries to '/query')", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level=cfg.log_level.lower())

if __name__ == "__main__":
    main()

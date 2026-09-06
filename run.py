#!/usr/bin/env python

"""
Agentic Retrieval — Console pipeline
"""

import json
import logging
import os
import time
from tqdm import tqdm

from src.agent_factory import build_agent
from src.config import get_use_system_prompt, parse_config
from src.data_loading import build_eval_subset, load_dataset_auto
from src.evaluation import build_results_table, per_query_diff_table, refine_step_ablation
from src.plotting import save_comparison_chart

from src.retrieval import (
    BM25Index,
    CrossEncoderReranker,
    DenseIndex,
    KeywordIndex,
    TfidfIndex,
    WordEmbeddingIndex,
)

from src.utils import detect_device, set_seed, setup_logging

logger = logging.getLogger("agentic_retrieval.run")

def section(title: str) -> None:
    logger.info("\n")
    logger.info(title)
    logger.info("-" * len(title))


def main() -> None:
    cfg = parse_config()
    setup_logging(cfg.log_level)
    set_seed(cfg.random_seed)

    device = detect_device(cfg.device)
    logger.info("Device: %s", device)
    if device == "cpu":
        logger.warning(
            "Running on CPU, a warning because the LLM steps (plan/judge) will be slow."
        )

    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.hf_token:
        from huggingface_hub import login

        login(token=cfg.hf_token)
        logger.info("Logged in to Hugging Face.")
    elif "gemma" in cfg.llm_name.lower():
        logger.warning(
            "%s is gated on Hugging Face and no HF_TOKEN was provided."
            " Accept the license on the model page. "
            f"{cfg.llm_name}",
        )

    use_system_prompt = get_use_system_prompt(cfg.llm_name)
    logger.info("Dataset: %s/%s", cfg.dataset_source, cfg.dataset_name)
    logger.info("LLM:     %s (use_system_prompt=%s)", cfg.llm_name, use_system_prompt)

    # Dataset Loading
    section("1. Loading dataset")
    corpus, queries, qrels, dataset_label = load_dataset_auto(cfg.dataset_source, cfg.dataset_name, cfg.split)

    # Building evaluation subset
    section("2. Building evaluation subset")
    eval_corpus, eval_queries, eval_qrels = build_eval_subset(
        corpus, queries, qrels, cfg.use_subset, cfg.corpus_subset_size, cfg.max_queries, cfg.random_seed
    )

    doc_ids = list(eval_corpus.keys())
    doc_texts = [eval_corpus[d] for d in doc_ids]

    if not eval_queries:
        logger.error("No evaluation queries remain after filtering. Exit.")
        return

    sample_qid = next(iter(eval_queries))

    # BM25 Baseline
    section("3.1 Building BM25 index")
    bm25_index = BM25Index(doc_ids, doc_texts)
    logger.info(
        "BM25 initial test (%r): %s",
        eval_queries[sample_qid][:80],
        list(bm25_index.search(eval_queries[sample_qid], 5).items()),
    )

    # Dense Baseline
    section("3.2 Building dense (FAISS) index")
    dense_index = DenseIndex(doc_ids, doc_texts, cfg.dense_model_name, device)
    logger.info(
        "Dense initial test (%r): %s",
        eval_queries[sample_qid][:80],
        list(dense_index.search(eval_queries[sample_qid], 5).items()),
    )

    # Keyword Search Baseline
    section("3.3 Building Keyword index")
    keyword_index = KeywordIndex(doc_ids, doc_texts)
    logger.info(
        "Keyword initial test (%r): %s",
        eval_queries[sample_qid][:80],
        list(keyword_index.search(eval_queries[sample_qid], 5).items()),
    )

    # TF-IDF Baseline ----
    section("3.4 Building TF-IDF index")
    tfidf_index = TfidfIndex(doc_ids, doc_texts)
    logger.info(
        "TF-IDF initial test (%r): %s",
        eval_queries[sample_qid][:80],
        list(tfidf_index.search(eval_queries[sample_qid], 5).items()),
    )

    # Word Embedding (GloVe) Baseline
    section("3.5 Building Word Embedding index")
    glove_index = WordEmbeddingIndex(doc_ids, doc_texts)
    logger.info(
        "Word Embedding initial test (%r): %s",
        eval_queries[sample_qid][:80],
        list(glove_index.search(eval_queries[sample_qid], 5).items()),
    )

    # Cross-encoder
    section("4. Loading cross-encoder re-ranker")
    cross_encoder = CrossEncoderReranker(cfg.cross_encoder_name, device, eval_corpus)

    # LLM backend + Agent
    section("5. Loading LLM backend and building the agent")
    agent = build_agent(cfg, use_system_prompt, device, bm25_index, dense_index, cross_encoder, eval_corpus)

    # Intial test with only one query
    if cfg.demo_query:
        section("6. Single-query test")
        _ = agent.retrieve(eval_queries[sample_qid], qid=sample_qid, top_k=cfg.top_k_retrieve)
        demo_trace = agent.trace_log[-1]
        logger.info("Query : %s", demo_trace["original_query"])
        if demo_trace.get("subqueries"):
            logger.info("Decomposed into: %s", demo_trace["subqueries"])
        logger.info("Tools : %s  (%s)", demo_trace["tools"], demo_trace["plan_reasoning"])

        for s in demo_trace["stages"]:
            logger.info("-- stage %s --", s["stage"])
            logger.info("   query      : %s", s["query"])
            logger.info("   top-3 docs : %s", s["ranking"][:3])
            if s["judge_sufficient"] is not None:
                logger.info("   sufficient : %s  (%s)", s["judge_sufficient"], s["judge_reasoning"])

        logger.info("Stopped because: %s | total %.2fs", demo_trace["stop_reason"], demo_trace["total_time"])

        # reset before the real run
        agent.trace_log.clear()
        agent.llm_calls = 0
        agent.parse_failures = 0
        agent.judge_calls = 0
        agent.decompose_calls = 0


    # ---- Run all conditions ----
    section("7.1 Running Keyword Search over evaluation queries")
    keyword_run, keyword_latency = {}, {}
    for qid, qtext in tqdm(eval_queries.items(), desc="Keyword Search"):
        t0 = time.perf_counter()
        keyword_run[qid] = keyword_index.search(qtext, cfg.top_k_retrieve)
        keyword_latency[qid] = time.perf_counter() - t0

    section("7.2 Running TF-IDF over evaluation queries")
    tfidf_run, tfidf_latency = {}, {}
    for qid, qtext in tqdm(eval_queries.items(), desc="TF-IDF"):
        t0 = time.perf_counter()
        tfidf_run[qid] = tfidf_index.search(qtext, cfg.top_k_retrieve)
        tfidf_latency[qid] = time.perf_counter() - t0

    section("7.3 Running Word Embeddings over evaluation queries")
    glove_run, glove_latency = {}, {}
    for qid, qtext in tqdm(eval_queries.items(), desc="Word Embeddings"):
        t0 = time.perf_counter()
        glove_run[qid] = glove_index.search(qtext, cfg.top_k_retrieve)
        glove_latency[qid] = time.perf_counter() - t0

    section("7.4 Running BM25 over evaluation queries")
    bm25_run, bm25_latency = {}, {}
    for qid, qtext in tqdm(eval_queries.items(), desc="BM25"):
        t0 = time.perf_counter()
        bm25_run[qid] = bm25_index.search(qtext, cfg.top_k_retrieve)
        bm25_latency[qid] = time.perf_counter() - t0

    section("7.5 Running Dense Search over evaluation queries")
    dense_run, dense_latency = {}, {}
    for qid, qtext in tqdm(eval_queries.items(), desc="Dense Search"):
        t0 = time.perf_counter()
        dense_run[qid] = dense_index.search(qtext, cfg.top_k_retrieve)
        dense_latency[qid] = time.perf_counter() - t0

    section(f"7.6. Running Agentic Search (max_refine={cfg.max_refine}) over evaluation queries")
    t_run = time.perf_counter()
    for qid, qtext in tqdm(eval_queries.items(), desc=f"Agentic Search (max_refine={cfg.max_refine})"):
        agent.retrieve(qtext, qid=qid, top_k=cfg.top_k_retrieve)
    wall = time.perf_counter() - t_run

    traces = agent.trace_log

    agentic_run = {
        tr["qid"]: {d: float(len(tr["stages"][-1]["ranking"]) - i) for i, d in enumerate(tr["stages"][-1]["ranking"])}
        for tr in traces
    }

    agentic_latency = {tr["qid"]: tr["total_time"] for tr in traces}

    logger.info("Completed %d queries in %.1f min", len(traces), wall / 60)
    logger.info(
        "LLM calls: %d | Judge calls: %d | Decompose calls: %d | JSON parse failures: %d (%.1f%%)",
        agent.llm_calls,
        agent.judge_calls,
        agent.decompose_calls,
        agent.parse_failures,
        100 * agent.parse_failures / max(agent.llm_calls, 1),
    )

    # Evaluation
    section("8. Evaluation")
    conditions = {
        "Keyword Search": (keyword_run, keyword_latency, 0),
        "TF-IDF": (tfidf_run, tfidf_latency, 0),
        "BM25": (bm25_run, bm25_latency, 0),
        "Word Embeddings": (glove_run, glove_latency, 0),
        "Dense Search": (dense_run, dense_latency, 0),
        "Agentic Search": (agentic_run, agentic_latency, agent.judge_calls),
    }

    results_df, per_query = build_results_table(conditions, eval_qrels)
    logger.info("\n%s", results_df.round(4).to_string())

    # Per-query comparison to see where Agentic performs better or worse than the baselines
    section("8.1 Per-query comparison: Agentic Search vs baselines")
    diff_df = per_query_diff_table(per_query, eval_queries, "Agentic Search", ["Dense Search", "BM25"])

    logger.info(
        "Top 5 queries where Agentic beats Dense Search most:\n%s",
        diff_df.head(5)[["query", "NDCG@10 diff vs Dense Search"]].to_string(),
    )

    # Chart printing
    section("8.2 Saving comparison chart")
    save_comparison_chart(results_df, cfg.output_dir)

    #  Refinement-step
    section("9. Refinement-step logging")
    ablation_df = refine_step_ablation(traces, eval_qrels, cfg.max_refine)
    logger.info("\n%s", ablation_df.round(4).to_string())

    # Save results for inspection
    section("10. Saving results")
    results_df.round(4).to_csv(os.path.join(cfg.output_dir, "main_comparison_table.csv"))
    ablation_df.round(4).to_csv(os.path.join(cfg.output_dir, "query_refine_steps_log.csv"))
    diff_df.round(4).to_csv(os.path.join(cfg.output_dir, "agentic_per_query_diff_log.csv"))

    if cfg.save_traces:
        with open(os.path.join(cfg.output_dir, "run_log.json"), "w") as f:
            json.dump(traces, f, indent=2, default=str)

    with open(os.path.join(cfg.output_dir, "run_parameters.json"), "w") as f:
        config_dict = cfg.to_dict()
        config_dict.pop("hf_token", None) # remove the Hugging Face token from the saved config for security reasons
        json.dump(config_dict, f, indent=2, default=str)

    logger.info("Saved to %s/:", cfg.output_dir)
    for fn in sorted(os.listdir(cfg.output_dir)):
        logger.info(" - %s", fn)

    logger.info("Dataset: %s | LLM: %s (use_system_prompt=%s)", dataset_label, cfg.llm_name, use_system_prompt)
    section("Done")


if __name__ == "__main__":
    main()

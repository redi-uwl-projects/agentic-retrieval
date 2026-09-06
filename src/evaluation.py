"""
Evaluation: pytrec_eval metrics (NDCG@10, Recall@10, MRR@10)
The main comparison table and the refine-step ablation.

Alternative libraries that can be used: ir-measures, ranx

"""

import logging
import numpy as np
import pandas as pd
import pytrec_eval

logger = logging.getLogger("agentic_retrieval.evaluation")

METRIC_COLS = ["NDCG@10", "Recall@10", "MRR@10", "Avg_Latency_ms", "Total_LLM_Calls"]

def truncate_run(run: dict, k: int) -> dict:
    return {qid: dict(sorted(docs.items(), key=lambda x: -x[1])[:k]) for qid, docs in run.items()}


def evaluate_run(qrels: dict, run: dict):
    run = {qid: docs for qid, docs in run.items() if qid in qrels}

    ev = pytrec_eval.RelevanceEvaluator(qrels, {"recall.10", "ndcg_cut.10"})
    res = ev.evaluate(run)
    mrr_ev = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    mrr_res = mrr_ev.evaluate(truncate_run(run, 10))

    per_q = pd.DataFrame(
        [
            {
                "qid": qid,
                "NDCG@10": res.get(qid, {}).get("ndcg_cut_10", 0.0),
                "Recall@10": res.get(qid, {}).get("recall_10", 0.0),
                "MRR@10": mrr_res.get(qid, {}).get("recip_rank", 0.0),
            }
            for qid in qrels
        ]
    ).set_index("qid")

    return per_q[["NDCG@10", "Recall@10", "MRR@10"]].mean().to_dict(), per_q


def build_results_table(conditions: dict, eval_qrels: dict):
    """conditions: {name: (run, latency_dict, judge_calls)}"""

    per_query = {}
    rows = []
    for name, (run, latency, judge_calls) in conditions.items():
        means, pq = evaluate_run(eval_qrels, run)
        per_query[name] = pq
        means["Condition"] = name
        means["Avg_Latency_ms"] = float(np.mean(list(latency.values()))) * 1000.0 if latency else 0.0
        means["Total_LLM_Calls"] = judge_calls
        rows.append(means)

    results_df = pd.DataFrame(rows).set_index("Condition")[METRIC_COLS]
    return results_df, per_query


def _judge_calls_in_trace(tr: dict) -> int:
    """Log the number of judge calls for logging purposes"""
    return (len(tr["stages"]) - 1) if tr["stop_reason"] == "max_steps" else len(tr["stages"])


def run_from_traces(traces: list[dict], max_steps: int):
    """Reconstruct the run, per-query latency, and judge-call count for an agent limited to `max_steps` refinement iterations."""

    run, latency, judge_calls = {}, {}, 0
    for tr in traces:
        j = min(max_steps, len(tr["stages"]) - 1)
        stage = tr["stages"][j]
        ranked = stage["ranking"]
        run[tr["qid"]] = {d: float(len(ranked) - i) for i, d in enumerate(ranked)}
        latency[tr["qid"]] = stage["t_end"] + (stage["t_judge"] if j < max_steps else 0.0)
        judge_calls += min(max_steps, _judge_calls_in_trace(tr))
    return run, latency, judge_calls


def refine_step_ablation(traces: list[dict], eval_qrels: dict, max_refine: int) -> pd.DataFrame:
    rows = []
    for steps in range(0, max_refine + 1):
        run, latency, judge_calls = run_from_traces(traces, steps)
        means, _ = evaluate_run(eval_qrels, run)
        means["Condition"] = f"Agentic (max_refine={steps})"
        means["Avg_Latency_ms"] = float(np.mean(list(latency.values()))) * 1000.0 if latency else 0.0
        means["Total_LLM_Calls"] = judge_calls
        rows.append(means)
    return pd.DataFrame(rows).set_index("Condition")[METRIC_COLS]


def per_query_diff_table(per_query: dict, eval_queries: dict, compare_name: str, baseline_names: list[str]) -> pd.DataFrame:
    """Compare NDCG@10/Recall@10/MRR@10 per each query alongside each baseline"""
    
    compare = per_query[compare_name]
    df = compare[["NDCG@10", "Recall@10", "MRR@10"]].add_suffix(f" ({compare_name})").copy()

    for name in baseline_names:
        baseline = per_query[name]
        df[f"NDCG@10 ({name})"] = baseline["NDCG@10"]
        df[f"NDCG@10 diff vs {name}"] = compare["NDCG@10"] - baseline["NDCG@10"]

    df.insert(0, "query", [eval_queries.get(qid, "") for qid in df.index])
    return df.sort_values(f"NDCG@10 diff vs {baseline_names[0]}", ascending=False)

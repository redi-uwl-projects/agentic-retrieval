# Comparison chart, saved to disk using fig.savefig()

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

logger = logging.getLogger("agentic_retrieval.plotting")

def save_comparison_chart(results_df: pd.DataFrame, output_dir: str) -> str:
    score_metrics = ["NDCG@10", "Recall@10", "MRR@10"]
    x = np.arange(len(results_df.index))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for i, metric in enumerate(score_metrics):
        ax.bar(x + i * width, results_df[metric], width, label=metric)
    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df.index, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Effectiveness")
    ax.legend()

    ax = axes[1]
    ax.bar(results_df.index, results_df["Avg_Latency_ms"], color="tab:orange")
    ax.set_ylabel("ms / query")
    ax.set_title("Average Latency")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[2]
    ax.bar(results_df.index, results_df["Total_LLM_Calls"], color="tab:green")
    ax.set_ylabel("count")
    ax.set_title("Total LLM Calls")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_chart.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Saved comparison chart -> %s", path)
    return path

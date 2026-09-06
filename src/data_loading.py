"""
Dataset loading:

BEIR (via ir_datasets, falling back to HF datasets) and
BRIGHT (via HF datasets `xlangai/BRIGHT`).

Data structures returned by the loaders:
    corpus:  doc_id   -> text
    queries: query_id -> text
    qrels:   query_id -> {doc_id: relevance}

"""
import logging
import random
from typing import Optional
import ir_datasets
from tqdm import tqdm
from datasets import load_dataset

logger = logging.getLogger("agentic_retrieval.data")

def load_via_ir_datasets(dataset_name: str, split: str):
    
    ds = ir_datasets.load(f"beir/{dataset_name}/{split}")

    corpus = {}
    for doc in tqdm(ds.docs_iter(), desc="Loading corpus (ir_datasets)"):
        corpus[doc.doc_id] = (doc.title + " " + doc.text) if getattr(doc, "title", "") else doc.text

    queries = {q.query_id: q.text for q in ds.queries_iter()}

    qrels: dict = {}
    for qr in ds.qrels_iter():
        qrels.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance

    return corpus, queries, qrels


def load_via_hf_datasets(dataset_name: str, split: str):
    corpus_ds = load_dataset(f"beir/{dataset_name}", "corpus")["corpus"]
    corpus = {row["_id"]: row["text"] for row in tqdm(corpus_ds, desc="Loading corpus (HF datasets)")} # type: ignore

    queries_ds = load_dataset(f"beir/{dataset_name}", "queries")["queries"]
    all_queries = {row["_id"]: row["text"] for row in queries_ds} # type: ignore

    qrels_ds = load_dataset(f"beir/{dataset_name}-qrels")[split]
    qrels: dict = {}
    for row in qrels_ds:
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"]) # type: ignore

    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
    return corpus, queries, qrels


def load_via_bright(domain: str):
    """
    Load the BRIGHT domain (https://huggingface.co/datasets/xlangai/BRIGHT).
    BRIGHT has no train/dev/test split. Every example is an evaluation query.
    Binary relevance (qrels[qid][doc_id] = 1) is built from each example's gold_ids.
    """

    doc_rows = load_dataset("xlangai/BRIGHT", "documents")[domain]
    corpus = {row["id"]: row["content"] for row in tqdm(doc_rows, desc=f"Loading BRIGHT/{domain} corpus")} # type: ignore

    example_rows = load_dataset("xlangai/BRIGHT", "examples")[domain]

    queries, qrels = {}, {}
    for row in example_rows:
        qid = str(row["id"]) # type: ignore
        queries[qid] = row["query"] # type: ignore
        qrels[qid] = {doc_id: 1 for doc_id in row["gold_ids"] if doc_id in corpus} # type: ignore

    qrels = {qid: rels for qid, rels in qrels.items() if rels}
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    return corpus, queries, qrels


def load_dataset_auto(dataset_source: str, dataset_name: str, split: str):
    """Choose the correct function to load the required dataset."""
    
    if dataset_source == "bright":
        corpus, queries, qrels = load_via_bright(dataset_name)
        logger.info("Loaded BRIGHT/%s via HF datasets", dataset_name)
    else:
        try:
            corpus, queries, qrels = load_via_ir_datasets(dataset_name, split)
            logger.info("Loaded via ir_datasets")
        except Exception as e:  # noqa: BLE001 - mirrors notebook's broad fallback
            logger.warning("ir_datasets path failed (%s); falling back to HF datasets", e)
            corpus, queries, qrels = load_via_hf_datasets(dataset_name, split)
            logger.info("Loaded via HF datasets")

    dataset_label = f"BRIGHT/{dataset_name}" if dataset_source == "bright" else f"BEIR/{dataset_name}"
    logger.info("Corpus size:      %s", f"{len(corpus):,}")
    logger.info("Queries:          %s", f"{len(queries):,}")
    logger.info("Queries w/ qrels: %s", f"{len(qrels):,}")
    return corpus, queries, qrels, dataset_label


def build_eval_subset(
    corpus: dict,
    queries: dict,
    qrels: dict,
    use_subset: bool,
    corpus_subset_size: int,
    max_queries: Optional[int],
    seed: int,
):
    """Uses its own random number generator seeded independently so the corpus subsample is reproducible."""
    rng = random.Random(seed)

    query_ids = list(qrels.keys())
    if max_queries is not None:
        query_ids = query_ids[:max_queries]

    eval_queries = {qid: queries[qid] for qid in query_ids if qid in queries}
    eval_qrels = {qid: qrels[qid] for qid in eval_queries}

    # Always keep every gold/relevant doc in the corpus we search over
    gold_doc_ids = set()
    for _qid, rels in eval_qrels.items():
        gold_doc_ids.update(rels.keys())

    if use_subset:
        all_doc_ids = list(corpus.keys())
        rng.shuffle(all_doc_ids)
        keep_ids = set(all_doc_ids[:corpus_subset_size]) | gold_doc_ids
        eval_corpus = {did: corpus[did] for did in keep_ids if did in corpus}
    else:
        eval_corpus = corpus

    logger.info("Eval corpus size: %s", f"{len(eval_corpus):,}")
    logger.info("Eval queries:     %s", f"{len(eval_queries):,}")

    return eval_corpus, eval_queries, eval_qrels

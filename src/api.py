"""
Demo API for the Hybrid Agentic Retrieval pipeline.
This is a single-process demo server (using global state dict).

Loads the corpus + BM25/Dense/CrossEncoder indices + LLM once, then serves single queries over HTTP
to run live, one-by-one queries.

Every /query response includes not just the ranked documents but the
agent's full explanation: 
- which tools it picked and why
- what it retrieved at each refinement stage
- the judge's sufficiency verdict + reasoning at each step.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("agentic_retrieval.api")

DEMO_HTML_PATH = Path(__file__).parent / "static" / "demo.html"

app = FastAPI(
    title="Hybrid Agentic Retrieval — Demo API",
    description="BM25 vs Dense vs Agentic Search, one query at a time, with the agent's reasoning trace.",
    version="1.0.0",
)

# Populated once at startup by api_server.py::load_state(). Read-only after
# that point from the request-handling functions below.
STATE: dict = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query to run.")
    top_k: int = Field(10, ge=1, le=100, description="How many documents to return per condition.")
    conditions: list[str] = Field(
        default=["bm25", "dense", "keyword", "tfidf", "glove", "agentic"],
        description=(
            "Which condition(s) to run: any of 'bm25', 'dense', 'keyword', "
            "'tfidf', 'glove', 'agentic'."
        ),
    )
    max_snippet_chars: int = Field(220, ge=20, le=2000)
    qid: Optional[str] = Field(
        None,
        description=(
            "Known eval query id (see GET /queries) to check retrieved doc_ids against that "
            "query's qrels. Free-text queries with no qid get no relevance annotation."
        ),
    )


class DocHit(BaseModel):
    doc_id: str
    score: float
    snippet: str
    is_relevant: Optional[bool] = Field(
        None, description="True/False if qrels are known for this query, else None (unknown)."
    )
    relevance: Optional[int] = Field(
        None, description="Graded relevance from qrels (0 if not judged relevant), else None (unknown)."
    )


class StageTrace(BaseModel):
    stage: int
    query: str
    top_docs: list[str]
    judge_sufficient: Optional[bool]
    judge_reasoning: str


class AgenticExplanation(BaseModel):
    tools_chosen: list[str]
    plan_reasoning: str
    subqueries: list[str]
    stages: list[StageTrace]
    stop_reason: str
    n_refinements: int
    total_time_ms: float
    llm_calls_this_query: int


class QueryResponse(BaseModel):
    query: str
    dataset: str
    llm_name: str
    bm25: Optional[list[DocHit]] = None
    dense: Optional[list[DocHit]] = None
    keyword: Optional[list[DocHit]] = None
    tfidf: Optional[list[DocHit]] = None
    glove: Optional[list[DocHit]] = None
    agentic: Optional[list[DocHit]] = None
    agentic_explanation: Optional[AgenticExplanation] = None
    latency_ms: dict
    qrels_available: bool = Field(False, description="Whether req.qid matched a known eval query with qrels.")
    num_relevant_in_qrels: Optional[int] = Field(
        None, description="Total number of docs judged relevant for this qid, when qrels_available."
    )


class QueryInfo(BaseModel):
    qid: str
    text: str
    num_relevant: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hits(
    ranked: dict,
    corpus: dict,
    n: int,
    snippet_chars: int,
    qrels_for_q: Optional[dict] = None,
) -> list[DocHit]:
    top = sorted(ranked.items(), key=lambda x: -x[1])[:n]
    hits = []
    for d, s in top:
        is_relevant = None if qrels_for_q is None else d in qrels_for_q
        relevance = None if qrels_for_q is None else qrels_for_q.get(d, 0)
        hits.append(
            DocHit(
                doc_id=d,
                score=float(s),
                snippet=corpus.get(d, "")[:snippet_chars],
                is_relevant=is_relevant,
                relevance=relevance,
            )
        )
    return hits


def _require_ready():
    if not STATE:
        raise HTTPException(503, "Index/model still loading -- check container logs, then retry.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    if not STATE:
        return {"status": "loading"}
    return {
        "status": "ready",
        "dataset": STATE["dataset_label"],
        "corpus_size": len(STATE["eval_corpus"]),
        "llm": STATE["llm_name"],
        "max_refine": STATE["cfg"].max_refine,
    }


@app.get("/queries", response_model=list[QueryInfo])
def queries(limit: int = 200):
    """List known eval queries (with qrels) so a caller can pick a `qid` for
    /query and get relevance annotations back instead of free text with no
    qrels to check against."""
    _require_ready()
    eval_queries = STATE.get("eval_queries", {})
    eval_qrels = STATE.get("eval_qrels", {})
    items = [
        QueryInfo(qid=qid, text=text, num_relevant=sum(1 for r in eval_qrels.get(qid, {}).values() if r > 0))
        for qid, text in eval_queries.items()
    ]
    return items[: max(1, min(limit, 2000))]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    _require_ready()

    corpus = STATE["eval_corpus"]
    bm25_index = STATE["bm25_index"]
    dense_index = STATE["dense_index"]
    keyword_index = STATE["keyword_index"]
    tfidf_index = STATE["tfidf_index"]
    glove_index = STATE["glove_index"]
    agent = STATE["agent"]

    known = {"bm25", "dense", "keyword", "tfidf", "glove", "agentic"}
    unknown = set(req.conditions) - known
    if unknown:
        raise HTTPException(400, f"Unknown condition(s): {sorted(unknown)}. Use one of {sorted(known)}.")

    eval_qrels = STATE.get("eval_qrels", {})
    qrels_for_q = eval_qrels.get(req.qid) if req.qid else None

    latency: dict = {}
    resp = QueryResponse(
        query=req.query,
        dataset=STATE["dataset_label"],
        llm_name=STATE["llm_name"],
        latency_ms=latency,
        qrels_available=qrels_for_q is not None,
        num_relevant_in_qrels=(sum(1 for r in qrels_for_q.values() if r > 0) if qrels_for_q is not None else None),
    )

    if "bm25" in req.conditions:
        t0 = time.perf_counter()
        ranked = bm25_index.search(req.query, req.top_k)
        latency["bm25"] = round((time.perf_counter() - t0) * 1000, 1)
        resp.bm25 = _hits(ranked, corpus, req.top_k, req.max_snippet_chars, qrels_for_q)

    if "dense" in req.conditions:
        t0 = time.perf_counter()
        ranked = dense_index.search(req.query, req.top_k)
        latency["dense"] = round((time.perf_counter() - t0) * 1000, 1)
        resp.dense = _hits(ranked, corpus, req.top_k, req.max_snippet_chars, qrels_for_q)

    if "keyword" in req.conditions:
        t0 = time.perf_counter()
        ranked = keyword_index.search(req.query, req.top_k)
        latency["keyword"] = round((time.perf_counter() - t0) * 1000, 1)
        resp.keyword = _hits(ranked, corpus, req.top_k, req.max_snippet_chars, qrels_for_q)

    if "tfidf" in req.conditions:
        t0 = time.perf_counter()
        ranked = tfidf_index.search(req.query, req.top_k)
        latency["tfidf"] = round((time.perf_counter() - t0) * 1000, 1)
        resp.tfidf = _hits(ranked, corpus, req.top_k, req.max_snippet_chars, qrels_for_q)

    if "glove" in req.conditions:
        t0 = time.perf_counter()
        ranked = glove_index.search(req.query, req.top_k)
        latency["glove"] = round((time.perf_counter() - t0) * 1000, 1)
        resp.glove = _hits(ranked, corpus, req.top_k, req.max_snippet_chars, qrels_for_q)

    if "agentic" in req.conditions:
        t0 = time.perf_counter()
        llm_calls_before = agent.llm_calls
        ranked = agent.retrieve(req.query, qid=req.qid, top_k=req.top_k)
        latency["agentic"] = round((time.perf_counter() - t0) * 1000, 1)
        resp.agentic = _hits(ranked, corpus, req.top_k, req.max_snippet_chars, qrels_for_q)

        trace = agent.trace_log[-1]
        resp.agentic_explanation = AgenticExplanation(
            tools_chosen=trace["tools"],
            plan_reasoning=trace["plan_reasoning"],
            subqueries=trace.get("subqueries", []),
            stages=[
                StageTrace(
                    stage=s["stage"],
                    query=s["query"],
                    top_docs=s["ranking"][:5],
                    judge_sufficient=s["judge_sufficient"],
                    judge_reasoning=s["judge_reasoning"],
                )
                for s in trace["stages"]
            ],
            stop_reason=trace["stop_reason"],
            n_refinements=trace["n_refinements"],
            total_time_ms=round(trace["total_time"] * 1000, 1),
            llm_calls_this_query=agent.llm_calls - llm_calls_before,
        )

    return resp


@app.get("/", response_class=HTMLResponse)
def demo_ui():
    return DEMO_HTML_PATH.read_text(encoding="utf-8")

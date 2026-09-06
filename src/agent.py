"""
The agent phases: plan -> retrieve -> fuse/re-rank -> judge -> reformulate -> retry.

AgenticRetriever never reads eval_qrels inside the loop.
The stopping decision is made purely from the LLM's assessment of retrieved text snippets.
"""

import json
import logging
import re
import time

from .retrieval import rrf_fuse

logger = logging.getLogger("agentic_retrieval.agent")

ALL_TOOL_NAMES = {"bm25_search", "dense_search", "rrf_fuse", "cross_encoder_rerank"}
RETRIEVAL_TOOLS = {"bm25_search", "dense_search"}

TOOL_MENU = """- bm25_search: Lexical BM25 retrieval. Best for exact terms, names, acronyms and rare words.
- dense_search: Dense embedding retrieval. Best for paraphrase, synonyms, conceptual similarity.
- rrf_fuse: Fuses the rank lists from bm25_search and dense_search via Reciprocal Rank Fusion.
- cross_encoder_rerank: Re-scores the current top candidates against the query with a cross-encoder for higher precision at the top of the ranking."""

PLAN_SYSTEM_PROMPT = f"""Build a retrieval pipeline for a search query by choosing among these tools:
{TOOL_MENU}

Pick at least one retrieval tool (bm25_search and/or dense_search). You may add rrf_fuse and/or cross_encoder_rerank if required.
Respond with ONLY this JSON and nothing else:
{{"reasoning": "<one short sentence>", "tools": ["bm25_search", "dense_search", "rrf_fuse", "cross_encoder_rerank"]}}"""

DECOMPOSE_SYSTEM_PROMPT = """Break down complex search queries into simpler sub-queries if that helps retrieval.
Decompose if the query asks about multiple distinct things, compares multiple items, or has parts that are each easier to search for on their own.
Each sub-query must be a standalone search query covering one part of the original question, not a restatement of the same query.
Respond with ONLY this JSON and nothing else:
{"reasoning": "<one short sentence>", "subqueries": []}"""

JUDGE_SYSTEM_PROMPT = """You are a strict semantic search quality grader.

You are shown a query and short snippets of the top retrieved documents.
Decide whether these documents answer or match the query.

Be critical. If results are not sufficient, write a reformulated query that would retrieve better documents:
- expand abbreviations
- add domain-specific terminology
- make the information need more clear.

The reformulation must be a search query, not a sentence about the query.

Respond with ONLY this JSON and nothing else:
{"reasoning": "<one short sentence>", "sufficient": true, "reformulated_query": ""}"""


class AgenticRetriever:
    """
    LLM-assisted retrieval with iterative query self-correction.
    Loop: plan -> execute tool pipeline -> judge sufficiency -> reformulate -> retry

    Each step's tool pipeline is chosen by the LLM at plan time from
    {bm25_search, dense_search, rrf_fuse, cross_encoder_rerank} and re-used on
    every reformulated query within that call to retrieve().

    Talks to the llm_backend to plan, judge, and reformulate queries.
    """

    def __init__(
        self,
        llm_backend,
        bm25_index,
        dense_index,
        cross_encoder,
        eval_corpus: dict,
        max_refine_steps: int = 2,
        tool_k: int = 50,
        rerank_depth: int = 25,
        judge_snippets: int = 3,
        snippet_chars: int = 200,
        max_new_tokens: int = 96,
        rrf_k: int = 60,
        enable_decomposition: bool = False,
        max_subqueries: int = 3,
    ):
        self.llm_backend = llm_backend
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.cross_encoder = cross_encoder
        self.eval_corpus = eval_corpus

        self.max_refine_steps = max_refine_steps
        self.tool_k = tool_k
        self.rerank_depth = rerank_depth
        self.judge_snippets = judge_snippets
        self.snippet_chars = snippet_chars
        self.max_new_tokens = max_new_tokens
        self.rrf_k = rrf_k
        self.enable_decomposition = enable_decomposition
        self.max_subqueries = max_subqueries

        self.trace_log: list[dict] = []
        self.parse_failures = 0
        self.llm_calls = 0
        self.judge_calls = 0
        self.decompose_calls = 0

    # ---------- LLM Conversation ----------

    def _call_llm(self, system_prompt: str, user_msg: str) -> str:
        self.llm_calls += 1
        return self.llm_backend.generate(system_prompt, user_msg, self.max_new_tokens)

    def _parse_json(self, text: str, fallback: dict) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            self.parse_failures += 1
            return dict(fallback, parse_ok=False)
        try:
            parsed = json.loads(match.group(0))
            parsed["parse_ok"] = True
            return parsed
        except Exception:
            self.parse_failures += 1
            return dict(fallback, parse_ok=False)

    # ---------- agent steps ----------

    def _decompose(self, query: str) -> list[str]:
        """
        This is used to split the query into decomposed sub-queries.
        Returns [] if there is an error or when the LLM thinks that the query doesn't need decomposing.
        """

        #  Returning [] means going back to single-query behaviour.
        if not self.enable_decomposition:
            return []

        raw = self._call_llm(DECOMPOSE_SYSTEM_PROMPT, f"Query: {query}")
        self.decompose_calls += 1
        parsed = self._parse_json(raw, {"subqueries": []})

        seen = {query.strip().lower()}
        subqueries = []
        for sq in parsed.get("subqueries") or []:
            sq = str(sq).strip()
            if sq and sq.lower() not in seen:
                subqueries.append(sq)
                seen.add(sq.lower())
        return subqueries[: self.max_subqueries]

    def _plan(self, query: str) -> dict:
        raw = self._call_llm(PLAN_SYSTEM_PROMPT, f"Query: {query}")
        plan = self._parse_json(
            raw, {"tools": ["bm25_search", "dense_search", "rrf_fuse"], "reasoning": "fallback"}
        )
        chosen = [t for t in (plan.get("tools") or []) if t in ALL_TOOL_NAMES]
        if not any(t in RETRIEVAL_TOOLS for t in chosen):  # never let a bad parse starve retrieval
            chosen = ["bm25_search", "dense_search"] + [t for t in chosen if t not in RETRIEVAL_TOOLS]
        plan["tools"] = chosen
        return plan

    def _run_retrievers(self, tool_names: list[str], query: str) -> list[list[str]]:
        lists = []
        if "bm25_search" in tool_names:
            lists.append(list(self.bm25_index.search(query, self.tool_k).keys()))
        if "dense_search" in tool_names:
            lists.append(list(self.dense_index.search(query, self.tool_k).keys()))
        if not lists:
            lists.append(list(self.bm25_index.search(query, self.tool_k).keys()))
        return lists

    def _fuse_and_rerank(self, tool_names, rank_lists, rerank_query, top_k):
        if "rrf_fuse" in tool_names and len(rank_lists) > 1:
            candidates = list(rrf_fuse(rank_lists, k=self.rrf_k, top_k=top_k).keys())
        else:
            candidates = list(dict.fromkeys(d for lst in rank_lists for d in lst))[:top_k]
        if "cross_encoder_rerank" in tool_names and self.cross_encoder is not None:
            candidates = self.cross_encoder.rerank(rerank_query, candidates, depth=self.rerank_depth, top_k=top_k)
        return candidates

    def _snippets(self, ranked_ids: list[str], n: int | None = None) -> str:
        n = n or self.judge_snippets
        return "\n".join(
            f"[{i + 1}] {self.eval_corpus.get(d, '')[: self.snippet_chars]}"
            for i, d in enumerate(ranked_ids[:n])
        )


    def _judge(self, query: str, ranked_ids: list[str]) -> dict:
        user_msg = f"Query: {query}\n\nTop retrieved documents:\n{self._snippets(ranked_ids)}"
        raw = self._call_llm(JUDGE_SYSTEM_PROMPT, user_msg)
        self.judge_calls += 1
        return self._parse_json(
            raw, {"sufficient": True, "reformulated_query": "", "reasoning": "parse failure -> stop"}
        )

    # ---------- main entry point ----------

    def retrieve(self, query: str, qid=None, top_k: int = 100) -> dict:
        t_start = time.perf_counter()


        # Decompose
        subqueries = self._decompose(query)
        search_queries = [query] + subqueries  # original query is always included


        # Plan the tool selection
        plan = self._plan(query)
        tool_names = plan["tools"]

        # Retrieve the search query
        rank_lists: list[list[str]] = []
        for q in search_queries:
            rank_lists.extend(self._run_retrievers(tool_names, q))
        
        # Fuse and re-rank
        ranked = self._fuse_and_rerank(tool_names, rank_lists, query, top_k)
        t_stage0 = time.perf_counter()

        trace = {
            "qid": qid,
            "original_query": query,
            "subqueries": subqueries,
            "tools": tool_names,
            "plan_reasoning": str(plan.get("reasoning", ""))[:200],
            "plan_parse_ok": plan.get("parse_ok", False),
            "stages": [
                {
                    "stage": 0,
                    "query": query,
                    "ranking": ranked[:top_k],
                    "t_end": t_stage0 - t_start,
                    "t_judge": 0.0,
                    "judge_sufficient": None,
                    "judge_reasoning": "",
                }
            ],
            "stop_reason": "max_steps" if self.max_refine_steps == 0 else None,
            "n_refinements": 0,
        }

        tried_queries = {query.strip().lower()}

        # Judge, Reformulate, Retry
        for step in range(self.max_refine_steps):
            t_j0 = time.perf_counter()
            verdict = self._judge(query, ranked)
            t_j1 = time.perf_counter()

            cur = trace["stages"][-1]
            cur["t_judge"] = t_j1 - t_j0
            cur["judge_sufficient"] = bool(verdict.get("sufficient", True))
            cur["judge_reasoning"] = str(verdict.get("reasoning", ""))[:200]

            new_query = str(verdict.get("reformulated_query", "") or "").strip()

            if verdict.get("sufficient", True):
                trace["stop_reason"] = "judged_sufficient"
                break
            if not new_query or new_query.lower() in tried_queries:
                trace["stop_reason"] = "no_usable_reformulation"
                break

            tried_queries.add(new_query.lower())

            # retry: check everything that has been retrieved so far
            rank_lists.extend(self._run_retrievers(tool_names, new_query))
            ranked = self._fuse_and_rerank(tool_names, rank_lists, new_query, top_k)
            t_stage = time.perf_counter()

            trace["stages"].append(
                {
                    "stage": step + 1,
                    "query": new_query,
                    "ranking": ranked[:top_k],
                    "t_end": t_stage - t_start,
                    "t_judge": 0.0,
                    "judge_sufficient": None,
                    "judge_reasoning": "",
                }
            )
            trace["n_refinements"] += 1
        else:
            if trace["stop_reason"] is None:
                trace["stop_reason"] = "max_steps"

        trace["total_time"] = time.perf_counter() - t_start
        self.trace_log.append(trace)

        # rank-based scores
        return {d: float(len(ranked) - i) for i, d in enumerate(ranked[:top_k])}

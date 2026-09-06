"""
Builds the LLM backend (transformers or Ollama) and the AgenticRetriever
from a RunConfig. Shared by run.py and api_server.py so both are consistent in how they build the agent.
"""

import logging

# transformers model import
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .llm_backends import TransformersBackend


from .agent import AgenticRetriever

logger = logging.getLogger("agentic_retrieval.agent_factory")


def build_llm_backend(cfg, use_system_prompt: bool, device: str):
    """Returns an object from `generate(system_prompt, user_msg, max_new_tokens) -> str`"""

    if cfg.llm_backend == "ollama":
        from .llm_backends import OllamaBackend

        logger.info(
            "LLM Backend: Ollama | model=%s host=%s",
            cfg.ollama_model,
            cfg.ollama_host,
        )

        return OllamaBackend(model=cfg.ollama_model, host=cfg.ollama_host, use_system_prompt=use_system_prompt)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_model = AutoModelForCausalLM.from_pretrained(
        cfg.llm_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    llm_model.eval()
    if device == "cpu":
        llm_model.to(device) # type: ignore

    n_params = sum(p.numel() for p in llm_model.parameters()) / 1e9

    logger.info(
        "LLM Backend: transformers | %s (%.2fB params) | use_system_prompt=%s",
        cfg.llm_name,
        n_params,
        use_system_prompt,
    )
    return TransformersBackend(llm_model, tokenizer, use_system_prompt=use_system_prompt)


def build_agent(cfg, use_system_prompt: bool, device: str, bm25_index, dense_index, cross_encoder, eval_corpus):
    llm_backend = build_llm_backend(cfg, use_system_prompt, device)

    agent = AgenticRetriever(
        llm_backend,
        bm25_index,
        dense_index,
        cross_encoder,
        eval_corpus,
        max_refine_steps=cfg.max_refine,
        tool_k=cfg.tool_k,
        rerank_depth=cfg.rerank_depth,
        judge_snippets=cfg.judge_snippets,
        snippet_chars=cfg.snippet_chars,
        max_new_tokens=cfg.max_new_tokens,
        rrf_k=cfg.rrf_k,
        enable_decomposition=cfg.enable_decomposition,
        max_subqueries=cfg.max_subqueries,

    )
    logger.info("The AgenticRetriever is now ready.")
    
    return agent

"""
Retrieval building blocks: 
- BM25 baseline, Dense (Sentence-Transformers + FAISS) baseline
- Keyword baseline, TF-IDF baseline, Word Embedding baseline
- Reciprocal Rank Fusion and CrossEncoder re-ranking

BM25Index and DenseIndex map onto the two retrieval tools the agent can choose from:
- bm25_search
- dense_search

"""

import logging
import re

import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import gensim.downloader as gensim_api
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi


logger = logging.getLogger("agentic_retrieval.retrieval")

token_regular_expression = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return token_regular_expression.findall(text.lower())


class BM25Index:
    """Lexical baseline / bm25_search tool."""

    def __init__(self, doc_ids: list[str], doc_texts: list[str]):
        self.doc_ids = doc_ids
        tokenized_corpus = [tokenize(t) for t in tqdm(doc_texts, desc="Tokenizing corpus (BM25)")]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query_text: str, top_k: int) -> dict:
        q_tokens = tokenize(query_text)
        scores = self.bm25.get_scores(q_tokens)
        if len(scores) == 0:
            return {}
        k = min(top_k, len(scores) - 1) if len(scores) > 1 else 0
        top_idx = np.argpartition(-scores, k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return {self.doc_ids[i]: float(scores[i]) for i in top_idx if scores[i] > 0}


class DenseIndex:
    """Dense embedding baseline."""

    def __init__(self, doc_ids: list[str], doc_texts: list[str], model_name: str, device: str):
        self.doc_ids = doc_ids
        self.model = SentenceTransformer(model_name, device=device)
        embeddings = self.model.encode(
            doc_texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query_text: str, top_k: int) -> dict:
        q_emb = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        scores, idx = self.index.search(q_emb, top_k)
        scores, idx = scores[0], idx[0]
        return {self.doc_ids[i]: float(s) for s, i in zip(scores, idx) if i != -1}


class KeywordIndex:
    """Keyword search baseline"""

    # score = number of distinct query
    # tokens that also appear in the document, no term weighting

    def __init__(self, doc_ids: list[str], doc_texts: list[str]):
        self.doc_ids = doc_ids
        self.doc_token_sets = [set(tokenize(t)) for t in tqdm(doc_texts, desc="Tokenizing corpus (Keyword)")]

    def search(self, query_text: str, top_k: int) -> dict:
        q_tokens = set(tokenize(query_text))
        scores = np.array([len(q_tokens & d_tokens) for d_tokens in self.doc_token_sets], dtype=np.float32)
        if len(scores) == 0:
            return {}
        k = min(top_k, len(scores) - 1) if len(scores) > 1 else 0
        top_idx = np.argpartition(-scores, k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return {self.doc_ids[i]: float(scores[i]) for i in top_idx if scores[i] > 0}


class TfidfIndex:
    """TF-IDF baseline: scikit-learn TfidfVectorizer + cosine similarity."""

    def __init__(self, doc_ids: list[str], doc_texts: list[str]):
        self.doc_ids = doc_ids
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None) # type: ignore
        self.matrix = self.vectorizer.fit_transform(tqdm(doc_texts, desc="Fitting TF-IDF"))

    def search(self, query_text: str, top_k: int) -> dict:
        q_vec = self.vectorizer.transform([query_text])
        scores = linear_kernel(q_vec, self.matrix).flatten()
        if len(scores) == 0:
            return {}
        k = min(top_k, len(scores) - 1) if len(scores) > 1 else 0
        top_idx = np.argpartition(-scores, k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return {self.doc_ids[i]: float(scores[i]) for i in top_idx if scores[i] > 0}


class WordEmbeddingIndex:
    """Word embedding baseline: GloVe vectors + cosine similarity."""

    def __init__(self, doc_ids: list[str], doc_texts: list[str], glove_name: str = "glove-wiki-gigaword-100"):
        self.doc_ids = doc_ids
        logger.info("Loading GloVe vectors (%s)...", glove_name)
        self.glove = gensim_api.load(glove_name)
        self.emb_dim = self.glove.vector_size # type: ignore

        doc_embeddings = np.vstack(
            [self._embed(t) for t in tqdm(doc_texts, desc="Embedding corpus")]
        )
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self.doc_embeddings_norm = doc_embeddings / norms

    def _embed(self, text: str) -> np.ndarray:
        vectors = [self.glove[t] for t in tokenize(text) if t in self.glove] # type: ignore
        if not vectors:
            return np.zeros(self.emb_dim, dtype=np.float32)
        return np.mean(vectors, axis=0).astype(np.float32) # type: ignore

    def search(self, query_text: str, top_k: int) -> dict:
        q_vec = self._embed(query_text)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return {}
        scores = self.doc_embeddings_norm @ (q_vec / q_norm)
        if len(scores) == 0:
            return {}
        k = min(top_k, len(scores) - 1) if len(scores) > 1 else 0
        top_idx = np.argpartition(-scores, k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return {self.doc_ids[i]: float(scores[i]) for i in top_idx if scores[i] > 0}


def rrf_fuse(rank_lists: list[list[str]], k: int, top_k: int) -> dict:
    """ Reciprocal Rank Fusion rrf_fuse tool. """
    scores: dict = {}
    for lst in rank_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda x: -x[1])[:top_k])


class CrossEncoderReranker:
    """cross_encoder_rerank tool. Re-scores the top candidates against the query."""

    def __init__(self, model_name: str, device: str, corpus: dict):
        self.model = CrossEncoder(model_name, device=device)
        self.corpus = corpus

    def rerank(self, query: str, candidate_ids: list[str], depth: int, top_k: int) -> list[str]:
        pool = candidate_ids[:depth]
        if not pool:
            return candidate_ids[:top_k]

        pairs = [[query, self.corpus.get(did, "")] for did in pool]
        scores = self.model.predict(pairs)
        reranked = [did for did, _ in sorted(zip(pool, scores), key=lambda x: -x[1])]

        backfill = [d for d in candidate_ids[depth:] if d not in reranked]
        return (reranked + backfill)[:top_k]

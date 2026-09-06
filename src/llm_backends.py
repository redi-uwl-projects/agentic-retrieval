"""
Two backends, one interface so AgenticRetriever doesn't need to know which is in use:
- TransformersBackend: using transformers and load LLM on it.
- OllamaBackend: talks to a running Ollama server's /api/chat endpoint and Ollama has the duty
  to load the LLM in it.
"""

import logging
import torch

import time
import requests

logger = logging.getLogger("agentic_retrieval.llm_backends")


class TransformersBackend:
    """Local HF transformers model."""

    def __init__(self, model, tokenizer, use_system_prompt: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.use_system_prompt = use_system_prompt

    def generate(self, system_prompt: str, user_msg: str, max_new_tokens: int) -> str:
       

        if self.use_system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
        else:
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_msg}"}]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic => reproducible
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


class OllamaBackend:
    """Talks to a running Ollama server. No model is loaded into this process."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        use_system_prompt: bool = True,
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_delay: float = 5.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.use_system_prompt = use_system_prompt
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(self, system_prompt: str, user_msg: str, max_new_tokens: int) -> str:

        if self.use_system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
        else:
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_msg}"}]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "num_predict": max_new_tokens,
                "seed": 42,
            },
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            resp = None
            try:
                resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()["message"]["content"]

            except requests.exceptions.ConnectionError as e:
                last_error = RuntimeError(
                    f"Ollama server could not be reached at {self.host}."
                )
                last_error.__cause__ = e

            except requests.exceptions.HTTPError as e:
                # Ollama puts the exception error in the JSON body
                # resp.raise_for_status() discards that, so there is need to show and format this message

                detail = ""
                if resp is not None:
                    try:
                        detail = resp.json().get("error", "")
                    except Exception:
                        detail = (resp.text or "")[:300]
                last_error = RuntimeError(
                    f"Ollama returned {resp.status_code if resp is not None else '?'} "
                    f"for the model '{self.model}': {detail or e}."
                )
                last_error.__cause__ = e

            except requests.exceptions.Timeout as e:
                last_error = RuntimeError(f"Ollama request timeout {self.timeout}s.")
                last_error.__cause__ = e

            if attempt < self.max_retries:
                logger.warning(
                    "Ollama call failed (attempt %d/%d): %s. Retry in %.0fs",
                    attempt + 1,
                    self.max_retries + 1,
                    last_error,
                    self.retry_delay,
                )
                time.sleep(self.retry_delay)

        raise last_error # type: ignore

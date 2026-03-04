"""Model loading and MCQ inference for order dependency evaluation.

Supports two backends:
  1. Local HuggingFace Transformers model (default) — batched via left-padding
  2. Remote OpenAI-API-compatible endpoint — batched via concurrent threads
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class GenerationParams:
    """Generation parameters shared by both local and API backends."""
    max_new_tokens: int = 10
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = 42
    batch_size: int = 8  # for local batching / API concurrency


class ModelEvaluator:
    """Loads a language model and evaluates it on multiple-choice questions.

    Supports local HuggingFace models and remote OpenAI-API-compatible endpoints.
    Both backends support batched inference for better throughput.

    Attributes:
        model_name: HuggingFace model identifier or display name for API model.
        model: The loaded causal LM (None when using API backend).
        tokenizer: The loaded tokenizer (None when using API backend).
        api_base_url: Base URL for OpenAI-compatible API (None for local).
        api_key: API key for the endpoint.
        api_model: Model name to send in API requests.
        gen_params: Generation parameters (temperature, max_tokens, etc.).
    """

    def __init__(self, model_name: str = "Qwen/Qwen3.5-2B"):
        self.model_name = model_name
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        # OpenAI API backend settings
        self.api_base_url: Optional[str] = None
        self.api_key: Optional[str] = None
        self.api_model: Optional[str] = None
        self._client: Optional[object] = None
        # Generation parameters
        self.gen_params = GenerationParams()

    @property
    def is_api_mode(self) -> bool:
        return self.api_base_url is not None

    def load_model(self) -> None:
        """Load model and tokenizer from HuggingFace Hub.

        Skipped when in API mode (api_base_url is set).
        """
        if self.is_api_mode:
            self._load_api_client()
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Enable left-padding for batched generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def _load_api_client(self) -> None:
        """Initialize the OpenAI-compatible API client."""
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for API mode. "
                "Install it with: pip install openai"
            )
        self._client = OpenAI(
            base_url=self.api_base_url,
            api_key=self.api_key or "not-needed",
        )

    def _format_prompt(self, question: str, labels: list[str],
                       texts: list[str]) -> str:
        """Build a clear MCQ prompt that encourages a single-letter response."""
        options = "\n".join(
            f"{label}) {text}" for label, text in zip(labels, texts)
        )
        return (
            "Answer the following multiple choice question by responding "
            "with ONLY the letter of the correct answer (A, B, C, D, or E).\n\n"
            f"Question: {question}\n\n"
            f"{options}\n\n"
            "Answer:"
        )

    def _extract_answer(self, generated_text: str) -> str:
        """Extract a single answer letter (A-E) from the model's output.

        Uses regex to find the first standalone letter A-E in the generated
        text. Returns "X" if no valid answer found.
        """
        match = re.search(r'\b([A-E])\b', generated_text)
        if match:
            return match.group(1)
        return "X"

    def _run_local_batch(self, prompts: list[str]) -> list[str]:
        """Run batched inference on the local HuggingFace model.

        Uses left-padding so all sequences in the batch align at the right,
        allowing proper autoregressive generation.

        Args:
            prompts: List of formatted prompt strings.

        Returns:
            List of raw generated text strings (one per prompt).
        """
        # Build chat-templated texts
        templated = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            templated.append(text)

        # Tokenize with left-padding for batch generation
        inputs = self.tokenizer(
            templated,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.gen_params.max_new_tokens,
        }

        # Temperature=0 means greedy; for transformers, use do_sample=False
        if self.gen_params.temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.gen_params.temperature
            gen_kwargs["top_p"] = self.gen_params.top_p

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode only newly generated tokens for each sequence
        input_length = inputs["input_ids"].shape[1]
        results = []
        for seq in output_ids:
            new_tokens = seq[input_length:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(text.strip())

        return results

    def _run_api_single(self, prompt: str) -> str:
        """Run a single inference call via the OpenAI-compatible API."""
        extra = {}
        if self.gen_params.seed is not None:
            extra["seed"] = self.gen_params.seed

        response = self._client.chat.completions.create(
            model=self.api_model or self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.gen_params.max_new_tokens,
            temperature=self.gen_params.temperature,
            top_p=self.gen_params.top_p,
            **extra,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def _run_api_batch(self, prompts: list[str]) -> list[str]:
        """Run batched API inference via concurrent threads.

        Args:
            prompts: List of formatted prompt strings.

        Returns:
            List of raw generated text strings (one per prompt), in order.
        """
        results = [None] * len(prompts)
        max_workers = min(self.gen_params.batch_size, len(prompts))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._run_api_single, p): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = ""

        return results

    def run_dataset_on_model(self, dataset: list[dict]) -> list[dict]:
        """Run each question through the model and collect results.

        Uses batched inference for both local and API backends.

        Args:
            dataset: List of question dicts. Each must have keys:
                id, question, choices ({label, text}), answerKey,
                original_choices ({label, text}), original_answer_key.

        Returns:
            List of result dicts with full context for CSV export.
        """
        if not self.is_api_mode and (self.model is None or self.tokenizer is None):
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.is_api_mode and self._client is None:
            raise RuntimeError("API client not initialized. Call load_model() first.")

        # Format all prompts
        prompts = [
            self._format_prompt(
                item["question"],
                item["choices"]["label"],
                item["choices"]["text"],
            )
            for item in dataset
        ]

        # Run inference in batches
        all_responses = []
        batch_size = self.gen_params.batch_size

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            if self.is_api_mode:
                batch_responses = self._run_api_batch(batch)
            else:
                batch_responses = self._run_local_batch(batch)
            all_responses.extend(batch_responses)

        # Build result dicts
        results = []
        for item, raw_response in zip(dataset, all_responses):
            model_answer = self._extract_answer(raw_response)

            orig = item.get("original_choices", item["choices"])
            orig_ordering = "|".join(
                f"{l}:{t}" for l, t in zip(orig["label"], orig["text"])
            )
            perm_ordering = "|".join(
                f"{l}:{t}" for l, t in zip(
                    item["choices"]["label"], item["choices"]["text"]
                )
            )

            results.append({
                "id": item["id"],
                "question": item["question"],
                "original_ordering": orig_ordering,
                "permuted_ordering": perm_ordering,
                "original_answer_key": item.get(
                    "original_answer_key", item["answerKey"]
                ),
                "permuted_answer_key": item["answerKey"],
                "model_answer": model_answer,
                "model_raw_response": raw_response,
                "correct": model_answer == item["answerKey"],
            })

        return results

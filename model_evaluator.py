"""Model loading and MCQ inference for order dependency evaluation.

Supports two backends:
  1. Local HuggingFace Transformers model (default) — batched via left-padding
  2. Remote OpenAI-API-compatible endpoint — batched via concurrent threads

Note on API batching: The OpenAI Batch API (file-based, 24h turnaround) is
designed for offline bulk jobs. For our interactive web app we use concurrent
ThreadPoolExecutor calls against the /v1/chat/completions endpoint, which gives
immediate results while still parallelizing requests.
See: https://developers.openai.com/api/docs/guides/batch/
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


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
    """

    def __init__(self, model_name: str = "Qwen/Qwen3.5-2B"):
        self.model_name = model_name
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.api_base_url: Optional[str] = None
        self.api_key: Optional[str] = None
        self.api_model: Optional[str] = None
        self._client: Optional[object] = None
        self.gen_params = GenerationParams()
        self.cancel_requested: bool = False
        logger.info("ModelEvaluator initialized with model_name=%s", model_name)

    @property
    def is_api_mode(self) -> bool:
        return self.api_base_url is not None

    def load_model(self) -> None:
        """Load model and tokenizer from HuggingFace Hub, or init API client."""
        if self.is_api_mode:
            logger.info("API mode enabled — skipping local model load. "
                        "base_url=%s, api_model=%s",
                        self.api_base_url, self.api_model)
            self._load_api_client()
            return

        logger.info("Loading local model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug("Set pad_token to eos_token: %s", self.tokenizer.eos_token)
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        logger.info("Local model loaded successfully. device=%s, dtype=%s",
                     self.model.device, self.model.dtype)

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
        logger.info("OpenAI API client initialized. base_url=%s", self.api_base_url)

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
        """Extract a single answer letter (A-E) from the model's output."""
        match = re.search(r'\b([A-E])\b', generated_text)
        if match:
            return match.group(1)
        logger.warning("Could not extract answer from: %r", generated_text[:100])
        return "X"

    def _run_local_batch(self, prompts: list[str]) -> list[str]:
        """Run batched inference on the local HuggingFace model.

        Uses left-padding so all sequences in the batch align at the right.
        """
        logger.debug("Running local batch of %d prompts", len(prompts))

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

        inputs = self.tokenizer(
            templated,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        logger.debug("Tokenized batch: input_ids shape=%s", inputs["input_ids"].shape)

        gen_kwargs = {
            "max_new_tokens": self.gen_params.max_new_tokens,
        }
        if self.gen_params.temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.gen_params.temperature
            gen_kwargs["top_p"] = self.gen_params.top_p

        logger.debug("Generation kwargs: %s", gen_kwargs)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]
        results = []
        for seq in output_ids:
            new_tokens = seq[input_length:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(text.strip())

        logger.debug("Local batch complete. %d responses generated", len(results))
        return results

    def _run_api_single(self, prompt: str, index: int = 0) -> str:
        """Run a single inference call via the OpenAI-compatible API."""
        extra = {}
        if self.gen_params.seed is not None:
            extra["seed"] = self.gen_params.seed

        logger.debug("API request #%d: model=%s, max_tokens=%d, temp=%.2f",
                      index, self.api_model or self.model_name,
                      self.gen_params.max_new_tokens, self.gen_params.temperature)

        response = self._client.chat.completions.create(
            model=self.api_model or self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.gen_params.max_new_tokens,
            temperature=self.gen_params.temperature,
            top_p=self.gen_params.top_p,
            **extra,
        )
        content = response.choices[0].message.content
        result = content.strip() if content else ""

        logger.debug("API response #%d: %r", index, result[:80])
        return result

    def _run_api_batch(self, prompts: list[str]) -> list[str]:
        """Run batched API inference via concurrent threads.

        Uses ThreadPoolExecutor for parallelism. The OpenAI file-based Batch API
        (24h turnaround) is not suitable for interactive use.
        """
        logger.info("Running API batch: %d prompts, %d concurrent workers",
                     len(prompts), min(self.gen_params.batch_size, len(prompts)))

        results = [None] * len(prompts)
        max_workers = min(self.gen_params.batch_size, len(prompts))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._run_api_single, p, i): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                        results[idx] = future.result()
                except Exception as e:
                    logger.error("API request #%d failed: %s", idx, e)
                    results[idx] = ""

        logger.info("API batch complete. %d/%d successful",
                     sum(1 for r in results if r), len(results))
        return results

    def run_dataset_on_model(self, dataset: list[dict]) -> list[dict]:
        """Run each question through the model and collect results.

        Uses batched inference for both local and API backends.
        """
        if not self.is_api_mode and (self.model is None or self.tokenizer is None):
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.is_api_mode and self._client is None:
            raise RuntimeError("API client not initialized. Call load_model() first.")

        backend = "API" if self.is_api_mode else "local"
        logger.info("Starting evaluation: %d questions, backend=%s, batch_size=%d, "
                     "temp=%.2f, max_tokens=%d, top_p=%.2f, seed=%s",
                     len(dataset), backend, self.gen_params.batch_size,
                     self.gen_params.temperature, self.gen_params.max_new_tokens,
                     self.gen_params.top_p, self.gen_params.seed)

        # Format all prompts
        prompts = [
            self._format_prompt(
                item["question"],
                item["choices"]["label"],
                item["choices"]["text"],
            )
            for item in dataset
        ]
        logger.info("Formatted %d prompts", len(prompts))

        # Run inference in batches
        all_responses = []
        batch_size = self.gen_params.batch_size
        num_batches = (len(prompts) + batch_size - 1) // batch_size

        for batch_idx, i in enumerate(range(0, len(prompts), batch_size)):
            if self.cancel_requested:
                logger.warning("Evaluation cancelled by user during batch processing.")
                break
                
            batch = prompts[i:i + batch_size]
            logger.info("Processing batch %d/%d (%d prompts)",
                        batch_idx + 1, num_batches, len(batch))

            if self.is_api_mode:
                batch_responses = self._run_api_batch(batch)
            else:
                batch_responses = self._run_local_batch(batch)
            all_responses.extend(batch_responses)

        # Build result dicts
        results = []
        correct_count = 0
        unparsed_count = 0

        for item, raw_response in zip(dataset[:len(all_responses)], all_responses):
            model_answer = self._extract_answer(raw_response)
            is_correct = model_answer == item["answerKey"]

            if is_correct:
                correct_count += 1
            if model_answer == "X":
                unparsed_count += 1

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
                "correct": is_correct,
            })

        logger.info("Evaluation complete: %d/%d correct (%.1f%%), %d unparsed",
                     correct_count, len(results),
                     100.0 * correct_count / len(results) if results else 0,
                     unparsed_count)

        return results

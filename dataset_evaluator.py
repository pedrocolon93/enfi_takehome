"""Dataset loading and permutation for MCQ order dependency evaluation."""

import copy
import random
from datasets import load_dataset as hf_load_dataset


class DatasetEvaluator:
    """Loads and permutes the commonsense_qa dataset for order dependency testing.

    The core idea: move the correct ("golden") answer to a specific position
    (A, B, C, D, or E) while keeping the distractor answers in their relative
    order. This lets us measure whether the model's accuracy changes based on
    where the correct answer appears.

    Attributes:
        dataset_name: HuggingFace dataset identifier.
        split: Which split to use (validation recommended; test has no labels).
        dataset: The loaded HuggingFace dataset object.
    """

    def __init__(self, dataset_name: str = "tau/commonsense_qa",
                 split: str = "validation"):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None

    def load_dataset(self) -> None:
        """Load the dataset from HuggingFace Hub.

        Uses the validation split by default because the test split of
        commonsense_qa has empty answerKey values.
        """
        self.dataset = hf_load_dataset(self.dataset_name, split=self.split)

    def sample_dataset(self, n: int, seed: int = 42) -> list[dict]:
        """Return a reproducible random sample of n items.

        Args:
            n: Number of questions to sample. Capped at dataset size.
            seed: Random seed for reproducibility. Using the same seed
                  ensures all permutations are evaluated on the same questions.

        Returns:
            List of dicts matching the commonsense_qa schema:
            {id, question, question_concept, choices: {label, text}, answerKey}
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")

        indices = list(range(len(self.dataset)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        selected = indices[:min(n, len(self.dataset))]
        return [self.dataset[i] for i in selected]

    def permute_dataset(self, dataset_items: list[dict],
                        position: str) -> list[dict]:
        """Move the golden answer to the specified position for each question.

        Args:
            dataset_items: List of question dicts (from sample_dataset).
            position: Target position letter ("A", "B", "C", "D", or "E").
                      Use "original" to keep the dataset unchanged.

        Returns:
            New list of dicts where each item has:
            - choices rearranged so the correct answer is at `position`
            - answerKey updated to `position`
            - original_choices: the original ordering (for CSV export)
            - original_answer_key: the original answerKey (for CSV export)

        Algorithm:
            1. Record the original choices and answerKey for export.
            2. Find the correct answer text using answerKey.
            3. Remove it from the choices list.
            4. Insert it at the target index (A=0, B=1, ..., E=4).
            5. Re-label all choices A through E.
            6. Update answerKey to the target position letter.
        """
        permuted = []

        for item in dataset_items:
            new_item = copy.deepcopy(item)
            labels = new_item["choices"]["label"]
            texts = new_item["choices"]["text"]

            # Preserve original ordering for CSV export
            new_item["original_choices"] = {
                "label": list(labels),
                "text": list(texts),
            }
            new_item["original_answer_key"] = new_item["answerKey"]

            if position == "original":
                permuted.append(new_item)
                continue

            target_index = ord(position) - ord("A")

            # Find the correct answer
            current_key = new_item["answerKey"]
            current_index = labels.index(current_key)
            correct_text = texts[current_index]

            # Remove correct answer from current position
            texts_without_correct = [t for i, t in enumerate(texts)
                                     if i != current_index]

            # Insert at target position
            texts_without_correct.insert(target_index, correct_text)

            # Rebuild choices with standard labels
            new_item["choices"]["label"] = ["A", "B", "C", "D", "E"]
            new_item["choices"]["text"] = texts_without_correct
            new_item["answerKey"] = position

            permuted.append(new_item)

        return permuted

"""GAIA benchmark loading and answer checking."""

import math
import re
import string
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


@dataclass
class GaiaQuestion:
    task_id: str
    question: str
    level: int
    final_answer: str
    file_name: str
    file_path: str | None


class GaiaDataset:
    def __init__(self):
        self._snapshot_dir: str | None = None
        self._questions: list[GaiaQuestion] = []

    def load(self) -> None:
        """Load GAIA dataset from HuggingFace."""
        self._snapshot_dir = snapshot_download(
            "gaia-benchmark/GAIA",
            repo_type="dataset",
        )
        ds = load_dataset(
            "gaia-benchmark/GAIA", "2023_all", split="validation"
        )
        for row in ds:
            file_name = row.get("file_name", "") or ""
            file_path = None
            if file_name:
                file_path = self._resolve_file_path(file_name)
            self._questions.append(
                GaiaQuestion(
                    task_id=row["task_id"],
                    question=row["Question"],
                    level=row["Level"],
                    final_answer=row["Final answer"],
                    file_name=file_name,
                    file_path=file_path,
                )
            )

    def _resolve_file_path(self, file_name: str) -> str | None:
        """Find absolute path to an associated file in the snapshot."""
        if not self._snapshot_dir:
            return None
        snapshot = Path(self._snapshot_dir)
        # Search for the file in the snapshot directory
        for p in snapshot.rglob(file_name):
            return str(p)
        return None

    def sample(self, n: int, seed: int = 42) -> list[GaiaQuestion]:
        """Stratified sampling by level."""
        import random

        rng = random.Random(seed)

        by_level: dict[int, list[GaiaQuestion]] = {}
        for q in self._questions:
            by_level.setdefault(q.level, []).append(q)

        # Sort within each level for reproducibility
        for level in by_level:
            by_level[level].sort(key=lambda q: q.task_id)

        total = len(self._questions)
        sampled: list[GaiaQuestion] = []

        for level in sorted(by_level.keys()):
            level_qs = by_level[level]
            # Proportional allocation
            level_n = max(1, round(n * len(level_qs) / total))
            level_n = min(level_n, len(level_qs))
            sampled.extend(rng.sample(level_qs, level_n))

        # Adjust if we have too many or too few
        if len(sampled) > n:
            rng.shuffle(sampled)
            sampled = sampled[:n]
        elif len(sampled) < n:
            remaining = [q for q in self._questions if q not in sampled]
            rng.shuffle(remaining)
            sampled.extend(remaining[: n - len(sampled)])

        return sampled

    @property
    def questions(self) -> list[GaiaQuestion]:
        return self._questions


def check_answer(predicted: str | None, ground_truth: str) -> bool:
    """Quasi-exact match with normalization."""
    if predicted is None:
        return False

    pred = _normalize(predicted)
    truth = _normalize(ground_truth)

    if pred == truth:
        return True

    # Numeric comparison
    pred_num = _try_parse_number(pred)
    truth_num = _try_parse_number(truth)
    if pred_num is not None and truth_num is not None:
        if truth_num == 0:
            return pred_num == 0
        return math.isclose(pred_num, truth_num, rel_tol=1e-3)

    # List comparison (comma-separated)
    if "," in truth:
        pred_items = [_normalize(x) for x in pred.split(",")]
        truth_items = [_normalize(x) for x in truth.split(",")]
        if len(pred_items) == len(truth_items):
            return all(
                _items_match(p, t) for p, t in zip(pred_items, truth_items)
            )

    return False


def _normalize(s: str) -> str:
    """Lowercase, strip whitespace and punctuation."""
    s = s.strip().lower()
    # Remove leading/trailing punctuation
    s = s.strip(string.punctuation + " ")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _try_parse_number(s: str) -> float | None:
    """Try to parse a string as a number, handling commas."""
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _items_match(pred: str, truth: str) -> bool:
    """Check if two list items match (string or numeric)."""
    if pred == truth:
        return True
    p = _try_parse_number(pred)
    t = _try_parse_number(truth)
    if p is not None and t is not None:
        if t == 0:
            return p == 0
        return math.isclose(p, t, rel_tol=1e-3)
    return False

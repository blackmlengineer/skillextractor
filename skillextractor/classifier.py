"""Sentence-level skill/no-skill classifier (Hugging Face text classification)."""

from __future__ import annotations

import logging
import warnings
from typing import Any, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIER = "loyoladatamining/skill-classifier-base-v2"


class SkillSentenceClassifier:
    """Wraps a binary sequence-classification model.

    LABEL_1 means "the sentence contains a skill statement"; LABEL_0 means it
    does not. ``score()`` returns P(LABEL_1) for each sentence.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLASSIFIER,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 64,
        positive_index: int = 1,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.positive_index = positive_index
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers and torch are required: pip install transformers torch") from exc

        dev = self.requested_device
        if dev and str(dev).startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(
                f"device='{dev}' requested for the classifier but torch.cuda.is_available() is False; "
                "falling back would hide the problem, so check your PyTorch build.",
                stacklevel=3,
            )
        if not dev:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(dev)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if model.config.num_labels < 2 or self.positive_index >= model.config.num_labels:
            raise ValueError(
                f"{self.model_name} has {model.config.num_labels} labels; "
                f"cannot use label index {self.positive_index} as the 'skill' class"
            )
        self._model = model.to(self._device).eval()
        logger.info("Loaded classifier %s on device: %s", self.model_name, self._device)

    @property
    def device(self) -> str:
        self._load()
        return str(self._device)

    def score(self, sentences: Sequence[str]) -> np.ndarray:
        """Return P(skill statement) for each sentence, shape (n,)."""
        if not sentences:
            return np.zeros(0, dtype=np.float32)
        self._load()
        import torch

        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(sentences), self.batch_size):
                batch = list(sentences[i : i + self.batch_size])
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, self.positive_index]
                out.append(probs.float().cpu().numpy())
        return np.concatenate(out)

"""SkillExtractor: sentence-level embedding match against skill taxonomies."""

from __future__ import annotations

import hashlib
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .classifier import DEFAULT_CLASSIFIER, SkillSentenceClassifier
from .sentences import split_sentences
from .taxonomies import (
    KNOWN_TAXONOMIES,
    PathLike,
    Taxonomy,
    TaxonomyNotFoundError,
    default_cache_dir,
    load_taxonomy,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "TechWolf/ConTeXT-Skill-Extraction-base"

# Result shape: {taxonomy: [per-text list of matches]}
Results = Dict[str, List[List[Dict[str, Any]]]]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class SkillExtractor:
    """Match sentences from free text to phrases in one or more taxonomies.

    Parameters
    ----------
    taxonomies:
        Names of taxonomies to match against (default: all four known ones).
    model:
        A sentence-transformers model name/path, or any object with an
        ``encode(list[str]) -> array`` method.
    threshold:
        Minimum cosine similarity (0-1) for a match. Default 0.8.
    refresh_embeddings:
        If True, recompute and overwrite cached phrase embeddings.
    taxonomy_paths / data_dir:
        Where to find ``<taxonomy>.csv`` files (columns: id, name, description).
        Only ESCO is bundled with the package.
    on_missing:
        ``"warn"`` (default) skips taxonomies whose CSV can't be found with a
        warning; ``"raise"`` raises ``TaxonomyNotFoundError`` instead.
    phrase_field:
        Which column is embedded for each taxonomy entry: ``"name"`` (default),
        ``"description"`` or ``"name+description"``.
    cache_dir:
        Where embeddings are cached (default ``~/.cache/skillextract``).
    return_scores:
        If True each match also carries ``score`` and ``sentence``.
    """

    def __init__(
        self,
        taxonomies: Sequence[str] = KNOWN_TAXONOMIES,
        model: Union[str, Any] = DEFAULT_MODEL,
        threshold: float = 0.8,
        refresh_embeddings: bool = False,
        taxonomy_paths: Optional[Mapping[str, PathLike]] = None,
        data_dir: Optional[PathLike] = None,
        on_missing: str = "warn",
        phrase_field: str = "name",
        cache_dir: Optional[PathLike] = None,
        batch_size: int = 64,
        device: Optional[str] = None,
        return_scores: bool = False,
        use_classifier: bool = False,
        classifier: Union[str, Any] = DEFAULT_CLASSIFIER,
        classifier_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 <= classifier_threshold <= 1.0:
            raise ValueError("classifier_threshold must be between 0 and 1")
        if on_missing not in ("warn", "raise"):
            raise ValueError("on_missing must be 'warn' or 'raise'")
        if phrase_field not in ("name", "description", "name+description"):
            raise ValueError("phrase_field must be 'name', 'description' or 'name+description'")
        if isinstance(taxonomies, str):
            taxonomies = [taxonomies]
        if not taxonomies:
            raise ValueError("at least one taxonomy is required")

        self.taxonomy_names = list(dict.fromkeys(taxonomies))
        self.threshold = threshold
        self.refresh_embeddings = refresh_embeddings
        self.phrase_field = phrase_field
        self.batch_size = batch_size
        self.return_scores = return_scores
        self.classifier_threshold = classifier_threshold
        self.classifier: Any = None
        if use_classifier:
            self.classifier = (
                SkillSentenceClassifier(classifier, device=device, batch_size=batch_size)
                if isinstance(classifier, str)
                else classifier  # any object with score(list[str]) -> P(skill) array
            )
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self._model_spec = model
        self._device = device
        self._model: Any = None if isinstance(model, str) else model

        self.taxonomies: Dict[str, Taxonomy] = {}
        self._embeddings: Dict[str, np.ndarray] = {}

        for name in self.taxonomy_names:
            try:
                self.taxonomies[name] = load_taxonomy(
                    name, taxonomy_paths=taxonomy_paths, data_dir=data_dir, cache_dir=self.cache_dir
                )
            except TaxonomyNotFoundError as exc:
                if on_missing == "raise":
                    raise
                warnings.warn(f"Skipping taxonomy '{name}': {exc}", stacklevel=2)
        if not self.taxonomies:
            raise TaxonomyNotFoundError("None of the requested taxonomies could be loaded.")

        for name, tax in self.taxonomies.items():
            self._embeddings[name] = self._load_or_build_embeddings(tax)

    # ------------------------------------------------------------------ model
    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is required: pip install sentence-transformers"
                ) from exc
            self._check_cuda_available()
            self._model = SentenceTransformer(self._model_spec, device=self._device)
            logger.info("Loaded %s on device: %s", self._model_spec, self.device)
        elif self._device and hasattr(self._model, "to") and not isinstance(self._model_spec, str):
            # A pre-built model object: honour the requested device once.
            if not getattr(self, "_moved", False):
                self._check_cuda_available()
                self._model = self._model.to(self._device)
                self._moved = True
        return self._model

    def _check_cuda_available(self) -> None:
        if not (self._device and str(self._device).startswith("cuda")):
            return
        try:
            import torch
        except ImportError:  # pragma: no cover
            return
        if not torch.cuda.is_available():
            warnings.warn(
                f"device='{self._device}' was requested but torch.cuda.is_available() is False "
                f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). "
                "You probably have a CPU-only PyTorch build; reinstall torch with CUDA support.",
                stacklevel=3,
            )

    @property
    def device(self) -> str:
        """The device the model is actually running on (loads the model if needed)."""
        return str(getattr(self.model, "device", "unknown"))

    def _encode(self, texts: List[str]) -> np.ndarray:
        model = self.model
        try:
            vectors = model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        except TypeError:  # minimal custom encoders
            vectors = model.encode(texts)
        return _normalize(np.asarray(vectors))

    # ------------------------------------------------------------- embeddings
    def _phrases(self, tax: Taxonomy) -> List[str]:
        if self.phrase_field == "name":
            return [r["name"] for r in tax.records]
        if self.phrase_field == "description":
            return [r["description"] or r["name"] for r in tax.records]
        return [f'{r["name"]}: {r["description"]}' if r["description"] else r["name"] for r in tax.records]

    def _cache_path(self, tax: Taxonomy, phrases: List[str]) -> Path:
        digest = hashlib.sha256()
        digest.update(str(self._model_spec if isinstance(self._model_spec, str) else type(self._model_spec).__name__).encode())
        digest.update(self.phrase_field.encode())
        for phrase in phrases:
            digest.update(phrase.encode("utf-8"))
            digest.update(b"\x00")
        model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(self._model_spec) if isinstance(self._model_spec, str) else type(self._model_spec).__name__)
        return self.cache_dir / "embeddings" / model_slug / f"{tax.name}-{digest.hexdigest()[:16]}.npy"

    def _load_or_build_embeddings(self, tax: Taxonomy) -> np.ndarray:
        phrases = self._phrases(tax)
        path = self._cache_path(tax, phrases)
        if path.is_file() and not self.refresh_embeddings:
            try:
                cached = np.load(path)
                if cached.shape[0] == len(phrases):
                    logger.info("Loaded cached embeddings for %s", tax.name)
                    return cached
            except Exception:  # corrupt cache: rebuild
                logger.warning("Ignoring unreadable embedding cache %s", path)
        logger.info("Embedding %d phrases for %s", len(phrases), tax.name)
        embeddings = self._encode(phrases)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, embeddings)
        except OSError as exc:
            logger.warning("Could not write embedding cache %s: %s", path, exc)
        return embeddings

    # ------------------------------------------------------------- extraction
    def _format(self, tax: Taxonomy, idx: int, score: float, sentence: str) -> Dict[str, Any]:
        rec = tax.records[idx]
        out: Dict[str, Any] = {"id": rec["id"], "name": rec["name"], "description": rec["description"]}
        if self.return_scores:
            out["score"] = round(float(score), 4)
            out["sentence"] = sentence
        return out

    def _filter_skill_sentences(self, per_text: List[List[str]]) -> List[List[str]]:
        """Keep only sentences the classifier labels as skill statements (LABEL_1)."""
        flat = [s for sents in per_text for s in sents]
        if not flat:
            return per_text
        probs = np.asarray(self.classifier.score(flat))
        if probs.shape[0] != len(flat):
            raise ValueError("classifier.score() must return one probability per sentence")
        keep = probs >= self.classifier_threshold
        logger.info("Classifier kept %d of %d sentences", int(keep.sum()), len(flat))
        filtered: List[List[str]] = []
        pos = 0
        for sents in per_text:
            filtered.append([s for s, k in zip(sents, keep[pos : pos + len(sents)]) if k])
            pos += len(sents)
        return filtered

    def extract(self, text: str, threshold: Optional[float] = None) -> Results:
        """Extract skills from one text. Returns ``{taxonomy: [[match, ...]]}``."""
        return self.batch_extract([text], threshold=threshold)

    def batch_extract(self, texts: Sequence[str], threshold: Optional[float] = None) -> Results:
        """Extract skills from several texts.

        Returns ``{taxonomy: [matches_for_text_0, matches_for_text_1, ...]}``
        where each ``matches_for_text_i`` lists one entry per (sentence, phrase)
        hit, in sentence order and, within a sentence, highest score first.
        """
        if isinstance(texts, str):
            raise TypeError("batch_extract expects a list of strings; use extract() for one text")
        thr = self.threshold if threshold is None else threshold
        if not 0.0 <= thr <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        per_text_sentences = [split_sentences(t or "") for t in texts]
        if self.classifier is not None:
            per_text_sentences = self._filter_skill_sentences(per_text_sentences)
        flat = [s for sents in per_text_sentences for s in sents]
        results: Results = {name: [[] for _ in texts] for name in self.taxonomies}
        if not flat:
            return results

        sentence_vecs = self._encode(flat)

        for name, tax in self.taxonomies.items():
            scores = sentence_vecs @ self._embeddings[name].T  # (n_sentences, n_phrases)
            offset = 0
            for t_idx, sents in enumerate(per_text_sentences):
                matches = results[name][t_idx]
                for s_idx, sentence in enumerate(sents):
                    row = scores[offset + s_idx]
                    hits = np.flatnonzero(row >= thr)
                    for idx in hits[np.argsort(-row[hits])]:
                        matches.append(self._format(tax, int(idx), row[idx], sentence))
                offset += len(sents)
        return results

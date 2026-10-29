"""Loading skill taxonomies from CSV files (columns: id, name, description)."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

KNOWN_TAXONOMIES = ("onet", "ukssc", "esco", "lightcast")
ENV_DATA_DIR = "SKILLEXTRACT_DATA_DIR"
REQUIRED_COLUMNS = ("id", "name", "description")

PathLike = Union[str, os.PathLike]


class TaxonomyNotFoundError(FileNotFoundError):
    """Raised when no CSV can be located for a taxonomy."""


@dataclass
class Taxonomy:
    name: str
    path: Path
    records: List[Dict[str, str]]

    def __len__(self) -> int:
        return len(self.records)


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "skillextract"


def _candidate_paths(
    name: str,
    taxonomy_paths: Optional[Mapping[str, PathLike]],
    data_dir: Optional[PathLike],
    cache_dir: Path,
) -> List[Path]:
    candidates: List[Path] = []
    if taxonomy_paths and name in taxonomy_paths:
        # An explicit path is authoritative: no silent fallback.
        return [Path(taxonomy_paths[name])]
    if data_dir:
        candidates.append(Path(data_dir) / f"{name}.csv")
    env_dir = os.environ.get(ENV_DATA_DIR)
    if env_dir:
        candidates.append(Path(env_dir) / f"{name}.csv")
    candidates.append(cache_dir / "taxonomies" / f"{name}.csv")
    candidates.append(Path(__file__).parent / "data" / f"{name}.csv")
    return candidates


def load_taxonomy(
    name: str,
    taxonomy_paths: Optional[Mapping[str, PathLike]] = None,
    data_dir: Optional[PathLike] = None,
    cache_dir: Optional[PathLike] = None,
) -> Taxonomy:
    """Locate and read ``<name>.csv``.

    Search order: explicit ``taxonomy_paths[name]``, ``data_dir``,
    ``$SKILLEXTRACT_DATA_DIR``, ``<cache_dir>/taxonomies``, bundled data.
    """
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    candidates = _candidate_paths(name, taxonomy_paths, data_dir, cache)
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        searched = "\n  ".join(str(p) for p in candidates)
        raise TaxonomyNotFoundError(
            f"No CSV found for taxonomy '{name}'. Searched:\n  {searched}\n"
            f"Provide a CSV with columns {', '.join(REQUIRED_COLUMNS)} via "
            f"taxonomy_paths={{'{name}': '/path/to/{name}.csv'}} or data_dir=..."
        )

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
        records = [
            {
                "id": (row["id"] or "").strip(),
                "name": (row["name"] or "").strip(),
                "description": (row["description"] or "").strip(),
            }
            for row in reader
            if (row["id"] or "").strip() and (row["name"] or "").strip()
        ]
    if not records:
        raise ValueError(f"{path} contains no usable rows")
    return Taxonomy(name=name, path=path, records=records)

"""Versioned page extraction cache and atomic output checkpoints."""

import json
from hashlib import sha256
from pathlib import Path


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def page_cache_key(page: dict) -> str:
    """Identify a parish/page pair without relying on a mutable CSV row index."""
    return _digest({key: page[key] for key in ("name", "diocese", "site_url", "page_url")})


def page_input_hash(page: dict) -> str:
    """Invalidate changed page information, ignoring CSV row indices."""
    return _digest({key: value for key, value in page.items() if key != "parish_index"})


def extractor_cache_dir(
    output_dir: Path, version: str, model: str, *, provider: str = "ollama",
    cache_root: Path | None = None,
) -> Path:
    root = cache_root if cache_root is not None else output_dir / "cache"
    identity = {"version": version, "model": model}
    if provider != "ollama":
        identity["provider"] = provider
    return root / "ministry_extraction" / _digest(identity)


def write_json_atomic(path: Path, value) -> None:
    """Replace a result only after the full JSON has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def write_csv_atomic(path: Path, frame) -> None:
    """Replace a CSV only after the full DataFrame has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False, encoding="utf-8")
    temporary_path.replace(path)


def load_cached_pages(cache_dir: Path, version: str, model: str, *, provider: str = "ollama") -> dict[str, dict]:
    """Load successful checkpoints; leave unreadable caches for inspection."""
    results = {}
    for path in sorted(cache_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("Expected a page object")
            if record.get("extractor_version") != version or record.get("llm_model") != model:
                continue
            if record.get("llm_provider", "ollama") != provider:
                continue
            if record.get("prediction_status") != "ok":
                continue
            key = page_cache_key(record)
            if path.stem != key or not record.get("input_hash"):
                raise ValueError("Cache identity or input hash is missing/invalid")
            ministry, category = record.get("ministry"), record.get("category")
            if not isinstance(ministry, str) or not isinstance(category, str) or bool(ministry) != bool(category):
                raise ValueError("Cache ministry/category must be strings, both set or both empty")
            results[key] = record
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Ignoring unreadable page cache {path.name}: {exc}")
    return results

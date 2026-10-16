"""Run page ministry extraction with cache migration, reuse, and checkpoints.

Notebook callers supply the rule-annotated pages and run settings. All cache
lookup, input invalidation, rule precedence, and CSV checkpointing stay here.
"""

from pathlib import Path

import pandas as pd

from utils.llm import LLMClient
from utils.ministry_prediction import EXTRACTOR_VERSION, predict_ministry
from utils.ministry_cache import (
    extractor_cache_dir, load_cached_pages, page_cache_key,
    page_input_hash, write_json_atomic, write_csv_atomic,
)


def extract_page_ministries(
    pages: pd.DataFrame, *, llm: LLMClient | None = None,
    output_dir: str | Path = "data/03_extract_ministries", page_limit: int | None = 10,
) -> pd.DataFrame:
    """Return/export combined current page results, limiting new prediction attempts.

    Successful page checkpoints and shared LLM responses are cached under the
    configured LLM_CACHE_DIR. Rule-excluded pages never call the LLM, and errors
    remain eligible for retry. Compatible prior page checkpoints are migrated.
    """
    llm = llm or LLMClient()
    llm_model = llm.model
    output_dir = Path(output_dir)
    if page_limit is not None and (type(page_limit) is not int or page_limit < 0):
        raise ValueError("page_limit must be None or a nonnegative integer")

    # Use the annotated page list produced by section 1.
    selected_pages = pages.copy()
    cache_dir = extractor_cache_dir(
        output_dir, EXTRACTOR_VERSION, llm_model,
        provider=llm.provider, cache_root=llm.cache_dir,
    )
    output_path = output_dir / "parish_ministries.csv"
    cached_results = load_cached_pages(cache_dir, EXTRACTOR_VERSION, llm_model, provider=llm.provider)
    # Reuse existing page checkpoints after moving the cache setting to .env.
    previous_cache_dir = extractor_cache_dir(output_dir, EXTRACTOR_VERSION, llm_model, provider=llm.provider)
    if previous_cache_dir != cache_dir:
        for key, record in load_cached_pages(previous_cache_dir, EXTRACTOR_VERSION, llm_model, provider=llm.provider).items():
            if key not in cached_results:
                record["llm_provider"] = llm.provider
                write_json_atomic(cache_dir / f"{key}.json", record)
                cached_results[key] = record
    # Do not carry obsolete predictions into output after rules exclude/change a page.
    combined_results = {
        page_cache_key(page): cached_results[page_cache_key(page)]
        for page in selected_pages.to_dict("records")
        if page["request_llm"] and page_cache_key(page) in cached_results
        and cached_results[page_cache_key(page)]["input_hash"] == page_input_hash(page)
    }
    result_columns = [
        "ministry", "category", "llm_provider", "llm_model", "extractor_version", "input_hash",
        "prediction_status", "prediction_error",
    ]
    # Include additional fields present in earlier cached batches as well as current inputs.
    output_columns = list(dict.fromkeys(
        list(selected_pages.columns)
        + [column for row in cached_results.values() for column in row if column not in result_columns]
        + result_columns
    ))
    new_calls = 0
    cache_hits = 0
    print(f"Extractor {EXTRACTOR_VERSION}; provider {llm.provider}; model {llm_model}; {len(cached_results)} cached pages")

    for page in selected_pages.to_dict("records"):
        key = page_cache_key(page)
        input_hash = page_input_hash(page)
        if not page["request_llm"]:
            # Rules run on every execution and take precedence over cached predictions.
            combined_results[key] = {
                **page,
                "ministry": page["rule_ministry"], "category": "",
                "llm_provider": llm.provider, "llm_model": llm_model,
                "extractor_version": EXTRACTOR_VERSION, "input_hash": input_hash,
                "prediction_status": "rule", "prediction_error": "",
            }
            print(f"rule {page['name']} {page['page_url']}: skip LLM ({page['matched_rule']})")
            continue
        cached = cached_results.get(key)
        if cached is not None and cached["input_hash"] == input_hash:
            combined_results[key] = {**cached, **page, "llm_provider": llm.provider}
            cache_hits += 1
            print(f"cache {page['name']} {page['page_url']}: {cached['ministry'] or 'no ministry'}")
            continue
        if page_limit is not None and new_calls >= page_limit:
            continue

        new_calls += 1
        ministry = ""
        category = ""
        status = "ok"
        error = ""
        try:
            result = predict_ministry(page, llm=llm)
            if result.ministry is not None:
                ministry = result.ministry.name
                category = result.ministry.category
        except Exception as exc:
            # Errors stay eligible for retry; successful no-ministry results are cached.
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        record = {
            **page,
            "ministry": ministry,
            "category": category,
            "llm_provider": llm.provider,
            "llm_model": llm_model,
            "extractor_version": EXTRACTOR_VERSION,
            "input_hash": input_hash,
            "prediction_status": status,
            "prediction_error": error,
        }
        if status == "ok":
            write_json_atomic(cache_dir / f"{key}.json", record)
            cached_results[key] = record
        combined_results[key] = record
        write_csv_atomic(output_path, pd.DataFrame(combined_results.values(), columns=output_columns))
        print(f"llm {new_calls} {page['name']} {page['page_url']}: {ministry or 'no ministry'} ({status})")

    page_results_df = pd.DataFrame(combined_results.values(), columns=output_columns)
    write_csv_atomic(output_path, page_results_df)
    print(f"Saved {len(page_results_df)} combined page results to {output_path}; {cache_hits} cache hits, {new_calls} LLM calls")
    return page_results_df

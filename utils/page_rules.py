"""Apply ordered ministry page rules using the apply_rules.py conventions."""

import re
from urllib.parse import unquote, urlparse

import pandas as pd


def load_rules(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"match_type", "pattern", "ministry", "request_llm"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Rules are missing columns: {sorted(required - set(frame.columns))}")
    rules = []
    for index, row in frame.iterrows():
        rule = {
            "line": index + 2,
            "match_type": row["match_type"].strip().lower(),
            "pattern": row["pattern"].strip().lower(),
            "ministry": row["ministry"].strip(),
            "request_llm": row["request_llm"].strip().lower() in {"true", "1", "yes", "y"},
        }
        if rule["match_type"] not in {"homepage", "equals", "contains", "starts_with", "ends_with", "regex"}:
            raise ValueError(f"Unknown match_type on rules line {rule['line']}: {rule['match_type']!r}")
        if rule["match_type"] == "regex":
            re.compile(rule["pattern"])
        rules.append(rule)
    return rules


def matches(rule, path):
    kind, pattern = rule["match_type"], rule["pattern"]
    if kind == "homepage":
        return path == ""
    if kind == "equals":
        return path == pattern.strip("/")
    if kind == "contains":
        return pattern in path
    if kind == "starts_with":
        return path.startswith(pattern) or (path + "/").startswith(pattern)
    if kind == "ends_with":
        return path.endswith(pattern)
    if kind == "regex":
        return re.search(pattern, path) is not None
    raise ValueError(f"Unknown match_type: {kind!r}")


def apply_page_rules(pages, rules, default_request_llm=True):
    """Annotate each parish/page pair; first matching rule wins.

    Keep the rule's ministry in rule_ministry, separate from LLM predictions.
    Unlike the reference script, do not collapse URLs across different parishes.
    """
    columns = ["path", "rule_ministry", "request_llm", "matched_rule"]
    result = pages.drop(columns=columns, errors="ignore").copy()
    identity = [column for column in ["parish_index", "name", "diocese", "site_url", "page_url"] if column in result.columns]
    result = result.drop_duplicates(subset=identity).reset_index(drop=True)
    annotations = []
    for url in result["page_url"]:
        path = unquote(urlparse(url).path).strip("/").lower()
        annotation = {"path": path, "rule_ministry": "", "request_llm": default_request_llm, "matched_rule": ""}
        for rule in rules:
            if matches(rule, path):
                annotation.update(
                    rule_ministry=rule["ministry"], request_llm=rule["request_llm"],
                    matched_rule=f"line {rule['line']}: {rule['match_type']} {rule['pattern']}".strip(),
                )
                break
        annotations.append(annotation)
    annotations_df = pd.DataFrame(annotations, columns=columns)
    annotations_df["request_llm"] = annotations_df["request_llm"].astype(bool)
    return pd.concat([result, annotations_df], axis=1)

# Each "# %%" marks a notebook cell. Paste each block into its own cell.

# %% Cell 1: config
PAGES_FILE = "1_site_pages.csv"
RULES_FILE = "rules.csv"
OUTPUT_FILE = "output.csv"
UNMATCHED_FILE = "unmatched.csv"  # pages that matched no rule
DEFAULT_REQUEST_LLM = True  # used when no rule matches

# %% Cell 2: helpers
import re
import pandas as pd
from urllib.parse import urlparse, unquote


def get_path(url):
    """'https://site.org/People/Bob/?x=1' -> 'people/bob'"""
    return unquote(urlparse(url).path).strip("/").lower()


def parse_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def load_rules(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rules = []
    for i, row in df.iterrows():
        rules.append(
            {
                "line": i + 2,  # line number in rules.csv (header is line 1)
                "match_type": row["match_type"].strip().lower(),
                "pattern": row["pattern"].strip().lower(),
                "ministry": row["ministry"].strip(),
                "request_llm": parse_bool(row["request_llm"]),
            }
        )
    return rules


def matches(rule, path):
    t, p = rule["match_type"], rule["pattern"]
    if t == "homepage":
        return path == ""
    if t == "equals":
        return path == p.strip("/")
    if t == "contains":
        return p in path
    if t == "starts_with":
        return path.startswith(p) or (path + "/").startswith(p)
    if t == "ends_with":
        return path.endswith(p)
    if t == "regex":
        return re.search(p, path) is not None
    raise ValueError(f"Unknown match_type: {t!r}")


def apply_rules(page_url, rules):
    path = get_path(page_url)
    for rule in rules:  # first match wins
        if matches(rule, path):
            return pd.Series(
                {
                    "path": path,
                    "ministry": rule["ministry"],
                    "request_llm": rule["request_llm"],
                    "matched_rule": f"line {rule['line']}: {rule['match_type']} {rule['pattern']}".strip(),
                }
            )
    return pd.Series(
        {
            "path": path,
            "ministry": "",
            "request_llm": DEFAULT_REQUEST_LLM,
            "matched_rule": "",
        }
    )


# %% Cell 3: run
pages = pd.read_csv(PAGES_FILE)

# Consolidate duplicate page URLs before applying rules
n_original = len(pages)
pages = pages.drop_duplicates(subset=["page_url"]).reset_index(drop=True)
# n_duplicates = n_original - len(pages)

rules = load_rules(RULES_FILE)

result = pd.concat(
    [pages, pages["page_url"].apply(lambda u: apply_rules(u, rules))],
    axis=1,
)

unmatched = result[result["matched_rule"] == ""][["site_url", "page_url", "path"]]

n_pages = len(result)
n_matched = (result["matched_rule"] != "").sum()
match_pct = n_matched / n_pages * 100 if n_pages else 0

print(f"{n_original} pages")
print(f"{n_pages} unique pages")
print(f"{n_matched} ({match_pct:.1f}%) matched a rule")
print(f"{len(unmatched)} matched no rule -> {UNMATCHED_FILE}")
print(f"{len(result)} pages -> {OUTPUT_FILE}")

unmatched.to_csv(UNMATCHED_FILE, index=False)
result.to_csv(OUTPUT_FILE, index=False)
result

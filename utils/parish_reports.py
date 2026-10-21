"""Build parish comparison reports from recorded ministries and cluster peers."""

import html
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote, urlsplit

import pandas as pd


IDENTITY_COLUMNS = ["parish_index", "name", "diocese", "site_url"]


def ministry_key(name):
    """Match case, punctuation, and spacing, without inventing synonym mappings."""
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", str(name)).casefold()).strip()


def validate_parish_rows(rows, parishes):
    """Join on row identifier AND parish identity to reject stale row-index joins."""
    if not parishes["parish_index"].is_unique:
        raise ValueError("Expected unique parish identifiers")
    checked = rows.merge(
        parishes[IDENTITY_COLUMNS], on=IDENTITY_COLUMNS, how="left",
        validate="many_to_one", indicator=True,
    )
    valid = checked["_merge"].eq("both")
    return checked.loc[valid].drop(columns="_merge"), checked.loc[~valid].drop(columns="_merge")


def summarize_ministries(results):
    """Aggregate successful labels per parish, retaining categories and evidence."""
    successful = results.loc[
        results["prediction_status"].isin(["ok", "rule"])
        & results["ministry"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    successful["ministry_key"] = successful["ministry"].map(ministry_key)
    rows = []
    for (parish_index, key), group in successful.groupby(["parish_index", "ministry_key"], sort=True):
        if not key:
            continue
        names = sorted({str(value).strip() for value in group["ministry"]}, key=str.casefold)
        categories = sorted({str(value).strip() for value in group["category"].fillna("") if str(value).strip()})
        urls = sorted({str(value).strip() for value in group["page_url"].fillna("") if str(value).strip()})
        rows.append({
            "parish_index": parish_index, "ministry_key": key, "ministry": names[0],
            "category": "; ".join(categories), "n_source_pages": len(urls),
            "source_urls": json.dumps(urls, ensure_ascii=False),
        })
    return pd.DataFrame(rows, columns=[
        "parish_index", "ministry_key", "ministry", "category", "n_source_pages", "source_urls",
    ])


def ministry_coverage(parishes, pages, results, ministries):
    """Distinguish absent labels from absent or incomplete extraction evidence."""
    coverage = parishes[IDENTITY_COLUMNS].copy()
    groups = {
        "n_listed_pages": pages,
        "n_result_pages": results,
        "n_resolved_pages": results.loc[results["prediction_status"].isin(["ok", "rule"])],
        "n_failed_pages": results.loc[~results["prediction_status"].isin(["ok", "rule"])],
    }
    for column, frame in groups.items():
        counts = frame.groupby("parish_index")["page_url"].nunique()
        coverage[column] = coverage["parish_index"].map(counts).fillna(0).astype(int)
    coverage["n_ministries"] = coverage["parish_index"].map(ministries.groupby("parish_index").size()).fillna(0).astype(int)
    listed = pages[["parish_index", "page_url"]].drop_duplicates()
    resolved = groups["n_resolved_pages"][["parish_index", "page_url"]].drop_duplicates()
    pending = listed.merge(resolved, on=["parish_index", "page_url"], how="left", indicator=True)
    counts = pending.loc[pending["_merge"].eq("left_only")].groupby("parish_index").size()
    coverage["n_unresolved_listed_pages"] = coverage["parish_index"].map(counts).fillna(0).astype(int)
    coverage["coverage_status"] = "resolved_or_rule_skipped"
    coverage.loc[coverage["n_unresolved_listed_pages"].gt(0), "coverage_status"] = "partial"
    coverage.loc[coverage["n_result_pages"].eq(0), "coverage_status"] = "no_extraction_results"
    coverage.loc[coverage["n_listed_pages"].eq(0), "coverage_status"] = "no_listed_pages"
    return coverage


def neighbor_ministry_gaps(matches, ministries):
    """One observation per target/peer ministry absent from the target's labels."""
    target_keys = ministries[["parish_index", "ministry_key"]].rename(columns={"parish_index": "target_parish_index"})
    gaps = matches.merge(
        ministries.rename(columns={"parish_index": "peer_parish_index"}),
        on="peer_parish_index", how="inner", validate="many_to_many",
    ).merge(target_keys.assign(on_target=True), on=["target_parish_index", "ministry_key"], how="left", validate="many_to_one")
    return gaps.loc[gaps["on_target"].isna()].drop(columns="on_target").reset_index(drop=True)


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def _md(value):
    text = html.escape(" ".join(_text(value).split()), quote=False)
    return re.sub(r"([\\`*_\[\]|])", r"\\\1", text)


def _link(label, url):
    url = _text(url)
    if urlsplit(url).scheme not in {"http", "https"}:
        return _md(label)
    encoded = quote(url, safe=":/?#&=%+@,;~!$'*-._")
    return f"[{_md(label)}]({encoded})"


def _number(value, decimals=0):
    try:
        number = float(value)
        return f"{number:,.{decimals}f}" if pd.notna(number) else "unavailable"
    except (ValueError, TypeError):
        return "unavailable"


def _school(value):
    return {"true": "yes", "false": "no", "1": "yes", "0": "no"}.get(_text(value).casefold(), "unavailable")


def _description(parish):
    return (
        f"{_md(parish.get('county', '')) or 'County unavailable'}; county population "
        f"{_number(parish.get('county_population'))}; median age {_number(parish.get('county_median_age'), 1)}; "
        f"social-link count {_number(parish.get('n_social_links'))}; predicted attached school: {_school(parish.get('attached_school'))}."
    )


def _coverage_text(row):
    status = {
        "resolved_or_rule_skipped": "all listed pages resolved or skipped by a rule",
        "partial": "partially processed", "no_extraction_results": "no extraction results available",
        "no_listed_pages": "no discovered pages available",
    }.get(row.coverage_status, row.coverage_status)
    return (
        f"{int(row.n_ministries)} distinct ministry label{'s' if row.n_ministries != 1 else ''}; {int(row.n_resolved_pages)} pages with successful "
        f"extraction or a rule decision; {int(row.n_listed_pages)} listed pages; "
        f"{int(row.n_unresolved_listed_pages)} listed pages unresolved. Coverage: {_md(status)}."
    )


def _filename(parish):
    ascii_name = unicodedata.normalize("NFKD", _text(parish["name"])).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:90] or "parish"
    return f"{int(parish['parish_index']):04d}_{slug}.md"


def generate_parish_reports(parishes, target_summary, matches, ministries, coverage, gaps, *, report_dir, target_diocese):
    """Write a Markdown report for every target, including targets without peers."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    lookup = parishes.set_index("parish_index", verify_integrity=True)
    coverage_lookup = coverage.set_index("parish_index", verify_integrity=True)
    expected = set(parishes.loc[parishes["diocese"].eq(target_diocese), "parish_index"])
    if not target_summary["target_parish_index"].is_unique or set(target_summary["target_parish_index"]) != expected:
        raise ValueError("Target summary does not match the configured diocese's parish records")
    manifest = []
    for target in target_summary.sort_values(["target_name", "target_parish_index"]).to_dict("records"):
        parish_id = target["target_parish_index"]
        parish = lookup.loc[parish_id]
        target_ministries = ministries.loc[ministries["parish_index"].eq(parish_id)].sort_values("ministry_key")
        peers = matches.loc[matches["target_parish_index"].eq(parish_id)].sort_values(["rank", "peer_parish_index"])
        target_gaps = gaps.loc[gaps["target_parish_index"].eq(parish_id)]
        if len(peers) != int(target["n_matches"]) or peers["peer_parish_index"].duplicated().any():
            raise ValueError(f"Inconsistent or duplicate neighbor records for target {parish_id}")
        if peers["rank"].tolist() != list(range(1, len(peers) + 1)) or not peers["distance"].is_monotonic_increasing:
            raise ValueError(f"Invalid neighbor ranks/distances for target {parish_id}")
        assignment = {
            "complete": "complete features", "provisional_imputed": "provisional (missing features imputed)",
            "unassigned_no_features": "unassigned (no features available)",
        }.get(_text(target["assignment_quality"]), _text(target["assignment_quality"]))
        feature_names = {
            "county_population": "county population", "county_median_age": "county median age",
            "n_social_links": "social-link count", "attached_school": "school affiliation",
        }
        imputed = ", ".join(feature_names.get(name.strip(), name.strip()) for name in _text(target.get("imputed_features", "")).split(",") if name.strip())
        county_method = {"address": "street address", "city_state": "city/state (approximate)", "zip": "ZIP (approximate)"}.get(_text(parish.get("county_lookup_method", "")), "unavailable")
        lines = [
            f"# {_md(parish['name'])}", "", "## Parish information", "",
            f"- Diocese: {_md(parish['diocese'])}",
            f"- Address: {_md(parish.get('address', '')) or 'Unavailable'}",
            f"- Website: {_link(parish.get('site_url', ''), parish.get('site_url', '')) or 'Unavailable'}",
            f"- Profile: {_description(parish)}",
            f"- Cluster: {_number(target['cluster'])}; assignment: {_md(assignment)}.",
            f"- Imputed features: {_md(imputed) or 'None'}.",
            f"- County lookup: {_md(county_method)}.",
            "", "## Current parish ministries", "", _coverage_text(coverage_lookup.loc[parish_id]), "",
        ]
        if target_ministries.empty:
            lines += ["No ministry labels are recorded for this parish. The parish may have ministries that were not identified in the available data.", ""]
        else:
            lines += ["| Ministry | Category | Source pages |", "| --- | --- | --- |"]
            for ministry in target_ministries.to_dict("records"):
                urls = json.loads(ministry["source_urls"])
                evidence = ", ".join(_link(f"Page {i + 1}", url) for i, url in enumerate(urls[:3]))
                if len(urls) > 3:
                    evidence += f" (+{len(urls) - 3} more)"
                lines.append(f"| {_md(ministry['ministry'])} | {_md(ministry['category']) or 'Uncategorized'} | {evidence or 'Unavailable'} |")
            lines.append("")
        lines += ["## Neighbor descriptions", "", "Peers are ranked within the assigned cluster by distance in standardized/log-transformed feature space. Distance is not geographic mileage. School affiliation is an AI prediction.", ""]
        if _text(target["assignment_quality"]) == "provisional_imputed":
            lines += ["**Provisional comparison:** missing target features were imputed for clustering and ranking. Review these matches before using them for planning.", ""]
        if peers.empty:
            lines += [f"No neighbors were returned. Match status: {_md(target['match_status'])}.", ""]
        for peer in peers.to_dict("records"):
            peer_id = peer["peer_parish_index"]
            neighbor = lookup.loc[peer_id]
            if peer_id == parish_id or _number(neighbor["cluster"]) != _number(target["cluster"]):
                raise ValueError(f"Invalid within-cluster neighbor for target {parish_id}")
            lines += [
                f"### {int(peer['rank'])}. {_link(neighbor['name'], neighbor.get('site_url', ''))}", "",
                f"{_md(neighbor['diocese'])}. {_md(neighbor.get('address', ''))}.", "",
                _description(neighbor), "",
                f"Distance: {_number(peer['distance'], 4)}. " + _coverage_text(coverage_lookup.loc[peer_id]), "",
            ]
        lines += ["## Neighbor ministries not currently identified for this parish", "", "These labels appear in neighbor extraction results but not in the target's recorded labels. They are comparison leads, not verified gaps or recommendations. Missing/partial extraction and differing ministry names can create apparent differences.", ""]
        candidates = []
        for key, group in target_gaps.groupby("ministry_key"):
            candidates.append((key, group["peer_parish_index"].nunique(), group.sort_values(["rank", "peer_parish_index"])))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        if not candidates:
            lines += ["No additional ministry labels were identified among the available neighbors.", ""]
        else:
            lines += ["| Ministry | Category | Neighbors | Neighbor evidence |", "| --- | --- | ---: | --- |"]
            for _, support, group in candidates:
                first = group.iloc[0]
                evidence = []
                for _, row in group.drop_duplicates("peer_parish_index").iterrows():
                    urls = json.loads(row["source_urls"])
                    evidence.append(_link(row["peer_name"], urls[0] if urls else row.get("peer_site_url", "")))
                categories = "; ".join(sorted(set(group["category"]) - {""}))
                lines.append(f"| {_md(first['ministry'])} | {_md(categories) or 'Uncategorized'} | {support} | {', '.join(evidence)} |")
            lines.append("")
        lines += ["## Reading this report", "", "Ministries are deduplicated by case, punctuation, and spacing; synonyms remain separate unless their recorded names match. Evidence comes from AI/rule outputs, which can be provisional or incomplete. All ministry source URLs and per-neighbor differences are retained in `data/06_generate_reports/`. County demographics describe 2020 conditions; web-derived features describe the cached information used by the pipeline.", ""]
        filename = _filename({**parish.to_dict(), "parish_index": parish_id})
        (report_dir / filename).write_text("\n".join(lines), encoding="utf-8")
        manifest.append({
            "target_parish_index": parish_id, "target_name": parish["name"], "target_diocese": target_diocese,
            "report_path": (report_dir / filename).as_posix(), "n_current_ministries": len(target_ministries),
            "n_neighbors": len(peers), "n_additional_ministry_labels": len(candidates),
            "ministry_coverage_status": coverage_lookup.loc[parish_id, "coverage_status"],
            "assignment_quality": target["assignment_quality"],
        })
    manifest_df = pd.DataFrame(manifest)
    lines = [f"# Parish comparison reports: {_md(target_diocese)}", "", "Each report combines parish details, recorded ministries, ranked neighbor descriptions, and neighbor ministry labels not currently identified for the target. Apparent ministry differences require review of source evidence and extraction coverage.", "", "| Parish report | Current ministry labels | Neighbors | Additional labels | Coverage |", "| --- | ---: | ---: | ---: | --- |"]
    for row in manifest:
        lines.append(f"| [{_md(row['target_name'])}]({Path(row['report_path']).name}) | {row['n_current_ministries']} | {row['n_neighbors']} | {row['n_additional_ministry_labels']} | {_md(row['ministry_coverage_status'])} |")
    (report_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_df

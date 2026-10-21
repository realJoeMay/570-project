"""Prepare parish features, compare KMeans fits, and retrieve cluster peers."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


FEATURES = ["county_population", "county_median_age", "n_social_links", "attached_school"]
MODEL_FEATURES = ["log_county_population", "county_median_age", "log_n_social_links", "attached_school"]


def prepare_clustering_data(parishes):
    """Fit preprocessing only on complete cases; transform partial rows too.

    Log-transform population/social counts and standardize the three continuous
    variables. Encode school as 0/1 without standardizing its minority class.
    Partial records use training medians/mode, with missingness retained for review.
    """
    if parishes["parish_index"].isna().any() or not parishes["parish_index"].is_unique:
        raise ValueError("parish_index must be unique and nonmissing")
    numeric = parishes[FEATURES].astype("Float64").astype(float)
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite feature values")
    for feature in ["county_population", "n_social_links"]:
        if numeric[feature].dropna().lt(0).any():
            raise ValueError(f"Negative values in {feature}")
    if not numeric["attached_school"].dropna().isin([0, 1]).all():
        raise ValueError("attached_school must be boolean or 0/1")
    if not numeric["county_median_age"].dropna().between(0, 120).all():
        raise ValueError("Invalid county median age")
    complete = numeric.notna().all(axis=1)
    if complete.sum() < 3:
        raise ValueError("At least three complete parish records are required")
    transformed = pd.DataFrame({
        "log_county_population": np.log1p(numeric["county_population"]),
        "county_median_age": numeric["county_median_age"],
        "log_n_social_links": np.log1p(numeric["n_social_links"]),
        "attached_school": numeric["attached_school"],
    }, index=parishes.index)
    preprocessor = ColumnTransformer([
        ("continuous", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), MODEL_FEATURES[:3]),
        ("school", SimpleImputer(strategy="most_frequent"), ["attached_school"]),
    ])
    preprocessor.fit(transformed.loc[complete])
    matrix = preprocessor.transform(transformed)
    quality = pd.DataFrame({
        "parish_index": parishes["parish_index"],
        "is_reference": complete,
        "n_observed_features": numeric.notna().sum(axis=1),
        "imputed_features": numeric.isna().apply(lambda row: ", ".join(row.index[row]), axis=1),
    }, index=parishes.index)
    return transformed, matrix, complete, quality, preprocessor


def evaluate_kmeans(matrix, k_values, seeds=(42, 7, 21, 83, 101), n_init=20):
    """Compare scores, cluster sizes, and agreement across random initializations."""
    scores, models = [], {}
    unique_rows = len(np.unique(matrix, axis=0))
    with threadpool_limits(limits=1):
        for k in k_values:
            if k < 2 or k >= len(matrix) or k > unique_rows:
                continue
            fits = [KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(matrix) for seed in seeds]
            model = fits[0]
            labels = model.labels_
            if len(np.unique(labels)) != k:
                continue
            sizes = np.bincount(labels, minlength=k)
            agreement = [adjusted_rand_score(labels, other.labels_) for other in fits[1:]]
            scores.append({
                "k": k, "silhouette": silhouette_score(matrix, labels),
                "davies_bouldin": davies_bouldin_score(matrix, labels),
                "inertia": model.inertia_, "min_cluster_size": sizes.min(),
                "max_cluster_size": sizes.max(),
                "mean_seed_ari": float(np.mean(agreement)) if agreement else 1.0,
                "min_seed_ari": float(np.min(agreement)) if agreement else 1.0,
            })
            models[k] = model
    if not scores:
        raise ValueError("No valid cluster counts for the available distinct records")
    return pd.DataFrame(scores), models


def cluster_profiles(parishes, labels):
    """Summarize original-scale features and geographic concentration."""
    labeled = parishes.copy()
    labeled["cluster"] = labels
    rows = []
    for cluster, group in labeled.groupby("cluster", sort=True):
        county_counts = group["county_fips"].value_counts()
        diocese_counts = group["diocese"].value_counts()
        row = {
            "cluster": cluster, "n_reference_parishes": len(group),
            "n_counties": group["county_fips"].nunique(), "n_dioceses": group["diocese"].nunique(),
            "largest_county_share": county_counts.iloc[0] / len(group),
            "largest_diocese": diocese_counts.index[0],
            "largest_diocese_share": diocese_counts.iloc[0] / len(group),
            "school_share": group["attached_school"].astype(float).mean(),
        }
        for feature in FEATURES[:3]:
            for statistic in ["median", "min", "max"]:
                row[f"{feature}_{statistic}"] = getattr(group[feature], statistic)()
        rows.append(row)
    return pd.DataFrame(rows)


def select_cluster_count(metrics, preferred_k=None):
    """Select from current diagnostics rather than a previously reviewed fixed k.

    Peer-pool size and stability are feasibility checks. Near-best silhouette
    is a preference: if no feasible candidate passes it, use the best feasible
    silhouette and explain the fallback. A reviewed override may also choose
    outside this soft shortlist while retaining size/stability requirements.
    """
    feasible = metrics.loc[metrics["peer_pool_ok"] & metrics["stable"]]
    if feasible.empty:
        raise ValueError(
            "No tested cluster count meets the peer-pool size and stability requirements. "
            "Inspect 3_cluster_metrics.csv and candidate profiles before changing the settings."
        )
    if preferred_k is not None:
        if type(preferred_k) is not int or preferred_k not in feasible["k"].values:
            raise ValueError("The reviewed cluster count must be a tested integer with sufficient peers and stability")
        selected = feasible.set_index("k").loc[preferred_k]
        suffix = " It is outside the near-best silhouette shortlist." if not selected["near_best_silhouette"] else ""
        return preferred_k, f"Reviewed override k={preferred_k}; peer-pool size and stability requirements pass.{suffix}"
    shortlist = feasible.loc[feasible["near_best_silhouette"]]
    candidates = shortlist if not shortlist.empty else feasible
    best = candidates.sort_values(
        ["silhouette", "davies_bouldin", "k"], ascending=[False, True, True], kind="stable",
    ).iloc[0]
    k = int(best["k"])
    reason = (
        f"Selected k={k} from current feasible candidates by highest silhouette; "
        "ties favor lower Davies-Bouldin, then fewer clusters. "
    )
    reason += (
        "The selected count is in the near-best silhouette shortlist."
        if not shortlist.empty else
        "No feasible count met the near-best silhouette preference, so the best feasible count was used."
    )
    return k, reason


def within_cluster_neighbors(parishes, matrix, *, target_diocese, top_n=10):
    """Rank complete reference peers within each target's assigned cluster.

    Distances use the same transformed/imputed model space as KMeans. Preserve
    every target in the summary, including targets without an assignable cluster.
    Ties are resolved by parish_index; each target itself is excluded.
    """
    if type(top_n) is not int or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if len(parishes) != len(matrix) or not parishes["parish_index"].is_unique:
        raise ValueError("Expected one aligned feature vector per unique parish")
    records, summaries = [], []
    detail_fields = [
        "parish_index", "name", "diocese", "site_url", "address", "county", "county_fips",
        "county_lookup_method", *FEATURES,
    ]
    targets = np.flatnonzero(parishes["diocese"].eq(target_diocese).to_numpy())
    for position in targets:
        target = parishes.iloc[position]
        cluster = target["cluster"]
        summary = {
            "target_parish_index": target["parish_index"], "target_name": target["name"],
            "cluster": cluster, "assignment_quality": target["assignment_quality"],
            "imputed_features": target["imputed_features"], "n_available_peers": 0,
            "n_matches": 0, "match_status": "unassigned",
        }
        if pd.isna(cluster):
            summaries.append(summary)
            continue
        eligible = (
            parishes["cluster"].eq(cluster).fillna(False)
            & parishes["is_reference"]
            & parishes["parish_index"].ne(target["parish_index"])
        )
        positions = np.flatnonzero(eligible.to_numpy(dtype=bool))
        distances = np.linalg.norm(matrix[positions] - matrix[position], axis=1)
        ranked = pd.DataFrame({
            "position": positions, "distance": distances,
            "peer_parish_index": parishes.iloc[positions]["parish_index"].to_numpy(),
        }).sort_values(["distance", "peer_parish_index"], kind="stable").head(top_n)
        summary.update(
            n_available_peers=len(positions), n_matches=len(ranked),
            match_status="ok" if len(ranked) == top_n else "fewer_than_requested",
        )
        summaries.append(summary)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            peer = parishes.iloc[row.position]
            record = {
                "cluster": int(cluster), "rank": rank, "distance": row.distance,
                "target_assignment_quality": target["assignment_quality"],
                "target_imputed_features": target["imputed_features"],
                "same_county": bool(target["county_fips"] == peer["county_fips"]) if pd.notna(target["county_fips"]) and pd.notna(peer["county_fips"]) else False,
            }
            record.update({f"target_{field}": target[field] for field in detail_fields if field in parishes})
            record.update({f"peer_{field}": peer[field] for field in detail_fields if field in parishes})
            records.append(record)
    base_columns = ["cluster", "rank", "distance", "target_assignment_quality", "target_imputed_features", "same_county"]
    columns = base_columns + [f"{prefix}_{field}" for prefix in ["target", "peer"] for field in detail_fields if field in parishes]
    summary_columns = [
        "target_parish_index", "target_name", "cluster", "assignment_quality", "imputed_features",
        "n_available_peers", "n_matches", "match_status",
    ]
    return pd.DataFrame(records, columns=columns), pd.DataFrame(summaries, columns=summary_columns)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Survival analysis for HyPERSTAC IMC patch representations.

This adapts the original repository's survival workflow to the new IMC AnnData
output. It maps ROI-level patches to clinical cases, aggregates patch cluster
frequencies per clinical case, and fits cross-validated Cox proportional hazards
models.

Use --cluster-col-search leiden to repeat the analysis for every matching
adata.obs column, such as leiden, leiden_1.0, and leiden_0.9. In that mode,
each output filename is prefixed by the current cluster column name.

Expected clinical CSV:

    index: ROI name, unless --roi-col is supplied
    required columns:
        clinical case ID column, e.g. clinical_case
        survival duration column in days, e.g. post_survival_diagnosis_days
        event/censoring column, unless --assume-all-events is used

The event column should be 1/true for observed event/death and 0/false for
censored observations.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from contextlib import redirect_stdout
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import KFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run case-level survival analysis from IMC HyPERSTAC representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adata-path",
        type=Path,
        default=Path(r"D:\Programming\2024 - GBM Paper 2\imc_hyperstac_representations.h5ad"),
        help="Representation AnnData file from imc_hyperstac_pipeline.py.",
    )
    parser.add_argument(
        "--clinical-csv",
        type=Path,
        default=Path(r"D:\Programming\2024 - GBM Paper 2\Survival_diagnosis.csv"),
        help="CSV containing ROI-to-clinical-case and survival metadata.",
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path(r"D:\Programming\2024 - GBM Paper 2\Hyperstac"),
        help="Directory for outputs.")

    parser.add_argument(
        "--roi-col",
        type=str,
        default="ROI",
        help="Column containing ROI names. If omitted, the CSV index is treated as the ROI name.",
    )
    parser.add_argument(
        "--case-col",
        type=str,
        default="Case",
        help="Column containing clinical case / patient identifier.",
    )
    parser.add_argument(
        "--duration-col",
        type=str,
        default="Survival_diagnosis",
        help="Column containing survival time after diagnosis, in days.",
    )
    parser.add_argument(
        "--event-col",
        type=str,
        default='Event',
        help="Column containing event indicator: 1/event, 0/censored.",
    )
    parser.add_argument(
        "--assume-all-events",
        action="store_true",
        help=(
            "Use only if every case has an observed event. If set and --event-col "
            "is omitted, all cases are treated as event=1."
        ),
    )
    parser.add_argument(
        "--cluster-col",
        type=str,
        default="leiden",
        help="Patch cluster column in adata.obs. Computed if missing and --compute-clusters is set.",
    )
    parser.add_argument(
        "--cluster-col-search",
        type=str,
        default=None,
        help=(
            "If supplied, run the full survival analysis separately for every "
            "adata.obs column whose name contains this search term, e.g. 'leiden' "
            "matches 'leiden', 'leiden_1.0', and 'leiden_0.9'."
        ),
    )
    parser.add_argument(
        "--compute-clusters",
        action="store_true",
        help="Compute Scanpy neighbors/Leiden clusters if --cluster-col is missing.",
    )
    parser.add_argument("--cluster-resolution", type=float, default=1.0, help="Leiden resolution.")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Neighbors used for clustering.")
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of case-level cross-validation folds.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for folds.")
    parser.add_argument(
        "--penalizer",
        type=float,
        default=0.1,
        help="CoxPH lifelines penalizer. Increase if the model has convergence issues.",
    )
    parser.add_argument("--l1-ratio", type=float, default=0.0, help="Elastic-net L1 ratio for CoxPH.")
    parser.add_argument(
        "--min-patches-per-case",
        type=int,
        default=1,
        help="Drop clinical cases with fewer than this many retained patches.",
    )
    parser.add_argument(
        "--drop-clusters-below-frequency",
        type=float,
        default=0.0,
        help=(
            "Drop cluster-frequency features whose nonzero case frequency is below this fraction. "
            "For example, 0.05 drops clusters present in fewer than 5% of cases."
        ),
    )
    parser.add_argument(
        "--covariate-cols",
        type=str,
        default=None,
        help=(
            "Optional comma-separated clinical covariates to include in the Cox model. "
            "Categorical covariates are one-hot encoded."
        ),
    )
    parser.add_argument(
        "--write-clustered-adata",
        action="store_true",
        help="If clusters are computed, write a copy of the AnnData containing them.",
    )
    return parser.parse_args()


def parse_covariates(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_filename_prefix(value: str) -> str:
    prefix = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return prefix or "cluster"


def output_path(output_dir: Path, prefix: str, filename: str) -> Path:
    if prefix:
        return output_dir / f"{prefix}{filename}"
    return output_dir / filename


def find_cluster_columns(adata: ad.AnnData, args: argparse.Namespace) -> list[str]:
    if args.cluster_col_search is None:
        return [args.cluster_col]

    search = args.cluster_col_search.lower()
    matches = [
        column
        for column in adata.obs.columns
        if search in str(column).lower()
    ]
    matches = sorted(matches)
    if not matches:
        raise ValueError(
            f"No adata.obs columns matched --cluster-col-search '{args.cluster_col_search}'. "
            f"Available columns include: {list(adata.obs.columns[:20])}"
        )
    print(f"Matched cluster columns: {matches}")
    return matches


def read_clinical_table(args: argparse.Namespace) -> pd.DataFrame:
    if args.roi_col is None:
        clinical = pd.read_csv(args.clinical_csv, index_col=0)
        clinical = clinical.reset_index().rename(columns={clinical.index.name or "index": "roi"})
    else:
        clinical = pd.read_csv(args.clinical_csv)
        clinical = clinical.rename(columns={args.roi_col: "roi"})

    required = {"roi", args.case_col, args.duration_col}
    if args.event_col is not None:
        required.add(args.event_col)
    missing = sorted(required - set(clinical.columns))
    if missing:
        raise ValueError(f"Clinical CSV is missing required columns: {missing}")

    if args.event_col is None and not args.assume_all_events:
        raise ValueError(
            "Survival analysis needs an event/censoring column. Supply --event-col, "
            "or use --assume-all-events only if every case has had the event."
        )

    clinical = clinical.rename(
        columns={
            args.case_col: "case_id",
            args.duration_col: "duration_days",
        }
    )
    if args.event_col is not None:
        clinical = clinical.rename(columns={args.event_col: "event"})
    else:
        clinical["event"] = 1

    clinical["roi"] = clinical["roi"].astype(str)
    clinical["case_id"] = clinical["case_id"].astype(str)
    clinical["duration_days"] = pd.to_numeric(clinical["duration_days"], errors="coerce")
    clinical["event"] = clinical["event"].map(normalize_event_value)
    bad_events = sorted(set(clinical["event"].dropna()) - {0.0, 1.0})
    if bad_events:
        raise ValueError(f"Event column must encode 0/1 values after parsing; found {bad_events}")
    return clinical


def normalize_event_value(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "t", "yes", "y", "event", "dead", "death", "deceased"}:
            return 1.0
        if cleaned in {"0", "false", "f", "no", "n", "censored", "alive"}:
            return 0.0
    return float(value)


def ensure_clusters(
    adata: ad.AnnData,
    cluster_col: str,
    compute_clusters: bool,
    n_neighbors: int,
    resolution: float,
) -> ad.AnnData:
    if cluster_col in adata.obs:
        return adata
    if not compute_clusters:
        raise ValueError(
            f"'{cluster_col}' is not present in adata.obs. Re-run the representation "
            "pipeline with RUN_SCANPY=1, or pass --compute-clusters."
        )

    print(f"Computing clusters in adata.obs['{cluster_col}']")
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=n_neighbors)
    sc.tl.leiden(adata, resolution=resolution, key_added=cluster_col)
    return adata


def case_level_unique(clinical: pd.DataFrame, column: str) -> pd.Series:
    values = {}
    for case_id, group in clinical.groupby("case_id"):
        unique_values = group[column].dropna().unique()
        if len(unique_values) == 0:
            values[case_id] = np.nan
        elif len(unique_values) == 1:
            values[case_id] = unique_values[0]
        else:
            raise ValueError(
                f"Clinical column '{column}' has multiple values for case '{case_id}': "
                f"{list(unique_values)}"
            )
    return pd.Series(values, name=column)


def build_case_table(
    adata: ad.AnnData,
    clinical: pd.DataFrame,
    cluster_col: str,
    min_patches_per_case: int,
    drop_clusters_below_frequency: float,
    covariate_cols: list[str],
) -> pd.DataFrame:
    if "roi" not in adata.obs:
        raise ValueError("Expected adata.obs['roi']; this is produced by imc_hyperstac_pipeline.py.")

    patch_df = adata.obs[["roi", cluster_col]].copy()
    patch_df["roi"] = patch_df["roi"].astype(str)
    patch_df[cluster_col] = patch_df[cluster_col].astype(str)
    patch_df = patch_df.merge(clinical[["roi", "case_id"]], on="roi", how="inner")

    if patch_df.empty:
        raise ValueError("No patches matched the ROI names in the clinical CSV.")

    patch_counts = patch_df.groupby("case_id").size().rename("n_patches")
    case_cluster_counts = pd.crosstab(patch_df["case_id"], patch_df[cluster_col])
    case_cluster_freq = case_cluster_counts.div(case_cluster_counts.sum(axis=1), axis=0)
    case_cluster_freq.columns = [f"cluster_freq_{col}" for col in case_cluster_freq.columns]

    if drop_clusters_below_frequency > 0:
        prevalence = (case_cluster_freq > 0).mean(axis=0)
        case_cluster_freq = case_cluster_freq.loc[:, prevalence >= drop_clusters_below_frequency]

    case_table = case_cluster_freq.copy()
    case_table["n_patches"] = patch_counts
    case_table["duration_days"] = case_level_unique(clinical, "duration_days")
    case_table["event"] = case_level_unique(clinical, "event")

    if covariate_cols:
        covariates = build_covariate_table(clinical, covariate_cols)
        case_table = case_table.join(covariates, how="left")

    case_table = case_table[case_table["n_patches"] >= min_patches_per_case]
    case_table = case_table.dropna(axis=0)
    case_table = case_table[case_table["duration_days"] > 0]
    case_table["event"] = case_table["event"].astype(int)
    if case_table["event"].sum() == 0:
        raise ValueError("No observed events remain after filtering; Cox survival analysis cannot be fit.")
    if case_table["event"].sum() == len(case_table):
        print("Warning: all retained cases are marked as events; no censoring will be modeled.")
    return case_table


def build_covariate_table(clinical: pd.DataFrame, covariate_cols: list[str]) -> pd.DataFrame:
    missing = sorted(set(covariate_cols) - set(clinical.columns))
    if missing:
        raise ValueError(f"Clinical CSV is missing requested covariates: {missing}")

    covariates = pd.DataFrame(index=sorted(clinical["case_id"].unique()))
    covariates.index.name = "case_id"
    for column in covariate_cols:
        covariates[column] = case_level_unique(clinical, column)

    numeric_columns = []
    categorical_columns = []
    for column in covariates.columns:
        converted = pd.to_numeric(covariates[column], errors="coerce")
        if converted.notna().all():
            covariates[column] = converted
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    if categorical_columns:
        encoded = pd.get_dummies(covariates[categorical_columns], columns=categorical_columns, drop_first=True)
        covariates = pd.concat([covariates[numeric_columns], encoded], axis=1)

    return covariates


def fit_cox(train_df: pd.DataFrame, feature_cols: list[str], args: argparse.Namespace) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=args.penalizer, l1_ratio=args.l1_ratio)
    cph.fit(
        train_df[[*feature_cols, "duration_days", "event"]],
        duration_col="duration_days",
        event_col="event",
        robust=True,
    )
    return cph


def safe_c_index(cph: CoxPHFitter, data: pd.DataFrame, feature_cols: list[str]) -> float:
    try:
        return float(
            cph.score(
                data[[*feature_cols, "duration_days", "event"]],
                scoring_method="concordance_index",
            )
        )
    except Exception as exc:
        print(f"Warning: could not calculate C-index for {len(data)} cases: {exc}")
        return float("nan")


def run_cross_validation(case_table: pd.DataFrame, feature_cols: list[str], args: argparse.Namespace):
    if len(case_table) < 2:
        raise ValueError("At least two clinical cases are required for survival analysis.")

    n_splits = min(args.num_folds, len(case_table))
    if n_splits < args.num_folds:
        print(f"Reducing --num-folds to {n_splits} because only {len(case_table)} cases are available.")

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    fold_summaries = []
    predictions = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(case_table), start=1):
        train_df = case_table.iloc[train_idx].copy()
        test_df = case_table.iloc[test_idx].copy()

        cph = fit_cox(train_df, feature_cols, args)
        train_score = safe_c_index(cph, train_df, feature_cols)
        test_score = safe_c_index(cph, test_df, feature_cols)

        train_risk = cph.predict_partial_hazard(train_df[feature_cols])
        test_risk = cph.predict_partial_hazard(test_df[feature_cols])
        risk_threshold = float(train_risk.median())

        fold_pred = pd.DataFrame(
            {
                "case_id": test_df.index,
                "fold": fold,
                "duration_days": test_df["duration_days"].values,
                "event": test_df["event"].values,
                "risk_score": test_risk.values.reshape(-1),
            }
        )
        fold_pred["risk_group"] = np.where(fold_pred["risk_score"] >= risk_threshold, "high", "low")
        predictions.append(fold_pred)

        fold_summaries.append(
            {
                "fold": fold,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "train_c_index": train_score,
                "test_c_index": test_score,
                "risk_threshold": risk_threshold,
            }
        )
        print(f"Fold {fold}: train C-index={train_score:.3f}, test C-index={test_score:.3f}")

    return pd.DataFrame(fold_summaries), pd.concat(predictions, axis=0, ignore_index=True)


def plot_km(predictions: pd.DataFrame, output_path: Path) -> dict[str, float]:
    high = predictions[predictions["risk_group"] == "high"]
    low = predictions[predictions["risk_group"] == "low"]

    stats = {
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "logrank_p": np.nan,
    }

    plt.figure(figsize=(5.5, 4.5))
    kmf = KaplanMeierFitter()
    for group_name, group_df in [("low", low), ("high", high)]:
        if group_df.empty:
            continue
        kmf.fit(
            durations=group_df["duration_days"],
            event_observed=group_df["event"],
            label=f"{group_name} risk",
        )
        kmf.plot_survival_function(ci_show=True)

    if not high.empty and not low.empty:
        result = logrank_test(
            low["duration_days"],
            high["duration_days"],
            event_observed_A=low["event"],
            event_observed_B=high["event"],
        )
        stats["logrank_p"] = float(result.p_value)
        plt.title(f"Cross-validated risk groups, log-rank p={result.p_value:.3g}")
    else:
        plt.title("Cross-validated risk groups")

    plt.xlabel("Days after diagnosis")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return stats


def plot_cluster_effects(cph: CoxPHFitter, output_path: Path, title: str) -> None:
    summary = cph.summary.copy()
    cluster_summary = summary.loc[
        [idx for idx in summary.index if str(idx).startswith("cluster_freq_")]
    ].copy()

    if cluster_summary.empty:
        print("Warning: no cluster_freq_* terms found; skipping cluster effect plot.")
        return

    if "exp(coef)" not in cluster_summary.columns:
        cluster_summary["exp(coef)"] = np.exp(cluster_summary["coef"])
    if "exp(coef) lower 95%" not in cluster_summary.columns:
        cluster_summary["exp(coef) lower 95%"] = np.exp(cluster_summary["coef lower 95%"])
    if "exp(coef) upper 95%" not in cluster_summary.columns:
        cluster_summary["exp(coef) upper 95%"] = np.exp(cluster_summary["coef upper 95%"])

    cluster_summary = cluster_summary.sort_values("coef", ascending=True)
    labels = [str(idx).replace("cluster_freq_", "cluster ") for idx in cluster_summary.index]
    hazard_ratio = cluster_summary["exp(coef)"].to_numpy(dtype=float)
    lower = cluster_summary["exp(coef) lower 95%"].to_numpy(dtype=float)
    upper = cluster_summary["exp(coef) upper 95%"].to_numpy(dtype=float)
    p_values = cluster_summary["p"].to_numpy(dtype=float) if "p" in cluster_summary.columns else None

    y = np.arange(len(cluster_summary))
    xerr = np.vstack([hazard_ratio - lower, upper - hazard_ratio])
    colors = np.where(hazard_ratio >= 1.0, "#b2182b", "#2166ac")

    height = max(4.0, 0.32 * len(cluster_summary) + 1.8)
    plt.figure(figsize=(7.5, height))
    plt.errorbar(
        hazard_ratio,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#666666",
        elinewidth=1.0,
        capsize=2,
        zorder=1,
    )
    plt.scatter(hazard_ratio, y, c=colors, s=34, zorder=2)
    plt.axvline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    plt.xscale("log")
    plt.yticks(y, labels)
    plt.xlabel("Hazard ratio per unit increase in case-level cluster frequency")
    plt.ylabel("Patch cluster")
    plt.title(title)

    if p_values is not None:
        x_text = np.nanmax(upper) * 1.15
        for y_pos, p_value in zip(y, p_values):
            plt.text(x_text, y_pos, f"p={p_value:.3g}", va="center", fontsize=8)
        plt.xlim(left=max(np.nanmin(lower) / 1.4, 1e-3), right=x_text * 1.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_final_model(
    case_table: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    filename_prefix: str = "",
    cluster_col: str = "",
) -> CoxPHFitter:
    cph = fit_cox(case_table, feature_cols, args)

    with open(output_path(args.output_dir, filename_prefix, "final_cox_model.pkl"), "wb") as f:
        pickle.dump(cph, f)

    with open(output_path(args.output_dir, filename_prefix, "final_cox_model_summary.txt"), "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            cph.print_summary(style="ascii")

    cph.summary.to_csv(output_path(args.output_dir, filename_prefix, "final_cox_model_coefficients.csv"))
    plot_cluster_effects(
        cph,
        output_path(args.output_dir, filename_prefix, "cluster_effects_forest_plot.png"),
        title=f"Cluster effects on outcome ({cluster_col or 'cluster model'})",
    )
    return cph


def run_analysis_for_cluster(
    args: argparse.Namespace,
    adata: ad.AnnData,
    clinical: pd.DataFrame,
    cluster_col: str,
    filename_prefix: str,
) -> dict[str, object]:
    print(f"Running survival analysis for adata.obs['{cluster_col}']")
    covariate_cols = parse_covariates(args.covariate_cols)
    case_table = build_case_table(
        adata=adata,
        clinical=clinical,
        cluster_col=cluster_col,
        min_patches_per_case=args.min_patches_per_case,
        drop_clusters_below_frequency=args.drop_clusters_below_frequency,
        covariate_cols=covariate_cols,
    )

    metadata_cols = {"duration_days", "event", "n_patches"}
    feature_cols = [col for col in case_table.columns if col not in metadata_cols]
    if not feature_cols:
        raise ValueError("No survival model features remain after filtering.")
    case_table[feature_cols] = case_table[feature_cols].apply(pd.to_numeric, errors="raise").astype(float)

    case_table.to_csv(output_path(args.output_dir, filename_prefix, "case_survival_features.csv"))
    with open(output_path(args.output_dir, filename_prefix, "survival_run_config.json"), "w", encoding="utf-8") as f:
        config = vars(args).copy()
        config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
        config["cluster_col"] = cluster_col
        config["filename_prefix"] = filename_prefix
        config["n_cases"] = int(len(case_table))
        config["n_features"] = int(len(feature_cols))
        json.dump(config, f, indent=2)

    print(f"Prepared {len(case_table)} cases with {len(feature_cols)} model features")
    fold_summary, predictions = run_cross_validation(case_table, feature_cols, args)
    fold_summary.to_csv(output_path(args.output_dir, filename_prefix, "cv_summary.csv"), index=False)
    predictions.to_csv(output_path(args.output_dir, filename_prefix, "cv_risk_predictions.csv"), index=False)

    km_stats = plot_km(
        predictions,
        output_path(args.output_dir, filename_prefix, "km_cross_validated_risk_groups.png"),
    )
    pd.DataFrame([km_stats]).to_csv(
        output_path(args.output_dir, filename_prefix, "km_cross_validated_risk_groups_stats.csv"),
        index=False,
    )

    cph = save_final_model(
        case_table,
        feature_cols,
        args,
        filename_prefix=filename_prefix,
        cluster_col=cluster_col,
    )
    mean_test_c = float(fold_summary["test_c_index"].mean())
    std_test_c = float(fold_summary["test_c_index"].std(ddof=0))
    print(
        f"{cluster_col}: mean test C-index="
        f"{mean_test_c:.3f} +/- {std_test_c:.3f}"
    )
    print(f"Final model concordance on all cases: {cph.concordance_index_:.3f}")
    return {
        "cluster_col": cluster_col,
        "filename_prefix": filename_prefix,
        "n_cases": int(len(case_table)),
        "n_features": int(len(feature_cols)),
        "mean_test_c_index": mean_test_c,
        "std_test_c_index": std_test_c,
        "final_model_c_index": float(cph.concordance_index_),
        **km_stats,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clinical = read_clinical_table(args)
    adata = ad.read_h5ad(args.adata_path)
    cluster_cols = find_cluster_columns(adata, args)
    multi_cluster_mode = args.cluster_col_search is not None

    if multi_cluster_mode and args.compute_clusters:
        print("Ignoring --compute-clusters because --cluster-col-search uses existing adata.obs columns.")

    if not multi_cluster_mode:
        adata = ensure_clusters(
            adata,
            cluster_col=cluster_cols[0],
            compute_clusters=args.compute_clusters,
            n_neighbors=args.n_neighbors,
            resolution=args.cluster_resolution,
        )
    else:
        missing = [column for column in cluster_cols if column not in adata.obs]
        if missing:
            raise ValueError(f"Matched cluster columns were not found in adata.obs: {missing}")

    summaries = []
    for cluster_col in cluster_cols:
        filename_prefix = f"{safe_filename_prefix(cluster_col)}__" if multi_cluster_mode else ""
        if args.write_clustered_adata:
            adata.write_h5ad(
                output_path(args.output_dir, filename_prefix, "representations_with_survival_clusters.h5ad")
            )
        summaries.append(
            run_analysis_for_cluster(
                args=args,
                adata=adata,
                clinical=clinical,
                cluster_col=cluster_col,
                filename_prefix=filename_prefix,
            )
        )

    if multi_cluster_mode:
        pd.DataFrame(summaries).to_csv(args.output_dir / "all_cluster_survival_summary.csv", index=False)


if __name__ == "__main__":
    main()

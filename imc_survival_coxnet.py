#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coxnet survival analysis for HyPERSTAC IMC patch representations.

This script builds case-level features from patch clusters, adds configured
clinical covariates, ranks features with univariate Cox models, fits an
elastic-net penalised Cox model with scikit-survival Coxnet, and writes standard
Cox PH summaries/plots for the selected features.

Edit the CONFIG section below for your dataset, or override values with CLI
arguments.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
except ImportError as exc:  # pragma: no cover - gives a clearer user-facing failure.
    raise ImportError(
        "scikit-survival is required for this script. Install it with conda-forge, "
        "for example: conda install -c conda-forge scikit-survival"
    ) from exc


# ---------------------------------------------------------------------------
# CONFIG: edit these defaults for your local/cluster dataset.
# ---------------------------------------------------------------------------

DEFAULT_ADATA_PATH = Path(r"D:\Programming\2024 - GBM Paper 2\imc_hyperstac_representations.h5ad")
DEFAULT_CLINICAL_CSV = Path(r"D:\Programming\2024 - GBM Paper 2\Survival_diagnosis.csv")
DEFAULT_OUTPUT_DIR = Path(r"D:\Programming\2024 - GBM Paper 2\Hyperstac_CoxNet")

# Clinical CSV columns. If ROI_COL is None, the CSV index is treated as ROI.
ROI_COL = 'ROI'
CASE_COL = "Case"
DURATION_COL = "Survival_diagnosis"
EVENT_COL = "Event"

# Clinical covariates to include alongside image-derived cluster frequencies.
#CLINICAL_COVARIATE_COLS = ["Age_at_diagnosis", "Sex"]
CLINICAL_COVARIATE_COLS= []

# Patch cluster column in adata.obs. Set CLUSTER_COL_SEARCH to e.g. "leiden" to
# run one analysis per matching adata.obs column, each in a subfolder.
CLUSTER_COL = "leiden"
CLUSTER_COL_SEARCH = "leiden"

# Feature/model settings.
MIN_PATCHES_PER_CASE = 1
DROP_CLUSTERS_BELOW_CASE_FREQUENCY = 0.0
FEATURE_SELECTION_TOP_N = 25
STANDARD_COX_MAX_FEATURES = 20
COXNET_L1_RATIO = 0.5
COXNET_N_ALPHAS = 100
COXNET_ALPHA_MIN_RATIO = 0.01
COXNET_CV_FOLDS = 5
STANDARD_COX_PENALIZER = 0.1
SEED = 1


@dataclass
class Config:
    adata_path: Path = DEFAULT_ADATA_PATH
    clinical_csv: Path = DEFAULT_CLINICAL_CSV
    output_dir: Path = DEFAULT_OUTPUT_DIR
    roi_col: str | None = ROI_COL
    case_col: str = CASE_COL
    duration_col: str = DURATION_COL
    event_col: str = EVENT_COL
    clinical_covariate_cols: tuple[str, ...] = tuple(CLINICAL_COVARIATE_COLS)
    cluster_col: str = CLUSTER_COL
    cluster_col_search: str | None = CLUSTER_COL_SEARCH
    min_patches_per_case: int = MIN_PATCHES_PER_CASE
    drop_clusters_below_case_frequency: float = DROP_CLUSTERS_BELOW_CASE_FREQUENCY
    feature_selection_top_n: int = FEATURE_SELECTION_TOP_N
    standard_cox_max_features: int = STANDARD_COX_MAX_FEATURES
    coxnet_l1_ratio: float = COXNET_L1_RATIO
    coxnet_n_alphas: int = COXNET_N_ALPHAS
    coxnet_alpha_min_ratio: float = COXNET_ALPHA_MIN_RATIO
    coxnet_cv_folds: int = COXNET_CV_FOLDS
    standard_cox_penalizer: float = STANDARD_COX_PENALIZER
    seed: int = SEED


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run Coxnet survival analysis from IMC HyPERSTAC representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--adata-path", type=Path, default=DEFAULT_ADATA_PATH)
    parser.add_argument("--clinical-csv", type=Path, default=DEFAULT_CLINICAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roi-col", type=str, default=ROI_COL)
    parser.add_argument("--case-col", type=str, default=CASE_COL)
    parser.add_argument("--duration-col", type=str, default=DURATION_COL)
    parser.add_argument("--event-col", type=str, default=EVENT_COL)
    parser.add_argument(
        "--clinical-covariate-cols",
        type=str,
        default=",".join(CLINICAL_COVARIATE_COLS),
        help="Comma-separated clinical covariates, e.g. Age,Sex.",
    )
    parser.add_argument("--cluster-col", type=str, default=CLUSTER_COL)
    parser.add_argument(
        "--cluster-col-search",
        type=str,
        default=CLUSTER_COL_SEARCH,
        help="Run one Coxnet analysis for every adata.obs column containing this term.",
    )
    parser.add_argument("--min-patches-per-case", type=int, default=MIN_PATCHES_PER_CASE)
    parser.add_argument(
        "--drop-clusters-below-case-frequency",
        type=float,
        default=DROP_CLUSTERS_BELOW_CASE_FREQUENCY,
    )
    parser.add_argument("--feature-selection-top-n", type=int, default=FEATURE_SELECTION_TOP_N)
    parser.add_argument("--standard-cox-max-features", type=int, default=STANDARD_COX_MAX_FEATURES)
    parser.add_argument("--coxnet-l1-ratio", type=float, default=COXNET_L1_RATIO)
    parser.add_argument("--coxnet-n-alphas", type=int, default=COXNET_N_ALPHAS)
    parser.add_argument("--coxnet-alpha-min-ratio", type=float, default=COXNET_ALPHA_MIN_RATIO)
    parser.add_argument("--coxnet-cv-folds", type=int, default=COXNET_CV_FOLDS)
    parser.add_argument("--standard-cox-penalizer", type=float, default=STANDARD_COX_PENALIZER)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    covariates = tuple(
        item.strip()
        for item in args.clinical_covariate_cols.split(",")
        if item.strip()
    )
    roi_col = args.roi_col if args.roi_col not in {"", "None", "none"} else None

    return Config(
        adata_path=args.adata_path,
        clinical_csv=args.clinical_csv,
        output_dir=args.output_dir,
        roi_col=roi_col,
        case_col=args.case_col,
        duration_col=args.duration_col,
        event_col=args.event_col,
        clinical_covariate_cols=covariates,
        cluster_col=args.cluster_col,
        cluster_col_search=args.cluster_col_search,
        min_patches_per_case=args.min_patches_per_case,
        drop_clusters_below_case_frequency=args.drop_clusters_below_case_frequency,
        feature_selection_top_n=args.feature_selection_top_n,
        standard_cox_max_features=args.standard_cox_max_features,
        coxnet_l1_ratio=args.coxnet_l1_ratio,
        coxnet_n_alphas=args.coxnet_n_alphas,
        coxnet_alpha_min_ratio=args.coxnet_alpha_min_ratio,
        coxnet_cv_folds=args.coxnet_cv_folds,
        standard_cox_penalizer=args.standard_cox_penalizer,
        seed=args.seed,
    )


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


def read_clinical_table(config: Config) -> pd.DataFrame:
    """Read ROI-level clinical metadata and normalise core column names."""
    if config.roi_col is None:
        clinical = pd.read_csv(config.clinical_csv, index_col=0)
        clinical = clinical.reset_index().rename(columns={clinical.index.name or "index": "roi"})
    else:
        clinical = pd.read_csv(config.clinical_csv)
        clinical = clinical.rename(columns={config.roi_col: "roi"})

    required = {"roi", config.case_col, config.duration_col, config.event_col}
    required.update(config.clinical_covariate_cols)
    missing = sorted(required - set(clinical.columns))
    if missing:
        raise ValueError(f"Clinical CSV is missing required columns: {missing}")

    clinical = clinical.rename(
        columns={
            config.case_col: "case_id",
            config.duration_col: "duration_days",
            config.event_col: "event",
        }
    )
    clinical["roi"] = clinical["roi"].astype(str)
    clinical["case_id"] = clinical["case_id"].astype(str)
    clinical["duration_days"] = pd.to_numeric(clinical["duration_days"], errors="coerce")
    clinical["event"] = clinical["event"].map(normalize_event_value)

    invalid_events = sorted(set(clinical["event"].dropna()) - {0.0, 1.0})
    if invalid_events:
        raise ValueError(f"Event column must encode 0/1 after parsing; found {invalid_events}")
    return clinical


def case_level_unique(clinical: pd.DataFrame, column: str) -> pd.Series:
    """Collapse ROI-level clinical columns to one value per clinical case."""
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


def build_covariate_table(clinical: pd.DataFrame, covariate_cols: tuple[str, ...]) -> pd.DataFrame:
    """Build numeric case-level covariates, one-hot encoding categorical columns."""
    covariates = pd.DataFrame(index=sorted(clinical["case_id"].unique()))
    covariates.index.name = "case_id"
    for column in covariate_cols:
        covariates[column] = case_level_unique(clinical, column)

    numeric_cols = []
    categorical_cols = []
    for column in covariates.columns:
        converted = pd.to_numeric(covariates[column], errors="coerce")
        if converted.notna().all():
            covariates[column] = converted
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)

    if categorical_cols:
        encoded = pd.get_dummies(covariates[categorical_cols], drop_first=True, dtype=float)
        covariates = pd.concat([covariates[numeric_cols], encoded], axis=1)
    return covariates


def find_cluster_columns(adata: ad.AnnData, config: Config) -> list[str]:
    if config.cluster_col_search is None:
        if config.cluster_col not in adata.obs:
            raise ValueError(f"Cluster column '{config.cluster_col}' was not found in adata.obs.")
        return [config.cluster_col]

    search = config.cluster_col_search.lower()
    matches = sorted([col for col in adata.obs.columns if search in str(col).lower()])
    if not matches:
        raise ValueError(f"No adata.obs columns match search term '{config.cluster_col_search}'.")
    return matches


def build_case_features(
    adata: ad.AnnData,
    clinical: pd.DataFrame,
    cluster_col: str,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate patch clusters to one row per case and add clinical covariates."""
    if "roi" not in adata.obs:
        raise ValueError("Expected adata.obs['roi']; this is produced by imc_hyperstac_pipeline.py.")

    patch_df = adata.obs[["roi", cluster_col]].copy()
    patch_df["roi"] = patch_df["roi"].astype(str)
    patch_df[cluster_col] = patch_df[cluster_col].astype(str)
    patch_df = patch_df.merge(clinical[["roi", "case_id"]], on="roi", how="inner")
    if patch_df.empty:
        raise ValueError("No AnnData ROIs matched the clinical CSV ROI names.")

    patch_counts = patch_df.groupby("case_id").size().rename("n_patches")
    cluster_counts = pd.crosstab(patch_df["case_id"], patch_df[cluster_col])
    cluster_freq = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)
    cluster_freq.columns = [f"cluster_freq_{cluster}" for cluster in cluster_freq.columns]

    if config.drop_clusters_below_case_frequency > 0:
        prevalence = (cluster_freq > 0).mean(axis=0)
        cluster_freq = cluster_freq.loc[:, prevalence >= config.drop_clusters_below_case_frequency]

    covariates = build_covariate_table(clinical, config.clinical_covariate_cols)
    outcome = pd.DataFrame(index=cluster_freq.index)
    outcome["duration_days"] = case_level_unique(clinical, "duration_days")
    outcome["event"] = case_level_unique(clinical, "event")
    outcome["n_patches"] = patch_counts

    case_table = cluster_freq.join(covariates, how="left").join(outcome, how="left")
    case_table = case_table[case_table["n_patches"] >= config.min_patches_per_case]
    case_table = case_table.dropna(axis=0)
    case_table = case_table[case_table["duration_days"] > 0]
    case_table["event"] = case_table["event"].astype(bool)

    if case_table["event"].sum() == 0:
        raise ValueError("No observed events remain after filtering.")

    feature_cols = [col for col in case_table.columns if col not in {"duration_days", "event", "n_patches"}]
    X = case_table[feature_cols].apply(pd.to_numeric, errors="raise").astype(float)
    non_constant = X.columns[X.nunique(dropna=True) > 1]
    X = X[non_constant]
    case_table = pd.concat([X, case_table[["duration_days", "event", "n_patches"]]], axis=1)
    return X, case_table


def make_survival_array(case_table: pd.DataFrame):
    return Surv.from_arrays(
        event=case_table["event"].astype(bool).to_numpy(),
        time=case_table["duration_days"].astype(float).to_numpy(),
    )


def standardize_features(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=X.columns,
    )
    return X_scaled, scaler


def safe_c_index(event, duration, risk) -> float:
    try:
        return float(concordance_index_censored(event, duration, risk)[0])
    except Exception:
        return float("nan")


def rank_features_univariate(
    X: pd.DataFrame,
    y,
    cv_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Fit one univariate Cox model per feature and rank by CV C-index."""
    n_splits = min(cv_folds, len(X))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []

    for feature in X.columns:
        fold_scores = []
        for train_idx, test_idx in splitter.split(X):
            X_train = X.iloc[train_idx][[feature]]
            X_test = X.iloc[test_idx][[feature]]
            y_train = y[train_idx]
            y_test = y[test_idx]
            try:
                model = CoxPHSurvivalAnalysis()
                model.fit(X_train, y_train)
                risk = model.predict(X_test)
                score = safe_c_index(y_test["event"], y_test["time"], risk)
            except Exception:
                score = float("nan")
            fold_scores.append(score)

        try:
            full_model = CoxPHSurvivalAnalysis()
            full_model.fit(X[[feature]], y)
            full_coef = float(full_model.coef_[0])
        except Exception:
            full_coef = float("nan")

        rows.append(
            {
                "feature": feature,
                "mean_cv_c_index": float(np.nanmean(fold_scores)),
                "std_cv_c_index": float(np.nanstd(fold_scores)),
                "univariate_coef": full_coef,
                "n_valid_folds": int(np.isfinite(fold_scores).sum()),
            }
        )

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(["mean_cv_c_index", "n_valid_folds"], ascending=[False, False])
    return ranking


def fit_coxnet_cv(
    X_scaled: pd.DataFrame,
    y,
    config: Config,
) -> tuple[CoxnetSurvivalAnalysis, pd.DataFrame, float]:
    """Fit Coxnet over an alpha path and choose alpha by CV C-index."""
    base_model = CoxnetSurvivalAnalysis(
        l1_ratio=config.coxnet_l1_ratio,
        n_alphas=config.coxnet_n_alphas,
        alpha_min_ratio=config.coxnet_alpha_min_ratio,
        max_iter=100000,
    )
    base_model.fit(X_scaled, y)
    alphas = base_model.alphas_

    n_splits = min(config.coxnet_cv_folds, len(X_scaled))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    score_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_scaled), start=1):
        model = CoxnetSurvivalAnalysis(
            l1_ratio=config.coxnet_l1_ratio,
            alphas=alphas,
            max_iter=100000,
        )
        model.fit(X_scaled.iloc[train_idx], y[train_idx])
        for alpha in alphas:
            try:
                risk = model.predict(X_scaled.iloc[test_idx], alpha=alpha)
                score = safe_c_index(y[test_idx]["event"], y[test_idx]["time"], risk)
            except Exception:
                score = float("nan")
            score_rows.append({"fold": fold, "alpha": alpha, "c_index": score})

    cv_scores = pd.DataFrame(score_rows)
    alpha_summary = (
        cv_scores.groupby("alpha", as_index=False)
        .agg(mean_c_index=("c_index", "mean"), std_c_index=("c_index", "std"))
        .sort_values("mean_c_index", ascending=False)
    )
    best_alpha = float(alpha_summary.iloc[0]["alpha"])

    final_model = CoxnetSurvivalAnalysis(
        l1_ratio=config.coxnet_l1_ratio,
        alphas=alphas,
        max_iter=100000,
    )
    final_model.fit(X_scaled, y)
    return final_model, alpha_summary, best_alpha


def plot_coxnet_path(model: CoxnetSurvivalAnalysis, feature_names: list[str], output_path: Path, best_alpha: float) -> None:
    """Plot coefficient trajectories over the regularisation path."""
    coefs = model.coef_
    alphas = model.alphas_
    max_abs = np.max(np.abs(coefs), axis=1)
    active_mask = max_abs > 1e-10
    active_indices = np.where(active_mask)[0]

    if len(active_indices) > 40:
        active_indices = active_indices[np.argsort(max_abs[active_indices])[-40:]]

    plt.figure(figsize=(8, 5.5))
    cmap = plt.colormaps["tab20"].resampled(max(len(active_indices), 1))
    for color_idx, idx in enumerate(active_indices):
        plt.plot(
            np.log10(alphas),
            coefs[idx, :],
            linewidth=1.4,
            color=cmap(color_idx),
            label=feature_names[idx],
        )
    plt.axvline(np.log10(best_alpha), color="black", linestyle="--", linewidth=1.0, label="best alpha")
    plt.xlabel("log10(alpha)")
    plt.ylabel("Coxnet coefficient")
    plt.title("Coxnet coefficient path")
    if len(active_indices) <= 25:
        plt.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    if len(active_indices) > 25:
        legend_path = output_path.with_name(f"{output_path.stem}_legend{output_path.suffix}")
        legend_height = max(4.0, 0.24 * len(active_indices))
        plt.figure(figsize=(6, legend_height))
        plt.axis("off")
        for color_idx, idx in enumerate(active_indices):
            y = 1 - (color_idx + 0.5) / len(active_indices)
            plt.plot([0.02, 0.12], [y, y], color=cmap(color_idx), linewidth=2.0)
            plt.text(0.15, y, feature_names[idx], va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(legend_path, dpi=200)
        plt.close()


def extract_best_coefficients(
    model: CoxnetSurvivalAnalysis,
    feature_names: list[str],
    best_alpha: float,
) -> pd.DataFrame:
    """Extract coefficients at the selected alpha."""
    alpha_idx = int(np.argmin(np.abs(model.alphas_ - best_alpha)))
    coefs = model.coef_[:, alpha_idx]
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
            "alpha": model.alphas_[alpha_idx],
            "nonzero": np.abs(coefs) > 1e-10,
        }
    )
    return coef_df.sort_values("abs_coefficient", ascending=False)


def plot_best_coefficients(coef_df: pd.DataFrame, output_path: Path) -> None:
    nonzero = coef_df[coef_df["nonzero"]].copy()
    if nonzero.empty:
        nonzero = coef_df.head(20).copy()
    nonzero = nonzero.sort_values("coefficient", ascending=True)

    colors = np.where(nonzero["coefficient"] >= 0, "#b2182b", "#2166ac")
    height = max(4.0, 0.35 * len(nonzero) + 1.5)
    plt.figure(figsize=(7.5, height))
    plt.barh(nonzero["feature"], nonzero["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Coxnet coefficient at selected alpha")
    plt.ylabel("Feature")
    plt.title("Best Coxnet model coefficients")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def select_standard_cox_features(
    feature_ranking: pd.DataFrame,
    coxnet_coefficients: pd.DataFrame,
    config: Config,
) -> list[str]:
    """Use Coxnet-selected variables, backed up by top univariate variables."""
    all_features = feature_ranking["feature"].tolist()
    clinical_features = []
    for covariate in config.clinical_covariate_cols:
        clinical_features.extend(
            [
                feature
                for feature in all_features
                if feature == covariate or feature.startswith(f"{covariate}_")
            ]
        )

    selected = coxnet_coefficients.loc[coxnet_coefficients["nonzero"], "feature"].tolist()
    if not selected:
        selected = feature_ranking.head(config.feature_selection_top_n)["feature"].tolist()

    top_ranked = feature_ranking.head(config.feature_selection_top_n)["feature"].tolist()
    selected = list(dict.fromkeys([*clinical_features, *selected, *top_ranked]))
    if len(selected) > config.standard_cox_max_features:
        ranking_lookup = feature_ranking.set_index("feature")["mean_cv_c_index"].to_dict()
        nonclinical = [feature for feature in selected if feature not in clinical_features]
        nonclinical = sorted(nonclinical, key=lambda value: ranking_lookup.get(value, -np.inf), reverse=True)
        remaining_slots = max(config.standard_cox_max_features - len(clinical_features), 0)
        selected = [*clinical_features, *nonclinical[:remaining_slots]]
    return selected


def fit_standard_cox(
    case_table: pd.DataFrame,
    selected_features: list[str],
    config: Config,
) -> tuple[CoxPHFitter, pd.DataFrame]:
    """Fit a conventional Cox PH model to selected features to get p-values/CIs."""
    fit_df = case_table[[*selected_features, "duration_days", "event"]].copy()
    fit_df["event"] = fit_df["event"].astype(int)
    cph = CoxPHFitter(penalizer=config.standard_cox_penalizer)
    cph.fit(fit_df, duration_col="duration_days", event_col="event", robust=True)
    result = cph.summary.reset_index().rename(columns={"covariate": "feature"})
    return cph, result


def plot_cox_forest(cox_results: pd.DataFrame, output_path: Path) -> None:
    """Create a forest plot of standard Cox coefficients with 95% CIs."""
    df = cox_results.copy()
    df = df.sort_values("coef", ascending=True)
    coef = df["coef"].to_numpy(dtype=float)
    lower = df["coef lower 95%"].to_numpy(dtype=float)
    upper = df["coef upper 95%"].to_numpy(dtype=float)
    y = np.arange(len(df))
    xerr = np.vstack([coef - lower, upper - coef])
    colors = np.where(coef >= 0, "#b2182b", "#2166ac")

    height = max(4.0, 0.35 * len(df) + 1.8)
    plt.figure(figsize=(8, height))
    plt.errorbar(coef, y, xerr=xerr, fmt="none", ecolor="#666666", capsize=2, zorder=1)
    plt.scatter(coef, y, c=colors, s=36, zorder=2)
    plt.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    plt.yticks(y, df["feature"])
    plt.xlabel("Cox coefficient (log hazard ratio)")
    plt.ylabel("Selected feature")
    plt.title("Standard Cox PH model: selected feature effects")

    x_range = np.nanmax(upper) - np.nanmin(lower)
    if not np.isfinite(x_range) or x_range == 0:
        x_range = 1.0
    x_text = np.nanmax(upper) + 0.08 * x_range
    for y_pos, p_value in zip(y, df["p"]):
        plt.text(x_text, y_pos, f"p={p_value:.3g}", va="center", fontsize=8)
    plt.xlim(left=np.nanmin(lower) - 0.08 * x_range, right=x_text + 0.25 * x_range)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_run_summary(
    output_dir: Path,
    config: Config,
    cluster_col: str,
    X: pd.DataFrame,
    feature_ranking: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    best_alpha: float,
    coxnet_coefficients: pd.DataFrame,
    selected_features: list[str],
) -> None:
    nonzero_count = int(coxnet_coefficients["nonzero"].sum())
    best_cv = float(alpha_summary.iloc[0]["mean_c_index"])
    with open(output_dir / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write("HyPERSTAC Coxnet survival analysis\n")
        f.write("==================================\n\n")
        f.write(f"Cluster column: {cluster_col}\n")
        f.write(f"Cases: {X.shape[0]}\n")
        f.write(f"Input features: {X.shape[1]}\n")
        f.write(f"Best alpha: {best_alpha:.8g}\n")
        f.write(f"Best CV C-index: {best_cv:.4f}\n")
        f.write(f"Non-zero Coxnet coefficients: {nonzero_count}\n")
        f.write(f"Standard Cox selected features: {len(selected_features)}\n\n")
        f.write("Top univariate features:\n")
        for row in feature_ranking.head(10).itertuples(index=False):
            f.write(f"  {row.feature}: CV C-index={row.mean_cv_c_index:.4f}\n")
        f.write("\nSelected standard Cox features:\n")
        for feature in selected_features:
            f.write(f"  {feature}\n")
        f.write("\nConfiguration:\n")
        json.dump({key: str(value) for key, value in asdict(config).items()}, f, indent=2)


def run_single_analysis(
    adata: ad.AnnData,
    clinical: pd.DataFrame,
    cluster_col: str,
    output_dir: Path,
    config: Config,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    X, case_table = build_case_features(adata, clinical, cluster_col, config)
    y = make_survival_array(case_table)
    X_scaled, _ = standardize_features(X)

    case_table.to_csv(output_dir / "case_features.csv")
    feature_ranking = rank_features_univariate(X_scaled, y, config.coxnet_cv_folds, config.seed)
    feature_ranking.to_csv(output_dir / "feature_selection.csv", index=False)

    coxnet_model, alpha_summary, best_alpha = fit_coxnet_cv(X_scaled, y, config)
    alpha_summary.to_csv(output_dir / "coxnet_alpha_cv.csv", index=False)
    plot_coxnet_path(coxnet_model, X.columns.tolist(), output_dir / "coxnet_path_plot.png", best_alpha)

    coxnet_coefficients = extract_best_coefficients(coxnet_model, X.columns.tolist(), best_alpha)
    coxnet_coefficients.to_csv(output_dir / "coxnet_coefficients.csv", index=False)
    plot_best_coefficients(coxnet_coefficients, output_dir / "best_model_coefficients.png")

    selected_features = select_standard_cox_features(feature_ranking, coxnet_coefficients, config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, standard_cox_results = fit_standard_cox(case_table, selected_features, config)
    standard_cox_results.to_csv(output_dir / "standard_cox_results.csv", index=False)
    plot_cox_forest(standard_cox_results, output_dir / "cox_forest_plot.png")

    write_run_summary(
        output_dir=output_dir,
        config=config,
        cluster_col=cluster_col,
        X=X,
        feature_ranking=feature_ranking,
        alpha_summary=alpha_summary,
        best_alpha=best_alpha,
        coxnet_coefficients=coxnet_coefficients,
        selected_features=selected_features,
    )

    return {
        "cluster_col": cluster_col,
        "n_cases": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "best_alpha": best_alpha,
        "best_cv_c_index": float(alpha_summary.iloc[0]["mean_c_index"]),
        "nonzero_coxnet_features": int(coxnet_coefficients["nonzero"].sum()),
        "standard_cox_features": int(len(selected_features)),
    }


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    clinical = read_clinical_table(config)
    adata = ad.read_h5ad(config.adata_path)
    cluster_cols = find_cluster_columns(adata, config)

    summaries = []
    for cluster_col in cluster_cols:
        if len(cluster_cols) == 1:
            analysis_dir = config.output_dir
        else:
            safe_name = "".join(ch if ch.isalnum() else "_" for ch in cluster_col).strip("_")
            analysis_dir = config.output_dir / safe_name
        print(f"Running Coxnet survival analysis for adata.obs['{cluster_col}']")
        summaries.append(run_single_analysis(adata, clinical, cluster_col, analysis_dir, config))

    if len(summaries) > 1:
        pd.DataFrame(summaries).to_csv(config.output_dir / "coxnet_all_cluster_summary.csv", index=False)


if __name__ == "__main__":
    main()

"""
Temporal Variance Analysis for the Bitcoin Fee Estimation Pipeline.

Decomposes the variance of model features and coefficients across epochs
to determine whether extending the study to cover more calendar time
would improve structural estimates.

Analyses performed:
  A. Intraclass Correlation Coefficients (ICC)
  B. Design Effect and Effective Sample Size
  C. Cumulative Precision Curve
  D. Rolling-Window Coefficient Stability
  E. Epoch Fixed Effect Autocorrelation
  F. Out-of-Sample Temporal Cross-Validation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CONTROL_FEATURES = [
    "log_Wprime",
    "has_rbf", "is_cpfp_package", "log_total_output",
    "log_n_in", "log_n_out",
    "has_op_return", "has_inscription",
]
_STATE_FEATURES = [
    "blockspace_utilization",
    "log_time_since_last_block",
    "log_size",
]
_ALL_FEATURES = _CONTROL_FEATURES + _STATE_FEATURES


# ---------------------------------------------------------------------------
# A. Intraclass Correlation Coefficient
# ---------------------------------------------------------------------------

def compute_icc(
    df: pd.DataFrame, value_col: str, group_col: str = "epoch_id"
) -> Dict[str, float]:
    """One-way random effects ICC(1).

    Returns dict with keys: icc, var_between, var_within, n_groups, n_bar, N.
    """
    groups = df.groupby(group_col)[value_col]
    grand_mean = df[value_col].mean()

    n_groups = groups.ngroups
    group_sizes = groups.size()
    n_bar = group_sizes.mean()
    N = len(df)

    group_means = groups.mean()
    SSB = np.sum(group_sizes.values * (group_means.values - grand_mean) ** 2)
    SSW = np.sum(groups.apply(lambda g: np.sum((g - g.mean()) ** 2)))

    dfB = n_groups - 1
    dfW = N - n_groups

    MSB = SSB / dfB if dfB > 0 else 0
    MSW = SSW / dfW if dfW > 0 else 1e-12

    sigma2_within = MSW
    sigma2_between = max((MSB - MSW) / n_bar, 0)
    sigma2_total = sigma2_between + sigma2_within

    icc = sigma2_between / sigma2_total if sigma2_total > 0 else 0

    return {
        "icc": icc,
        "var_between": sigma2_between,
        "var_within": sigma2_within,
        "n_groups": n_groups,
        "n_bar": n_bar,
        "N": N,
    }


def _icc_from_arrays(values: np.ndarray, groups: np.ndarray) -> float:
    """Fast ICC(1) from raw arrays (returns scalar)."""
    unique_g, inverse = np.unique(groups, return_inverse=True)
    k = len(unique_g)
    if k < 2:
        return 0.0
    N = len(values)

    group_sums = np.bincount(inverse, weights=values)
    group_counts = np.bincount(inverse).astype(float)
    group_means = group_sums / group_counts
    grand_mean = values.mean()
    n_bar = group_counts.mean()

    SSB = np.sum(group_counts * (group_means - grand_mean) ** 2)
    SSW = np.sum((values - group_means[inverse]) ** 2)

    MSB = SSB / (k - 1)
    MSW = SSW / max(N - k, 1)

    sigma2_between = max((MSB - MSW) / n_bar, 0.0)
    sigma2_total = sigma2_between + MSW
    return sigma2_between / sigma2_total if sigma2_total > 0 else 0.0


# ---------------------------------------------------------------------------
# B. Fit intensive margin on a subset of epochs (FWL approach)
# ---------------------------------------------------------------------------

def _filter_for_intensive(
    df: pd.DataFrame, eps: float, slope_trim: float,
) -> pd.DataFrame:
    """Apply the same filters as HurdleFeeModel.fit()."""
    required = ["fee_rate", "epoch_id"] + _ALL_FEATURES
    mask = pd.Series(True, index=df.index)
    for col in required:
        if col in df.columns:
            mask &= df[col].notna()
    mask &= df["fee_rate"] > eps
    out = df[mask].copy()

    if slope_trim > 0 and len(out) > 0:
        threshold = out["log_Wprime"].quantile(slope_trim)
        out = out[out["log_Wprime"] > threshold].copy()
    return out


def _fwl_ols_with_clustered_se(
    df: pd.DataFrame, feature_cols: List[str],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """OLS via Frisch-Waugh-Lovell (within-epoch demeaning) with clustered SEs.

    Concentrates out epoch fixed effects by demeaning, then computes the
    Liang-Zeger sandwich on the small dense (N x p) demeaned system.

    Returns (beta, se) arrays aligned with *feature_cols*, or None.
    """
    y_col = "_log_fee_rate"
    df[y_col] = np.log(df["fee_rate"].values)

    cols_to_demean = feature_cols + [y_col]
    epoch_means = df.groupby("epoch_id")[cols_to_demean].transform("mean")

    X_dm = df[feature_cols].values - epoch_means[feature_cols].values
    y_dm = df[y_col].values - epoch_means[y_col].values

    n, p = X_dm.shape
    n_epochs = df["epoch_id"].nunique()

    XtX = X_dm.T @ X_dm
    try:
        beta = np.linalg.solve(XtX, X_dm.T @ y_dm)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]

    residuals = y_dm - X_dm @ beta

    ridge = 1e-10 * np.trace(XtX) / max(p, 1)
    XtX_inv = np.linalg.inv(XtX + ridge * np.eye(p))

    epoch_ids = df["epoch_id"].values
    unique_clusters = df["epoch_id"].unique()
    n_clusters = len(unique_clusters)

    meat = np.zeros((p, p))
    for c in unique_clusters:
        mask = epoch_ids == c
        score_c = X_dm[mask].T @ residuals[mask]
        meat += np.outer(score_c, score_c)

    dof = max(n - n_epochs - p, 1)
    adjustment = (n_clusters / max(n_clusters - 1, 1)) * ((n - 1) / dof)
    V = XtX_inv @ (adjustment * meat) @ XtX_inv

    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return beta, se


def fit_intensive_margin_subset(
    df_subset: pd.DataFrame,
    stage2_template: Any,
) -> Optional[Dict[str, Any]]:
    """Fit the intensive margin on *df_subset* using FWL and return
    coefficient / SE for ``log_Wprime`` plus a summary dict.

    Uses within-epoch demeaning (Frisch-Waugh-Lovell) to concentrate out
    epoch fixed effects, avoiding numerical instability from large sparse
    dummy matrices.
    """
    eps = stage2_template.config.fee_threshold_sat_vb
    slope_trim = stage2_template.config.slope_trim

    df_v = _filter_for_intensive(df_subset, eps, slope_trim)
    if len(df_v) < 100 or df_v["epoch_id"].nunique() < 10:
        return None

    result = _fwl_ols_with_clustered_se(df_v, _ALL_FEATURES)
    if result is None:
        return None

    beta, se = result
    wprime_idx = _ALL_FEATURES.index("log_Wprime")

    return {
        "coef_wprime": beta[wprime_idx],
        "se_wprime": se[wprime_idx],
        "n_epochs": df_v["epoch_id"].nunique(),
        "n_obs": len(df_v),
        "coef_summary": {
            name: {"coef": beta[i], "se": se[i]}
            for i, name in enumerate(_ALL_FEATURES)
        },
    }


# ---------------------------------------------------------------------------
# B2. Out-of-sample temporal evaluation
# ---------------------------------------------------------------------------

def _evaluate_oos_temporal(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    eps: float,
    slope_trim: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Extended OOS evaluation with three components:

    1. **Within-epoch OOS** (FWL): demean within test epochs, predict with
       training betas.  Tests whether structural coefficients generalize.
    2. **Strict OOS** (no test-epoch info): predict using only structural
       features + training intercept, with no access to test-epoch means.
       Tests true forecasting ability.
    3. **OOS variance decomposition**: ICC of test outcomes vs. ICC of
       strict residuals.  Shows whether the model absorbs between-epoch
       or within-epoch variance.
    """
    df_tr = _filter_for_intensive(df_train, eps, slope_trim)
    df_te = _filter_for_intensive(df_test, eps, slope_trim)

    if len(df_tr) < 200 or len(df_te) < 200:
        return None
    if df_tr["epoch_id"].nunique() < 10 or df_te["epoch_id"].nunique() < 10:
        return None

    feature_cols = list(_ALL_FEATURES)
    restricted_cols = [c for c in feature_cols if c != "log_Wprime"]
    ri = [feature_cols.index(c) for c in restricted_cols]

    # ---- Train: FWL on training epochs ----
    df_tr["_log_fee_rate"] = np.log(df_tr["fee_rate"].values)
    cols_dm = feature_cols + ["_log_fee_rate"]
    tr_means = df_tr.groupby("epoch_id")[cols_dm].transform("mean")
    X_dm_tr = df_tr[feature_cols].values - tr_means[feature_cols].values
    y_dm_tr = df_tr["_log_fee_rate"].values - tr_means["_log_fee_rate"].values

    beta_full = np.linalg.lstsq(X_dm_tr, y_dm_tr, rcond=None)[0]
    beta_restr = np.linalg.lstsq(X_dm_tr[:, ri], y_dm_tr, rcond=None)[0]

    # ---- 1. Within-epoch OOS ----
    df_te["_log_fee_rate"] = np.log(df_te["fee_rate"].values)
    te_means = df_te.groupby("epoch_id")[cols_dm].transform("mean")
    X_dm_te = df_te[feature_cols].values - te_means[feature_cols].values
    y_dm_te = df_te["_log_fee_rate"].values - te_means["_log_fee_rate"].values

    pred_full_we = X_dm_te @ beta_full
    pred_restr_we = X_dm_te[:, ri] @ beta_restr

    ss_tot_we = np.sum(y_dm_te ** 2)
    if ss_tot_we < 1e-12:
        return None

    r2_full = 1.0 - np.sum((y_dm_te - pred_full_we) ** 2) / ss_tot_we
    r2_restr = 1.0 - np.sum((y_dm_te - pred_restr_we) ** 2) / ss_tot_we
    rmse_full = np.sqrt(np.mean((y_dm_te - pred_full_we) ** 2))
    rmse_restr = np.sqrt(np.mean((y_dm_te - pred_restr_we) ** 2))

    # ---- 2. Strict OOS (no test-epoch demeaning) ----
    y_tr_raw = df_tr["_log_fee_rate"].values
    X_tr_raw = df_tr[feature_cols].values
    y_te_raw = df_te["_log_fee_rate"].values
    X_te_raw = df_te[feature_cols].values

    intercept_full = np.mean(y_tr_raw) - X_tr_raw.mean(axis=0) @ beta_full
    intercept_restr = np.mean(y_tr_raw) - X_tr_raw[:, ri].mean(axis=0) @ beta_restr

    pred_strict_full = intercept_full + X_te_raw @ beta_full
    pred_strict_restr = intercept_restr + X_te_raw[:, ri] @ beta_restr

    ss_tot_strict = np.sum((y_te_raw - y_te_raw.mean()) ** 2)
    r2_strict_full = 1.0 - np.sum((y_te_raw - pred_strict_full) ** 2) / ss_tot_strict
    r2_strict_restr = 1.0 - np.sum((y_te_raw - pred_strict_restr) ** 2) / ss_tot_strict
    rmse_strict = np.sqrt(np.mean((y_te_raw - pred_strict_full) ** 2))

    # ---- 3. OOS variance decomposition ----
    epoch_ids_te = df_te["epoch_id"].values
    icc_outcome = _icc_from_arrays(y_te_raw, epoch_ids_te)
    strict_resid = y_te_raw - pred_strict_full
    icc_resid = _icc_from_arrays(strict_resid, epoch_ids_te)

    var_outcome = np.var(y_te_raw)
    var_resid = np.var(strict_resid)
    var_between_outcome = icc_outcome * var_outcome
    var_within_outcome = (1 - icc_outcome) * var_outcome
    var_between_resid = icc_resid * var_resid
    var_within_resid = (1 - icc_resid) * var_resid

    deff_outcome = 1 + (np.mean(np.bincount(
        np.unique(epoch_ids_te, return_inverse=True)[1])) - 1) * icc_outcome
    deff_resid = 1 + (np.mean(np.bincount(
        np.unique(epoch_ids_te, return_inverse=True)[1])) - 1) * icc_resid

    return {
        "n_train_epochs": df_tr["epoch_id"].nunique(),
        "n_test_epochs": df_te["epoch_id"].nunique(),
        "n_train": len(df_tr),
        "n_test": len(df_te),
        # Within-epoch OOS
        "r2_full": r2_full,
        "r2_restricted": r2_restr,
        "rmse_full": rmse_full,
        "rmse_restricted": rmse_restr,
        "r2_gain_wprime": r2_full - r2_restr,
        # Strict OOS
        "r2_strict_full": r2_strict_full,
        "r2_strict_restricted": r2_strict_restr,
        "rmse_strict": rmse_strict,
        # Variance decomposition
        "icc_outcome": icc_outcome,
        "icc_resid": icc_resid,
        "var_between_outcome": var_between_outcome,
        "var_within_outcome": var_within_outcome,
        "var_between_resid": var_between_resid,
        "var_within_resid": var_within_resid,
        "deff_outcome": deff_outcome,
        "deff_resid": deff_resid,
    }


# ---------------------------------------------------------------------------
# C–E. Main analysis driver
# ---------------------------------------------------------------------------

def run_temporal_variance_analysis(
    df: pd.DataFrame,
    stage2: Any,
    config: Any,
) -> Dict[str, Any]:
    """Run the full temporal variance analysis suite.

    Parameters
    ----------
    df : pd.DataFrame
        The prepared dataframe used to fit the pipeline (``estimator.df_prepared``).
    stage2 : HurdleFeeModel
        The fitted Stage 2 model.
    config : EstimationConfig
        Pipeline configuration.

    Returns
    -------
    dict
        Results dict with keys:
        icc_df, icc_results, DEFF, N_eff, n_obs, n_clusters, icc_wprime,
        cumulative_results, window_results, acf_values, fe_values, n_windows.
    """
    eps = config.fee_threshold_sat_vb

    valid_mask = (
        df["log_Wprime"].notna()
        & df["fee_rate"].notna()
        & (df["fee_rate"] > eps)
        & df["epoch_id"].notna()
    )
    df_valid = df[valid_mask].copy()
    df_valid["log_fee_rate_pos"] = np.log(df_valid["fee_rate"])

    # ---- A. ICC ----
    variables_for_icc = {
        "log(fee_rate)": "log_fee_rate_pos",
        "log_Wprime": "log_Wprime",
        "has_rbf": "has_rbf",
        "is_cpfp_package": "is_cpfp_package",
        "log_total_output": "log_total_output",
        "log_n_in": "log_n_in",
        "log_n_out": "log_n_out",
        "has_op_return": "has_op_return",
        "has_inscription": "has_inscription",
        "blockspace_utilization": "blockspace_utilization",
        "log_time_since_last_block": "log_time_since_last_block",
        "log_size": "log_size",
    }

    icc_results: Dict[str, Dict] = {}
    for label, col in variables_for_icc.items():
        if col in df_valid.columns and df_valid[col].notna().sum() > 0:
            icc_results[label] = compute_icc(df_valid, col)

    icc_df = pd.DataFrame({
        "variable": list(icc_results.keys()),
        "ICC": [r["icc"] for r in icc_results.values()],
        "var_between": [r["var_between"] for r in icc_results.values()],
        "var_within": [r["var_within"] for r in icc_results.values()],
    }).sort_values("ICC", ascending=False)

    n_obs = icc_results["log(fee_rate)"]["N"]
    n_clusters = icc_results["log(fee_rate)"]["n_groups"]
    n_bar = icc_results["log(fee_rate)"]["n_bar"]
    icc_outcome = icc_results["log(fee_rate)"]["icc"]

    DEFF = 1 + (n_bar - 1) * icc_outcome
    N_eff = n_obs / DEFF

    icc_wprime = icc_results.get("log_Wprime", {}).get("icc", 0)

    logger.info("=" * 70)
    logger.info("TEMPORAL VARIANCE ANALYSIS")
    logger.info("=" * 70)
    logger.info(
        "Data: %d observations in %d epoch-clusters (mean size %.1f)",
        n_obs, n_clusters, n_bar,
    )
    logger.info("ICC(log_Wprime) = %.4f", icc_wprime)
    logger.info(
        "Design Effect = %.1f  |  Effective N = %d / %d",
        DEFF, int(N_eff), n_obs,
    )

    # ---- C. Cumulative precision curve ----
    sorted_epochs = sorted(df_valid["epoch_id"].unique())
    n_total_epochs = len(sorted_epochs)

    fractions = [0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
    cumulative_results: List[Dict] = []

    logger.info("-" * 70)
    logger.info("Cumulative Precision Curve")

    for frac in fractions:
        k = max(10, int(n_total_epochs * frac))
        k = min(k, n_total_epochs)
        epoch_subset = sorted_epochs[:k]
        df_sub = df_valid[df_valid["epoch_id"].isin(epoch_subset)].copy()

        result = fit_intensive_margin_subset(df_sub, stage2)
        if result is not None:
            cumulative_results.append(result)
            sig = (
                "***" if abs(result["coef_wprime"]) / result["se_wprime"] > 3.29
                else "**" if abs(result["coef_wprime"]) / result["se_wprime"] > 2.58
                else "*" if abs(result["coef_wprime"]) / result["se_wprime"] > 1.96
                else ""
            )
            logger.info(
                "  Epochs: %4d  |  N: %7d  |  β(log_Wprime) = %8.4f  |  SE = %.4f %s",
                result["n_epochs"], result["n_obs"],
                result["coef_wprime"], result["se_wprime"], sig,
            )

    # ---- D. Rolling-window coefficient stability ----
    N_WINDOWS = 5
    window_size = n_total_epochs // N_WINDOWS
    window_results: List[Dict] = []

    logger.info("-" * 70)
    logger.info("Rolling-Window Coefficient Stability")

    for w in range(N_WINDOWS):
        start_idx = w * window_size
        end_idx = (w + 1) * window_size if w < N_WINDOWS - 1 else n_total_epochs
        window_epochs = sorted_epochs[start_idx:end_idx]
        df_window = df_valid[df_valid["epoch_id"].isin(window_epochs)].copy()
        result = fit_intensive_margin_subset(df_window, stage2)

        if result is not None:
            window_results.append(result)
            logger.info(
                "  Window %d (epochs %d-%d, N=%d): β(log_Wprime)=%.4f  SE=%.4f",
                w + 1, start_idx, end_idx - 1,
                result["n_obs"], result["coef_wprime"], result["se_wprime"],
            )

    if len(window_results) == N_WINDOWS and cumulative_results:
        pooled = cumulative_results[-1]
        key_features = [
            f for f in [
                "log_Wprime", "has_rbf", "log_total_output",
                "log_n_in", "log_n_out", "log_time_since_last_block", "log_size",
            ]
            if f in pooled["coef_summary"]
        ]
        for feat in key_features:
            coefs = [
                wr["coef_summary"][feat]["coef"]
                for wr in window_results if feat in wr["coef_summary"]
            ]
            ses = [
                wr["coef_summary"][feat]["se"]
                for wr in window_results if feat in wr["coef_summary"]
            ]
            if coefs:
                coef_range = max(coefs) - min(coefs)
                avg_se = np.mean(ses)
                ratio = coef_range / avg_se if avg_se > 0 else float("inf")
                stable = "Yes" if ratio < 2.0 else ("Maybe" if ratio < 3.0 else "No")
                logger.info(
                    "  %-30s  [%.4f, %.4f]  %.2f×SE  %s",
                    feat, min(coefs), max(coefs), ratio, stable,
                )

    # ---- E. Epoch FE autocorrelation ----
    epoch_fe = stage2.get_epoch_effects()
    acf_values = None
    fe_values = None

    if epoch_fe is not None and len(epoch_fe) > 0:
        epoch_fe = epoch_fe.sort_values("epoch_id").copy()
        fe_values = epoch_fe["intensive_fe"].values

        MAX_LAGS = min(48, len(fe_values) // 4)
        fe_demean = fe_values - fe_values.mean()
        var_fe = np.var(fe_demean)

        acf_vals: List[float] = []
        for lag in range(MAX_LAGS + 1):
            if var_fe > 0:
                autocorr = np.mean(fe_demean[: len(fe_demean) - lag] * fe_demean[lag:]) / var_fe
            else:
                autocorr = 0.0
            acf_vals.append(autocorr)
        acf_values = np.array(acf_vals)

        logger.info("-" * 70)
        logger.info("Epoch FE Autocorrelation (N=%d)", len(fe_values))
        report_lags = [1, 2, 4, 8, 12, 24, 48]
        for lag in report_lags:
            if lag <= MAX_LAGS:
                hours = lag * config.epoch_duration_minutes / 60
                logger.info("  Lag %3d (%5.1f h)  ACF=%.4f", lag, hours, acf_values[lag])
    else:
        logger.info("No epoch fixed effects available (model may not use epoch FE).")

    # ---- F. Out-of-sample temporal cross-validation ----
    logger.info("-" * 70)
    logger.info("Out-of-Sample Temporal Cross-Validation")

    oos_fractions = [0.2, 0.4, 0.6, 0.8]
    oos_results: List[Dict] = []

    for frac in oos_fractions:
        k = max(10, int(n_total_epochs * frac))
        k = min(k, n_total_epochs - 10)
        train_epochs = set(sorted_epochs[:k])
        test_epochs = set(sorted_epochs[k:])

        df_train = df_valid[df_valid["epoch_id"].isin(train_epochs)]
        df_test = df_valid[df_valid["epoch_id"].isin(test_epochs)]

        oos = _evaluate_oos_temporal(df_train, df_test, eps, config.slope_trim)
        if oos is not None:
            oos_results.append(oos)
            logger.info(
                "  Train: %4d ep (%7d obs)  Test: %4d ep (%7d obs)  |"
                "  within-epoch R²=%.4f  strict R²=%.4f  ΔR²(Wprime)=%.4f",
                oos["n_train_epochs"], oos["n_train"],
                oos["n_test_epochs"], oos["n_test"],
                oos["r2_full"], oos["r2_strict_full"], oos["r2_gain_wprime"],
            )

    if oos_results:
        best = oos_results[-1]
        logger.info("-" * 70)
        logger.info("OOS Variance Decomposition (80/20 split)")
        logger.info(
            "  Test outcome:  ICC=%.4f  Var(between)=%.4f  Var(within)=%.4f  DEFF=%.1f",
            best["icc_outcome"], best["var_between_outcome"],
            best["var_within_outcome"], best["deff_outcome"],
        )
        logger.info(
            "  Strict resid:  ICC=%.4f  Var(between)=%.4f  Var(within)=%.4f  DEFF=%.1f",
            best["icc_resid"], best["var_between_resid"],
            best["var_within_resid"], best["deff_resid"],
        )
        between_reduction = 1 - best["var_between_resid"] / best["var_between_outcome"] \
            if best["var_between_outcome"] > 0 else 0
        within_reduction = 1 - best["var_within_resid"] / best["var_within_outcome"] \
            if best["var_within_outcome"] > 0 else 0
        logger.info(
            "  Model absorbs: %.1f%% of between-epoch var, %.1f%% of within-epoch var",
            between_reduction * 100, within_reduction * 100,
        )

    # ---- Summary recommendation ----
    _log_summary(
        icc_wprime, DEFF, N_eff, n_obs,
        window_results, acf_values, oos_results, config,
    )

    return {
        "icc_df": icc_df,
        "icc_results": icc_results,
        "DEFF": DEFF,
        "N_eff": N_eff,
        "n_obs": n_obs,
        "n_clusters": n_clusters,
        "icc_wprime": icc_wprime,
        "cumulative_results": cumulative_results,
        "window_results": window_results,
        "acf_values": acf_values,
        "fe_values": fe_values,
        "n_windows": N_WINDOWS,
        "oos_results": oos_results,
    }


def _log_summary(
    icc_wprime: float,
    DEFF: float,
    N_eff: float,
    n_obs: int,
    window_results: List[Dict],
    acf_values: Optional[np.ndarray],
    oos_results: List[Dict],
    config: Any,
) -> None:
    """Log the final summary and recommendation."""
    icc_assessment = "high" if icc_wprime > 0.3 else ("moderate" if icc_wprime > 0.1 else "low")
    deff_assessment = "large" if DEFF > 10 else ("moderate" if DEFF > 3 else "small")

    stability_assessment = "unknown"
    stability_ratio = None
    if window_results:
        key_feat = "log_Wprime"
        if all(key_feat in wr.get("coef_summary", {}) for wr in window_results):
            wprime_coefs = [wr["coef_summary"][key_feat]["coef"] for wr in window_results]
            wprime_ses = [wr["coef_summary"][key_feat]["se"] for wr in window_results]
            coef_range = max(wprime_coefs) - min(wprime_coefs)
            avg_se = np.mean(wprime_ses)
            stability_ratio = coef_range / avg_se if avg_se > 0 else float("inf")
            stability_assessment = (
                "stable" if stability_ratio < 2.0
                else "borderline" if stability_ratio < 3.0
                else "unstable"
            )

    persistence = "unknown"
    if acf_values is not None:
        acf_at_12 = acf_values[12] if len(acf_values) > 12 else 0
        persistence = "high" if acf_at_12 > 0.3 else ("moderate" if acf_at_12 > 0.1 else "low")

    oos_assessment = "unknown"
    oos_r2_we = None
    oos_r2_strict = None
    oos_wprime_gain = None
    oos_icc_outcome = None
    oos_icc_resid = None
    if oos_results:
        best = oos_results[-1]
        oos_r2_we = best["r2_full"]
        oos_r2_strict = best["r2_strict_full"]
        oos_wprime_gain = best["r2_gain_wprime"]
        oos_icc_outcome = best["icc_outcome"]
        oos_icc_resid = best["icc_resid"]
        if oos_r2_we > 0.3:
            oos_assessment = "strong"
        elif oos_r2_we > 0.1:
            oos_assessment = "moderate"
        elif oos_r2_we > 0:
            oos_assessment = "weak"
        else:
            oos_assessment = "none"

    logger.info("=" * 70)
    logger.info("TEMPORAL VARIANCE ANALYSIS — SUMMARY")
    logger.info("=" * 70)
    logger.info("  1. ICC(log_Wprime) = %.4f → %s", icc_wprime, icc_assessment)
    logger.info("  2. Design Effect = %.1f → %s  (N_eff=%d / %d)", DEFF, deff_assessment, int(N_eff), n_obs)
    logger.info("  3. Coefficient stability → %s%s",
                stability_assessment,
                f"  (range/SE = {stability_ratio:.2f})" if stability_ratio else "")
    logger.info("  4. Epoch FE persistence → %s", persistence)
    if oos_r2_we is not None:
        logger.info("  5. OOS within-epoch R² → %s  (%.4f, ΔR²(Wprime)=%.4f)",
                     oos_assessment, oos_r2_we, oos_wprime_gain)
    if oos_r2_strict is not None:
        logger.info("  6. OOS strict R² (no epoch info) → %.4f", oos_r2_strict)
    if oos_icc_outcome is not None and oos_icc_resid is not None:
        logger.info("  7. OOS variance: ICC(outcome)=%.4f → ICC(residual)=%.4f",
                     oos_icc_outcome, oos_icc_resid)

    # ---- Recommendations ----
    if stability_assessment == "unstable" and oos_assessment in ("none", "weak"):
        logger.info("  RECOMMENDATION: More time may HURT — structural parameters drift and model is weak OOS.")
    elif stability_assessment == "unstable" and oos_assessment in ("moderate", "strong"):
        logger.info("  RECOMMENDATION: Parameters drift but model still generalizes OOS — "
                     "more data helps characterize regime variation.")
    elif oos_assessment == "none":
        logger.info("  RECOMMENDATION: Model does NOT generalize OOS — rethink specification before scaling up.")
    elif icc_assessment == "high" and deff_assessment == "large":
        logger.info("  RECOMMENDATION: More time helps SUBSTANTIALLY.")
    elif icc_assessment in ("moderate", "high") or deff_assessment in ("moderate", "large"):
        logger.info("  RECOMMENDATION: More time helps MODESTLY.")
    else:
        logger.info("  RECOMMENDATION: More time has LIMITED benefit.")

    if oos_wprime_gain is not None and oos_wprime_gain > 0.005:
        logger.info("  STRUCTURAL VARIABLE: log_Wprime adds %.4f R² OOS — VCG channel is informative.",
                     oos_wprime_gain)
    elif oos_wprime_gain is not None:
        logger.info("  STRUCTURAL VARIABLE: log_Wprime adds only %.4f R² OOS — marginal contribution is small.",
                     oos_wprime_gain)

    if oos_icc_resid is not None and oos_icc_outcome is not None:
        if oos_icc_resid > oos_icc_outcome * 0.9:
            logger.info("  VARIANCE: Model captures mostly within-epoch variation — "
                         "more epochs add independent info.")
        elif oos_icc_resid < oos_icc_outcome * 0.5:
            logger.info("  VARIANCE: Model absorbs substantial between-epoch variation — "
                         "structural features capture regime shifts.")
        else:
            logger.info("  VARIANCE: Model absorbs some between-epoch variation (ICC %.4f → %.4f).",
                         oos_icc_outcome, oos_icc_resid)

    if persistence == "high" and stability_assessment != "unstable":
        logger.info("  SAMPLING DESIGN: Strong FE autocorrelation suggests spaced-out collection.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_temporal_variance_analysis(
    results: Dict[str, Any],
    config: Any,
    output_dir: str = "plots",
) -> str:
    """Generate the 3x2 summary figure and save to *output_dir*.

    Returns the path to the saved figure.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    icc_df = results["icc_df"]
    cumulative_results = results["cumulative_results"]
    window_results = results["window_results"]
    acf_values = results["acf_values"]
    fe_values = results["fe_values"]
    oos_results = results.get("oos_results", [])

    has_oos = len(oos_results) > 0
    nrows = 3 if has_oos else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows))
    fig.suptitle(
        "Temporal Variance Analysis: Value of Extending the Study",
        fontsize=14, fontweight="bold",
    )

    # --- Panel A: ICC Bar Chart ---
    ax = axes[0, 0]
    icc_sorted = icc_df.sort_values("ICC", ascending=True)
    colors = [
        "#2ca02c" if v > 0.3 else "#ff7f0e" if v > 0.1 else "#1f77b4"
        for v in icc_sorted["ICC"]
    ]
    ax.barh(range(len(icc_sorted)), icc_sorted["ICC"], color=colors)
    ax.set_yticks(range(len(icc_sorted)))
    ax.set_yticklabels(icc_sorted["variable"], fontsize=9)
    ax.set_xlabel("ICC (fraction of variance between epochs)")
    ax.set_title("A. Intraclass Correlation Coefficients")
    ax.axvline(x=0.1, color="gray", linestyle="--", alpha=0.5, label="Low/Moderate")
    ax.axvline(x=0.3, color="gray", linestyle=":", alpha=0.5, label="Moderate/High")
    ax.legend(fontsize=8)
    ax.set_xlim(0, max(icc_sorted["ICC"].max() * 1.1, 0.5))

    # --- Panel B: Cumulative SE Curve ---
    ax = axes[0, 1]
    if cumulative_results:
        epochs_arr = np.array([r["n_epochs"] for r in cumulative_results])
        se_arr = np.array([r["se_wprime"] for r in cumulative_results])

        valid = np.isfinite(se_arr)
        if valid.any():
            ax.plot(
                epochs_arr[valid], se_arr[valid],
                "o-", color="#d62728", linewidth=2, markersize=6,
                label="Observed SE(log_Wprime)",
            )

            first_valid = np.where(valid)[0][0]
            theoretical = (se_arr[first_valid] * np.sqrt(epochs_arr[first_valid])
                           / np.sqrt(epochs_arr[valid]))
            ax.plot(
                epochs_arr[valid], theoretical, "--", color="gray", alpha=0.7,
                label=r"Theoretical $1/\sqrt{k}$",
            )

            last_se = se_arr[valid][-1]
            last_k = epochs_arr[valid][-1]
            if np.isfinite(last_se) and last_se > 0:
                target_se = last_se / 2
                needed_k = int(last_k * (last_se / target_se) ** 2)
                ax.axhline(y=target_se, color="green", linestyle=":", alpha=0.5)
                ax.annotate(
                    f"SE/2 → ~{needed_k} epochs",
                    xy=(last_k, target_se),
                    xytext=(epochs_arr[valid][0], target_se * 0.8),
                    fontsize=8, color="green",
                    arrowprops=dict(arrowstyle="->", color="green", alpha=0.5),
                )

        ax.set_xlabel("Number of epochs")
        ax.set_ylabel("SE(log_Wprime)")
        ax.set_title("B. Cumulative Precision Curve")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # --- Panel C: Rolling-Window Stability ---
    ax = axes[1, 0]
    if window_results:
        key_features_plot = [
            f for f in [
                "log_Wprime", "has_rbf", "log_total_output",
                "log_time_since_last_block", "log_size",
            ]
            if all(f in wr.get("coef_summary", {}) for wr in window_results)
        ]

        x_positions = np.arange(1, len(window_results) + 1)
        for i, feat in enumerate(key_features_plot):
            coefs = [wr["coef_summary"][feat]["coef"] for wr in window_results]
            ses = [wr["coef_summary"][feat]["se"] for wr in window_results]
            offset = (i - len(key_features_plot) / 2) * 0.08
            ax.errorbar(
                x_positions + offset, coefs,
                yerr=[1.96 * s for s in ses],
                fmt="o-", capsize=3, markersize=4, label=feat, linewidth=1,
            )

        ax.set_xlabel("Window")
        ax.set_ylabel("Coefficient (± 95% CI)")
        ax.set_title("C. Rolling-Window Coefficient Stability")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"W{i}" for i in x_positions])
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    # --- Panel D: Epoch FE Autocorrelation ---
    ax = axes[1, 1]
    if acf_values is not None and fe_values is not None:
        lags_to_plot = np.arange(1, len(acf_values))
        acf_to_plot = acf_values[1:]
        lag_hours = lags_to_plot * config.epoch_duration_minutes / 60

        ax.bar(
            lag_hours, acf_to_plot,
            width=config.epoch_duration_minutes / 60 * 0.8,
            color="#9467bd", alpha=0.7,
        )

        ci_95 = 1.96 / np.sqrt(len(fe_values))
        ax.axhline(y=ci_95, color="red", linestyle="--", alpha=0.5,
                    label=f"95% CI (±{ci_95:.3f})")
        ax.axhline(y=-ci_95, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=0, color="black", linewidth=0.5)

        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("D. Epoch Fixed Effect Autocorrelogram")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # --- Panels E–F: Out-of-Sample Results ---
    if has_oos:
        train_epochs = np.array([r["n_train_epochs"] for r in oos_results])
        r2_full = np.array([r["r2_full"] for r in oos_results])
        r2_restr = np.array([r["r2_restricted"] for r in oos_results])
        r2_strict = np.array([r["r2_strict_full"] for r in oos_results])

        # Panel E: OOS R² Learning Curve (within-epoch + strict)
        ax = axes[2, 0]
        ax.plot(train_epochs, r2_full, "o-", color="#2ca02c", linewidth=2,
                markersize=7, label="Within-epoch (full)")
        ax.plot(train_epochs, r2_restr, "s--", color="#ff7f0e", linewidth=2,
                markersize=7, label="Within-epoch (no Wprime)")
        ax.plot(train_epochs, r2_strict, "^-", color="#d62728", linewidth=2,
                markersize=7, label="Strict (no epoch info)")
        ax.fill_between(train_epochs, r2_restr, r2_full,
                        alpha=0.12, color="#2ca02c", label="ΔR² from log_Wprime")
        ax.fill_between(train_epochs, r2_strict, r2_restr,
                        alpha=0.08, color="#ff7f0e")
        ax.set_xlabel("Training epochs")
        ax.set_ylabel("Out-of-sample R²")
        ax.set_title("E. OOS R²: Within-Epoch vs. Strict Forecasting")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

        # Panel F: OOS Variance Decomposition (from last split)
        ax = axes[2, 1]
        best = oos_results[-1]
        categories = ["Test\noutcome", "Strict OOS\nresidual"]
        between = [best["var_between_outcome"], best["var_between_resid"]]
        within = [best["var_within_outcome"], best["var_within_resid"]]

        x = np.arange(len(categories))
        w = 0.5
        ax.bar(x, between, w, label="Between-epoch variance",
               color="#e74c3c", alpha=0.8)
        ax.bar(x, within, w, bottom=between, label="Within-epoch variance",
               color="#3498db", alpha=0.8)

        for i in range(len(categories)):
            total = between[i] + within[i]
            icc_val = between[i] / total if total > 0 else 0
            ax.text(x[i], total + 0.005 * max(between[0] + within[0], 0.01),
                    f"ICC={icc_val:.3f}", ha="center", fontsize=9, fontweight="bold")

        if best["var_between_outcome"] > 0:
            btw_pct = (1 - best["var_between_resid"] / best["var_between_outcome"]) * 100
        else:
            btw_pct = 0
        if best["var_within_outcome"] > 0:
            wit_pct = (1 - best["var_within_resid"] / best["var_within_outcome"]) * 100
        else:
            wit_pct = 0

        ax.annotate(
            f"Model absorbs\n{btw_pct:.0f}% between-epoch\n{wit_pct:.0f}% within-epoch",
            xy=(0.98, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9),
        )

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylabel("Variance")
        ax.set_title("F. OOS Variance Decomposition")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = str(output_path / "temporal_variance_analysis.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to: %s", save_path)
    return save_path

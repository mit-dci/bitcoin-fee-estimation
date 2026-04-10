#!/usr/bin/env python3
"""
Bitcoin Fee Estimation Pipeline — CLI entry point.

Runs the complete two-stage structural fee model (VCG-style) as a normal
Python process with progress logging, checkpointing, and explicit memory
management.  All pipeline logic is copied verbatim from
bitcoin_fee_estimation_pipeline.ipynb.

Usage examples:
    # Run from a pre-built dataset (fastest — no DB access needed)
    python run_pipeline.py --dataset data/bitcoin_data.parquet

    # Full run from raw databases
    python run_pipeline.py --db-path /path/to/data.db

    # Smoke-test on 100 blocks, no plots
    python run_pipeline.py --db-path /path/to/data.db --block-limit 100 --skip-plots

    # Full run with checkpointing (resumable after crash)
    python run_pipeline.py --db-path /path/to/data.db --checkpoint-dir /tmp/ckpt/
"""

# ---------------------------------------------------------------------------
# 1. stdlib imports
# ---------------------------------------------------------------------------
import argparse
import logging
import os
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

# ---------------------------------------------------------------------------
# 2. third-party imports
# ---------------------------------------------------------------------------
import joblib
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.interpolate import BSpline, interp1d
from scipy.optimize import lsq_linear, minimize

from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from build_dataset import build_analysis_dataset, prepare_features

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 3. Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 4. Constants / defaults
# ---------------------------------------------------------------------------

CHECKPOINT_FEATURES = "checkpoint_features.pkl"
CHECKPOINT_STAGE1   = "checkpoint_stage1.pkl"
IMPATIENCE_SPLINE_DEGREE = 3
IMPATIENCE_SPLINE_N_KNOTS = 5


# ---------------------------------------------------------------------------
# 5. EstimationConfig dataclass
# ---------------------------------------------------------------------------
@dataclass
class EstimationConfig:
    """Configuration for the two-stage fee estimation pipeline."""

    # Database settings
    db_path: str = "/home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db"
    annotation_path: str = "/home/armin/datalake/data-samples/tx-annotations-15m-2-7-2026.pkl"
    block_limit: Optional[int] = None

    # Epoch parameters
    epoch_mode: str = "time"
    blocks_per_epoch: int = 2
    target_epoch_size: int = 1000
    epoch_duration_minutes: int = 30

    # Stage 1 parameters
    stage1_model: str = "rf"
    n_folds: int = 5
    rf_n_estimators: int = 200
    rf_max_depth: int = 15
    rf_min_samples_leaf: int = 20
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: float = 10.0

    # Slope computation
    slope_delta: float = 0.05
    slope_trim: float = 0.0
    p_grid_size: int = 51

    # Stage 2 parameters
    fee_threshold_sat_vb: float = 0.1

    # Impatience proxy
    respend_truncation_blocks: int = 14
    epsilon: float = 1e-6

    # Bitcoin consensus
    max_block_weight: int = 4_000_000

    @property
    def feature_params(self) -> Dict:
        """Parameters needed by build_dataset.prepare_features()."""
        return dict(
            epoch_mode=self.epoch_mode,
            blocks_per_epoch=self.blocks_per_epoch,
            target_epoch_size=self.target_epoch_size,
            epoch_duration_minutes=self.epoch_duration_minutes,
            respend_truncation_blocks=self.respend_truncation_blocks,
            epsilon=self.epsilon,
            max_block_weight=self.max_block_weight,
        )


class ISplineTransformer:
    """I-spline basis transformer for a weakly increasing impatience effect."""

    def __init__(self, n_knots: int = 5, degree: int = 3):
        self.n_knots = n_knots
        self.degree = degree
        self._knot_vector = None
        self._n_basis = None
        self._lower = None
        self._upper = None

    def fit(self, X) -> "ISplineTransformer":
        x = np.asarray(X).ravel()
        quantiles = np.linspace(0, 100, self.n_knots)
        knot_positions = np.percentile(x, quantiles)

        self._lower = float(knot_positions[0])
        self._upper = float(knot_positions[-1])
        internal = knot_positions[1:-1]

        self._knot_vector = np.concatenate([
            np.repeat(self._lower, self.degree + 1),
            internal,
            np.repeat(self._upper, self.degree + 1),
        ])
        self._n_basis = len(self._knot_vector) - self.degree - 1
        return self

    def transform(self, X) -> np.ndarray:
        x = np.asarray(X).ravel()
        n = len(x)
        order = self.degree + 1
        basis = np.zeros((n, self._n_basis))

        x_clamped = np.clip(x, self._lower, self._upper)

        for i in range(self._n_basis):
            span = self._knot_vector[i + order] - self._knot_vector[i]
            if span <= 0:
                continue

            coefs = np.zeros(self._n_basis)
            coefs[i] = order / span
            mspl = BSpline(self._knot_vector, coefs, self.degree)

            ispl = mspl.antiderivative()
            vals = ispl(x_clamped) - ispl(self._lower)

            total = ispl(self._upper) - ispl(self._lower)
            if total > 1e-12:
                vals /= total

            basis[:, i] = np.clip(vals, 0.0, 1.0)

        return basis

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# 9. Class: DelayTechnologyEstimator
# ---------------------------------------------------------------------------

class DelayTechnologyEstimator:
    """Stage 1: Estimate delay technology W(p, s) following VCG notebook approach."""

    FEATURES = ["fee_rate_percentile", "mempool_size", "mempool_tx_count", "blockspace_utilization"]
    TARGET   = "log_waittime"

    def __init__(self, config: EstimationConfig):
        self.config = config
        self.model  = None
        self._is_fitted   = False
        self._valid_indices = None
        self._model_label = None

    def _build_stage1_model(self):
        """Construct the configured Stage 1 learner."""
        model_type = getattr(self.config, "stage1_model", "rf")

        if model_type == "rf":
            self._model_label = "Random Forest"
            return RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                random_state=42,
                n_jobs=-1,
            )

        if model_type == "xgb_monotone":
            self._model_label = "Monotone XGBoost"
            return XGBRegressor(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                min_child_weight=self.config.xgb_min_child_weight,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
                monotone_constraints=(-1, 0, 0, 0),
            )

        raise ValueError(f"Unknown stage1_model: {model_type}")

    def _uses_intrinsic_monotonicity(self) -> bool:
        return getattr(self.config, "stage1_model", "rf") == "xgb_monotone"

    def fit(self, df: pd.DataFrame) -> "DelayTechnologyEstimator":
        logger.info("=" * 60)
        logger.info("STAGE 1: Fitting Delay Technology Model")
        logger.info("=" * 60)

        missing_cols = [c for c in self.FEATURES + [self.TARGET] if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        valid_mask  = df[self.FEATURES + [self.TARGET]].notna().all(axis=1)
        df_valid    = df[valid_mask].copy()
        self._valid_indices = df_valid.index

        logger.info("Training on %d valid observations", len(df_valid))

        X = df_valid[self.FEATURES].values
        y = df_valid[self.TARGET].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info("Train/test split: %d / %d", len(X_train), len(X_test))

        self.model = self._build_stage1_model()
        logger.info("Training %s...", self._model_label)
        self.model.fit(X_train, y_train)

        r2_test  = r2_score(y_test, self.model.predict(X_test))
        rmse_test = np.sqrt(mean_squared_error(y_test, self.model.predict(X_test)))
        logger.info("Test R²: %.4f  RMSE: %.4f", r2_test, rmse_test)

        if hasattr(self.model, "feature_importances_"):
            logger.info("Feature Importance:")
            for feat, imp in sorted(
                zip(self.FEATURES, self.model.feature_importances_), key=lambda x: -x[1]
            ):
                logger.info("  %-25s %.4f", feat, imp)

        self._is_fitted = True
        self._r2_test   = r2_test
        return self

    def predict_W_hat(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        df_valid = df.loc[self._valid_indices]
        return self.model.predict(df_valid[self.FEATURES].values)

    def predict_delay_monotone_per_obs(
        self, df: pd.DataFrame, W_hat: np.ndarray, verbose: bool = True
    ) -> np.ndarray:
        if self._uses_intrinsic_monotonicity():
            if verbose:
                logger.info(
                    "Stage 1 model is monotone by construction; using raw predictions as W_monotone."
                )
            return W_hat

        if verbose:
            logger.info("Enforcing monotonicity via per-epoch isotonic regression...")

        df_valid = df.loc[self._valid_indices].copy()
        df_valid["W_hat"] = W_hat
        W_monotone = np.zeros(len(W_hat))

        for epoch_id in df_valid["epoch_id"].unique():
            mask       = df_valid["epoch_id"] == epoch_id
            epoch_data = df_valid[mask]
            p = epoch_data["fee_rate_percentile"].values
            w = epoch_data["W_hat"].values

            if len(p) < 5:
                W_monotone[mask.values] = w
                continue

            sort_idx = np.argsort(p)
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            w_mono_sorted = iso.fit_transform(p[sort_idx], w[sort_idx])
            W_monotone[mask.values] = w_mono_sorted[np.argsort(sort_idx)]

        if verbose:
            logger.info(
                "Monotonicity enforced. W range: [%.2f, %.2f]",
                W_monotone.min(), W_monotone.max(),
            )
        return W_monotone

    def compute_slope_finite_diff_per_obs(
        self,
        df: pd.DataFrame,
        _W_ignored: np.ndarray = None,   # kept for call-site compatibility
        delta: float = 0.05,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Compute W'(p) via an epoch-median Stage 1 schedule.

        For each epoch:
          1. Hold all non-priority features at their epoch median.
          2. Sweep a 99-point priority grid through the learner schedule.
          3. Enforce monotonicity via isotonic regression for RF, or use the
             raw schedule for intrinsically monotone learners.
          4. Finite-difference the smooth schedule at each obs's actual priority.

        This avoids the two failure modes of the old interpolator approach:
          - Per-obs partial derivatives: RF is piecewise-constant, so ±delta rarely
            crosses a split → slope = 0 for most observations.
          - Per-epoch (p, W_hat) interpolator: mixes all feature effects; flat
            wherever congestion features dominate within the epoch.
        """
        if verbose:
            logger.info(
                "Computing local slopes via epoch-median %s schedule (delta=%.2f)...",
                self._model_label or self.config.stage1_model,
                delta,
            )

        df_valid = df.loc[self._valid_indices].copy()
        p_values = df_valid["fee_rate_percentile"].values
        p_idx    = self.FEATURES.index("fee_rate_percentile")

        p_grid = np.linspace(0.01, 0.99, 99)
        Wprime = np.zeros(len(p_values))

        for epoch_id in df_valid["epoch_id"].unique():
            mask       = df_valid["epoch_id"] == epoch_id
            epoch_data = df_valid[mask]

            # Epoch-median background features — isolates the priority effect
            X_base = epoch_data[self.FEATURES].median().values.astype(float)

            # Vary only priority across a fine grid
            X_grid = np.tile(X_base, (len(p_grid), 1))
            X_grid[:, p_idx] = p_grid

            W_grid = self.model.predict(X_grid)

            if self._uses_intrinsic_monotonicity():
                W_mono = W_grid
            else:
                iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
                W_mono = iso.fit_transform(p_grid, W_grid)

            interpolator = interp1d(
                p_grid, W_mono, kind="linear",
                bounds_error=False, fill_value=(W_mono[0], W_mono[-1]),
            )

            p_obs  = p_values[mask.values]
            p_low  = np.clip(p_obs - delta, 0.01, 0.99)
            p_high = np.clip(p_obs + delta, 0.01, 0.99)
            slopes = (interpolator(p_low) - interpolator(p_high)) / (p_high - p_low)
            Wprime[mask.values] = np.maximum(slopes, 1e-6)

        if verbose:
            pct_nontrivial = (Wprime > 1e-5).mean() * 100
            logger.info(
                "Slopes: range [%.6f, %.6f], mean %.6f, non-trivial %.1f%%",
                Wprime.min(), Wprime.max(), Wprime.mean(), pct_nontrivial,
            )
        return Wprime


# ---------------------------------------------------------------------------
# 10. Estimation helpers for Stage 2
# ---------------------------------------------------------------------------

def _prepend_intercept(X):
    """Prepend a column of ones, preserving sparsity."""
    n = X.shape[0]
    ones = np.ones((n, 1))
    if sparse.issparse(X):
        return sparse.hstack([sparse.csr_matrix(ones), X], format="csr")
    return np.hstack([ones, X])



def fit_regularized_ols(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0, bounds: Optional[list] = None
) -> np.ndarray:
    """Fit OLS/Ridge with optional coefficient bounds."""
    if bounds is None:
        if alpha > 0:
            model = Ridge(alpha=alpha, fit_intercept=False)
        else:
            model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        return model.coef_

    if alpha == 0:
        lb = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
        ub = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
        result = lsq_linear(X, y, bounds=(lb, ub))
        return result.x

    n = X.shape[0]

    def objective(beta):
        resid = y - X @ beta
        return 0.5 * np.sum(resid ** 2) / n + 0.5 * alpha * np.sum(beta ** 2)

    def gradient(beta):
        resid = y - X @ beta
        return -X.T @ resid / n + alpha * beta

    x0 = np.zeros(X.shape[1])
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds,
        options={"maxiter": 2000, "disp": False},
    )
    return result.x


# ---------------------------------------------------------------------------
# 10b. Clustered Standard Errors
# ---------------------------------------------------------------------------

def compute_clustered_se(X, residuals: np.ndarray,
                         cluster_ids: np.ndarray) -> np.ndarray:
    """Liang-Zeger cluster-robust sandwich standard errors.

    Works with both dense (ndarray) and scipy sparse matrices.
    """
    is_sp = sparse.issparse(X)

    if is_sp:
        n, p = X.shape
        col_var = np.asarray(X.power(2).mean(axis=0)).ravel() - \
                  np.asarray(X.mean(axis=0)).ravel() ** 2
    else:
        n, p = X.shape
        col_var = np.var(X, axis=0)
    degenerate = col_var < 1e-12

    XtX = X.T @ X
    if is_sp:
        XtX = XtX.toarray()
    # Small ridge on diagonal for numerical stability with large N / many epoch FEs
    ridge = 1e-8 * np.trace(XtX) / max(p, 1)
    XtX += ridge * np.eye(p)
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        logger.warning("Matrix inversion failed in clustered SE computation; returning NaN")
        return np.full(p, np.nan)

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((p, p))
    for c in unique_clusters:
        mask = cluster_ids == c
        X_c = X[mask]
        e_c = residuals[mask]
        score_c = X_c.T @ e_c
        if is_sp:
            score_c = np.asarray(score_c).ravel()
        meat += np.outer(score_c, score_c)

    adjustment = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - p))
    V = XtX_inv @ (adjustment * meat) @ XtX_inv

    diag_V = np.diag(V)
    se = np.where(diag_V > 0, np.sqrt(diag_V), np.nan)
    se[degenerate] = np.nan
    return se


def compute_intensive_margin_se(stage2, df: pd.DataFrame) -> pd.DataFrame:
    """Compute clustered SEs for intensive-margin coefficients."""
    logger.info("Computing clustered standard errors for intensive margin...")

    all_cols = ["fee_rate", "epoch_id", "impatience"] + \
               stage2.CONTROL_FEATURES + stage2.STATE_FEATURES
    valid_mask = pd.Series(True, index=df.index)
    for col in dict.fromkeys(all_cols):
        valid_mask &= df[col].notna()
    valid_mask &= df["fee_rate"] > 0
    df_valid = df[valid_mask].copy()

    # Match the slope_trim applied during fit() (if any)
    slope_trim = stage2.config.slope_trim
    if slope_trim > 0 and "log_Wprime" in df_valid.columns:
        threshold = df_valid["log_Wprime"].quantile(slope_trim)
        df_valid = df_valid[df_valid["log_Wprime"] > threshold].copy()

    X_base, feature_names = stage2._build_feature_matrix(df_valid, fit_spline=False)
    X = _prepend_intercept(X_base)

    y = np.log(df_valid["fee_rate"].values)
    y_pred = X @ stage2.int_coefs
    if sparse.issparse(y_pred):
        y_pred = np.asarray(y_pred).ravel()
    residuals = y - y_pred

    cluster_ids = df_valid["epoch_id"].values
    se_clustered = compute_clustered_se(X, residuals, cluster_ids)

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    n = len(y)
    sigma2 = np.sum(residuals ** 2) / (n - X.shape[1])
    XtX = X.T @ X
    if sparse.issparse(XtX):
        XtX = XtX.toarray()
    XtX_inv = np.linalg.pinv(XtX)
    diag_ols = sigma2 * np.diag(XtX_inv)
    se_naive = np.where(diag_ols > 0, np.sqrt(diag_ols), np.nan)

    from scipy import stats as sp_stats
    names = ["intercept"] + feature_names
    coefs = stage2.int_coefs
    t_vals = np.where(np.isnan(se_clustered), np.nan, coefs / se_clustered)
    df_dof = n_clusters - 1
    p_vals = np.where(
        np.isnan(t_vals), np.nan,
        2 * sp_stats.t.sf(np.abs(t_vals), df=df_dof),
    )
    ci_crit = sp_stats.t.ppf(0.975, df=df_dof)

    def _sig(p):
        if np.isnan(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        if p < 0.1:
            return "."
        return ""

    summary = pd.DataFrame({
        "feature":      names,
        "coef":         coefs,
        "se_clustered": se_clustered,
        "t_stat":       t_vals,
        "p_value":      p_vals,
        "ci_lower":     coefs - ci_crit * se_clustered,
        "ci_upper":     coefs + ci_crit * se_clustered,
        "sig":          [_sig(p) for p in p_vals],
    })

    imp_feat_idx = [i for i, nm in enumerate(feature_names) if nm.startswith("impatience_spl_")]
    if imp_feat_idx and stage2._impatience_spline is not None:
        imp_vals = df_valid["impatience"].clip(*stage2._impatience_clip)
        imp_p50 = float(np.percentile(imp_vals, 50))
        imp_p95 = float(np.percentile(imp_vals, 95))

        X_lo = stage2._impatience_spline.transform([[imp_p50]])
        X_hi = stage2._impatience_spline.transform([[imp_p95]])
        d = (X_hi - X_lo).ravel()

        imp_coef_idx = [i + 1 for i in imp_feat_idx]
        partial_effect = float(d @ coefs[imp_coef_idx])

        XtX_full = X.T @ X
        if sparse.issparse(XtX_full):
            XtX_full = XtX_full.toarray()
        ridge = 1e-8 * np.trace(XtX_full) / max(X.shape[1], 1)
        XtX_full = XtX_full + ridge * np.eye(X.shape[1])
        XtX_inv_full = np.linalg.pinv(XtX_full)
        unique_clusters_full = np.unique(cluster_ids)
        meat = np.zeros((X.shape[1], X.shape[1]))
        for c in unique_clusters_full:
            mask = cluster_ids == c
            X_c = X[mask]
            e_c = residuals[mask]
            score_c = X_c.T @ e_c
            if sparse.issparse(X_c):
                score_c = np.asarray(score_c).ravel()
            meat += np.outer(score_c, score_c)
        adjustment = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - X.shape[1]))
        V_cluster = XtX_inv_full @ (adjustment * meat) @ XtX_inv_full
        V_imp = V_cluster[np.ix_(imp_coef_idx, imp_coef_idx)]
        partial_var = float(d @ V_imp @ d)
        partial_se = np.sqrt(max(partial_var, 0))
        t_agg = partial_effect / partial_se if partial_se > 0 else np.nan
        p_agg = float(2 * sp_stats.t.sf(np.abs(t_agg), df=df_dof)) if not np.isnan(t_agg) else np.nan

        agg_row = {
            "feature":      f"impatience (I-spline Δ p50={imp_p50:.4g}→p95={imp_p95:.4g})",
            "coef":         partial_effect,
            "se_clustered": partial_se,
            "t_stat":       t_agg,
            "p_value":      p_agg,
            "ci_lower":     partial_effect - ci_crit * partial_se,
            "ci_upper":     partial_effect + ci_crit * partial_se,
            "sig":          _sig(p_agg),
        }
        summary = pd.concat(
            [summary[~summary["feature"].str.startswith("impatience_spl_")], pd.DataFrame([agg_row])],
            ignore_index=True,
        )
        logger.info(
            "Impatience I-spline aggregated: Δlog(fee) p50=%.4g→p95=%.4g = %.4f (SE=%.4f, p=%.4f)",
            imp_p50, imp_p95, partial_effect, partial_se, p_agg,
        )

    valid_se = ~np.isnan(se_clustered) & ~np.isnan(se_naive) & (se_naive > 0)
    n_degenerate = int(np.sum(np.isnan(se_clustered)))

    logger.info("Clustered SEs computed — %d clusters, dof=%d", n_clusters, df_dof)
    if n_degenerate > 0:
        degen_names = [names[i] for i in range(len(names)) if np.isnan(se_clustered[i])]
        logger.info("  %d degenerate feature(s): %s", n_degenerate, degen_names)
    inflation = se_clustered[valid_se] / se_naive[valid_se]
    if len(inflation) > 0:
        logger.info("  Mean SE inflation (clustered/naive): %.2fx", inflation.mean())

    return summary


def compute_impatience_spline_effect(stage2, df: pd.DataFrame, n_grid: int = 300) -> pd.DataFrame:
    """Evaluate the fitted impatience I-spline effect over a grid."""
    if stage2._impatience_spline is None:
        raise RuntimeError("Stage 2 must be fitted before computing spline effects")

    valid_mask = (
        df["fee_rate"].notna()
        & (df["fee_rate"] > 0)
        & df["impatience"].notna()
    )
    df_valid = df.loc[valid_mask].copy()
    imp_vals = df_valid["impatience"].clip(*stage2._impatience_clip)
    grid = np.linspace(float(imp_vals.min()), float(imp_vals.max()), n_grid)
    basis = stage2._impatience_spline.transform(grid.reshape(-1, 1))

    coef_df = stage2.get_coefficient_summary()
    imp_mask = coef_df["feature"].str.startswith("impatience_spl_")
    imp_coefs = coef_df.loc[imp_mask, "intensive_coef"].values

    effect = basis @ imp_coefs
    out = pd.DataFrame({"impatience": grid, "spline_effect": effect})
    for j in range(basis.shape[1]):
        out[f"basis_{j}"] = basis[:, j]
        out[f"contribution_{j}"] = basis[:, j] * imp_coefs[j]
    return out


# ---------------------------------------------------------------------------
# 11. Class: HurdleFeeModel
# ---------------------------------------------------------------------------

class HurdleFeeModel:
    """Stage 2: Fee model with monotone I-spline impatience effect and epoch FE."""

    CONTROL_FEATURES = [
        "log_Wprime",
        "has_rbf", "is_cpfp_package", "log_total_output",
        "log_n_in", "log_n_out",
        "has_op_return", "has_inscription",
    ]
    STATE_FEATURES = [
        "blockspace_utilization",
        "log_time_since_last_block",
        "log_size",
    ]

    def __init__(
        self,
        config: EstimationConfig,
        use_epoch_fe: bool = True,
        use_ridge: bool = True,
        ridge_alpha: float = 1.0,
        quantile: Optional[float] = None,
    ):
        self.config    = config
        self.use_epoch_fe  = use_epoch_fe
        self.use_ridge     = use_ridge
        self.ridge_alpha   = ridge_alpha
        self.quantile      = quantile
        self.int_coefs     = None
        self.smearing_factor = None
        self.epoch_labels            = None
        self.epoch_labels_for_dummies = None
        self._n_epoch_dummies = 0
        self._feature_names   = None
        self._trim_threshold  = None
        self._impatience_spline = None
        self._impatience_clip = None
        self._ispline_col_start = None
        self._n_ispline_basis = 0
        self._is_fitted = False

    def _build_feature_matrix(
        self, df: pd.DataFrame, fit_spline: bool = False
    ) -> Tuple[Any, List[str]]:
        linear_dense = np.hstack([df[self.CONTROL_FEATURES].values, df[self.STATE_FEATURES].values])
        feature_names = list(self.CONTROL_FEATURES) + list(self.STATE_FEATURES)

        imp_raw = df[["impatience"]].values
        if fit_spline:
            lo, hi = np.percentile(imp_raw, [1, 99])
            self._impatience_clip = (float(lo), float(hi))
            imp_clipped = np.clip(imp_raw, lo, hi)
            self._impatience_spline = ISplineTransformer(
                n_knots=IMPATIENCE_SPLINE_N_KNOTS,
                degree=IMPATIENCE_SPLINE_DEGREE,
            )
            imp_basis = self._impatience_spline.fit_transform(imp_clipped)
            self._ispline_col_start = len(feature_names)
            self._n_ispline_basis = imp_basis.shape[1]
            logger.info(
                "Impatience I-spline: %d basis functions (clip=[%.4g, %.4g])",
                self._n_ispline_basis, lo, hi,
            )
        else:
            imp_clipped = np.clip(imp_raw, *self._impatience_clip)
            imp_basis = self._impatience_spline.transform(imp_clipped)

        imp_names = [f"impatience_spl_{i}" for i in range(imp_basis.shape[1])]
        feature_names += imp_names
        X_dense = np.hstack([linear_dense, imp_basis])

        if self.use_epoch_fe:
            if fit_spline:
                self.epoch_labels              = sorted(df["epoch_id"].unique())
                self.epoch_labels_for_dummies  = self.epoch_labels[1:]
                self._n_epoch_dummies          = len(self.epoch_labels_for_dummies)
                self._epoch_to_col = {e: i for i, e in enumerate(self.epoch_labels_for_dummies)}
                logger.info(
                    "Epoch FE: %d dummies from %d epochs",
                    self._n_epoch_dummies, len(self.epoch_labels),
                )

            epoch_ids = df["epoch_id"].values
            rows, cols = [], []
            for idx, eid in enumerate(epoch_ids):
                col = self._epoch_to_col.get(eid)
                if col is not None:
                    rows.append(idx)
                    cols.append(col)
            epoch_sparse = sparse.csr_matrix(
                (np.ones(len(rows), dtype=np.float64), (rows, cols)),
                shape=(len(df), self._n_epoch_dummies),
            )

            X = sparse.hstack([sparse.csr_matrix(X_dense), epoch_sparse], format="csr")
            feature_names = feature_names + [f"epoch_{e}" for e in self.epoch_labels_for_dummies]
        else:
            X = X_dense

        return X, feature_names

    def _make_coef_bounds(self, n_total: int) -> list:
        """Constrain I-spline coefficients to be nonnegative."""
        bounds = [(None, None)] * n_total
        if self._ispline_col_start is None or self._n_ispline_basis == 0:
            return bounds
        start = self._ispline_col_start + 1
        for j in range(self._n_ispline_basis):
            bounds[start + j] = (0.0, None)
        return bounds

    def fit(self, df: pd.DataFrame) -> "HurdleFeeModel":
        logger.info("=" * 60)
        logger.info(
            "STAGE 2: Fitting Fee Model%s",
            " (with Epoch FE)" if self.use_epoch_fe else "",
        )
        logger.info("=" * 60)

        required = ["fee_rate", "epoch_id", "impatience"] + self.CONTROL_FEATURES + self.STATE_FEATURES
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        all_feature_cols = ["fee_rate", "epoch_id"] + self.CONTROL_FEATURES + self.STATE_FEATURES
        valid_mask = pd.Series(True, index=df.index)
        for col in dict.fromkeys(all_feature_cols):
            valid_mask &= df[col].notna()
        df_valid = df[valid_mask].copy()
        logger.info("Fitting on %d valid observations (dropped %d with NaN)",
                     len(df_valid), (~valid_mask).sum())

        # Drop zero/negative fee_rate before log-transforming
        pos_mask = df_valid["fee_rate"] > 0
        n_dropped = (~pos_mask).sum()
        if n_dropped > 0:
            logger.warning("Dropping %d observations with fee_rate <= 0 (log undefined)", n_dropped)
        df_valid = df_valid[pos_mask].copy()

        # Trim bottom slope_trim fraction by log_Wprime (removes flat-plateau artefacts)
        slope_trim = self.config.slope_trim
        if slope_trim > 0:
            threshold = df_valid["log_Wprime"].quantile(slope_trim)
            self._trim_threshold = threshold
            trim_mask = df_valid["log_Wprime"] > threshold
            n_trimmed = (~trim_mask).sum()
            logger.info(
                "slope_trim=%.2f: dropping %d obs with log_Wprime <= %.4f",
                slope_trim, n_trimmed, threshold,
            )
            df_valid = df_valid[trim_mask].copy()

        X, self._feature_names = self._build_feature_matrix(df_valid, fit_spline=True)
        logger.info("Feature matrix shape: %s", X.shape)
        coef_bounds = self._make_coef_bounds(X.shape[1] + 1)
        n_nonneg = sum(1 for lo, _ in coef_bounds if lo is not None and lo >= 0)
        logger.info("Monotonicity bounds: %d I-spline coefficients constrained >= 0", n_nonneg)

        # Intensive margin on all transactions with positive fee_rate
        y_int = np.log(df_valid["fee_rate"].values)
        X_int_raw, _ = self._build_feature_matrix(df_valid, fit_spline=False)

        if self.quantile is not None:
            logger.info("Fitting intensive margin (quantile τ=%.2f)...", self.quantile)
            from sklearn.linear_model import QuantileRegressor
            qr = QuantileRegressor(
                quantile=self.quantile, alpha=0.0, fit_intercept=True, solver="highs"
            )
            qr.fit(X_int_raw, y_int)
            self.int_coefs = np.concatenate([[qr.intercept_], qr.coef_])
            self.smearing_factor = None
        else:
            logger.info("Fitting intensive margin (OLS)...")
            X_int = _prepend_intercept(X_int_raw)
            if self.use_ridge:
                logger.info("  Using Ridge (alpha=%.2f)", self.ridge_alpha)
                self.int_coefs = fit_regularized_ols(
                    X_int, y_int, alpha=self.ridge_alpha, bounds=coef_bounds
                )
            else:
                self.int_coefs = fit_regularized_ols(X_int, y_int, alpha=0.0, bounds=coef_bounds)

        X_int_with_intercept = _prepend_intercept(X_int_raw)
        y_pred_int = X_int_with_intercept @ self.int_coefs
        if sparse.issparse(y_pred_int):
            y_pred_int = np.asarray(y_pred_int).ravel()
        residuals = y_int - y_pred_int
        if self.quantile is None:
            self.smearing_factor = np.mean(np.exp(residuals))

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_int - y_int.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        logger.info("Fee model R²: %.4f", r2)
        if self.smearing_factor is not None:
            logger.info("Smearing factor: %.4f", self.smearing_factor)

        self._is_fitted = True
        logger.info("Stage 2 fitting complete")
        return self

    def predict_conditional_log_fee(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._build_feature_matrix(df, fit_spline=False)
        X_int = _prepend_intercept(X)
        result = X_int @ self.int_coefs
        if sparse.issparse(result):
            result = np.asarray(result).ravel()
        return result

    def predict_conditional_fee(self, df: pd.DataFrame) -> np.ndarray:
        log_fee = self.predict_conditional_log_fee(df)
        if self.smearing_factor is not None:
            return np.exp(log_fee) * self.smearing_factor
        return np.exp(log_fee)

    def get_coefficient_summary(self) -> pd.DataFrame:
        n_base = len(self._feature_names) - self._n_epoch_dummies
        base_names = ["intercept"] + self._feature_names[:n_base]
        n_coefs = len(base_names)
        return pd.DataFrame({
            "feature":        base_names,
            "intensive_coef": self.int_coefs[:n_coefs],
        })

    def get_epoch_effects(self) -> Optional[pd.DataFrame]:
        if not self._is_fitted or not self.use_epoch_fe:
            return None
        n_base = 1 + len(self._feature_names) - self._n_epoch_dummies
        epoch_coefs = self.int_coefs[n_base:]
        return pd.DataFrame({
            "epoch_id":   self.epoch_labels_for_dummies,
            "intensive_fe": epoch_coefs,
        })


# ---------------------------------------------------------------------------
# 12. Class: BitcoinFeeEstimator (orchestrator)
# ---------------------------------------------------------------------------

class BitcoinFeeEstimator:
    """Main orchestrator for the two-stage fee estimation pipeline."""

    def __init__(self, config: Optional[EstimationConfig] = None):
        self.config      = config or EstimationConfig()
        self.stage1      = None
        self.stage2      = None
        self.df_prepared = None
        self._is_fitted  = False

    def fit(self, df: pd.DataFrame) -> "BitcoinFeeEstimator":
        logger.info("=" * 60)
        logger.info("BITCOIN FEE ESTIMATION PIPELINE")
        logger.info("=" * 60)

        logger.info("[1/4] Preparing features...")
        self.df_prepared = prepare_features(df.copy(), **self.config.feature_params)

        logger.info("[2/4] Fitting Stage 1 (Delay Technology)...")
        self.stage1 = DelayTechnologyEstimator(self.config)
        self.stage1.fit(self.df_prepared)

        logger.info("[3/4] Computing Stage 1 outputs...")
        W_hat     = self.stage1.predict_W_hat(self.df_prepared)
        W_monotone = self.stage1.predict_delay_monotone_per_obs(
            self.df_prepared, W_hat, verbose=False
        )
        Wprime_hat = self.stage1.compute_slope_finite_diff_per_obs(
            self.df_prepared, W_monotone, delta=0.05, verbose=False
        )
        self.df_prepared["W_hat"]     = np.nan
        self.df_prepared["W_monotone"] = np.nan
        self.df_prepared["Wprime_hat"] = np.nan
        self.df_prepared.loc[self.stage1._valid_indices, "W_hat"]      = W_hat
        self.df_prepared.loc[self.stage1._valid_indices, "W_monotone"] = W_monotone
        self.df_prepared.loc[self.stage1._valid_indices, "Wprime_hat"] = Wprime_hat
        self.df_prepared["log_Wprime"] = np.log(
            self.df_prepared["Wprime_hat"].clip(lower=1e-6)
        )

        logger.info("[4/4] Fitting Stage 2 (Hurdle Fee Model)...")
        self.stage2 = HurdleFeeModel(self.config, use_ridge=False)
        self.stage2.fit(self.df_prepared)

        self._is_fitted = True
        logger.info("Pipeline fitting complete")
        return self

    def predict(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if df is None:
            df = self.df_prepared
        else:
            df = prepare_features(df.copy(), **self.config.feature_params)
            W_hat     = self.stage1.predict_W_hat(df)
            W_monotone = self.stage1.predict_delay_monotone_per_obs(df, W_hat, verbose=False)
            Wprime_hat = self.stage1.compute_slope_finite_diff_per_obs(
                df, W_monotone, delta=0.05, verbose=False
            )
            df["W_hat"]      = np.nan
            df["W_monotone"] = np.nan
            df["Wprime_hat"] = np.nan
            df.loc[self.stage1._valid_indices, "W_hat"]      = W_hat
            df.loc[self.stage1._valid_indices, "W_monotone"] = W_monotone
            df.loc[self.stage1._valid_indices, "Wprime_hat"] = Wprime_hat
            df["log_Wprime"] = np.log(df["Wprime_hat"].clip(lower=1e-6))

        valid_mask = df["log_Wprime"].notna()
        df_valid   = df[valid_mask].copy()

        in_support = pd.Series(True, index=df_valid.index)
        if self.stage2._trim_threshold is not None:
            in_support = df_valid["log_Wprime"] > self.stage2._trim_threshold

        return pd.DataFrame({
            "tx_id":             df_valid["tx_id"],
            "actual_fee_rate":   df_valid["fee_rate"],
            "W_hat":             df_valid["W_hat"],
            "Wprime_hat":        df_valid["Wprime_hat"],
            "log_fee_hat":       self.stage2.predict_conditional_log_fee(df_valid),
            "fee_hat":           self.stage2.predict_conditional_fee(df_valid),
            "in_model_support":  in_support,
        })

    def summary(self) -> Dict[str, Any]:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting summary")
        return {
            "config": {
                "epoch_duration_minutes": self.config.epoch_duration_minutes,
                "n_folds":                self.config.n_folds,
                "fee_threshold":          self.config.fee_threshold_sat_vb,
            },
            "data": {
                "n_observations": len(self.df_prepared),
                "n_epochs":       self.df_prepared["epoch_id"].nunique(),
                "time_range":     (
                    str(self.df_prepared["found_at"].min()),
                    str(self.df_prepared["found_at"].max()),
                ),
            },
            "stage1": {
                "model_type": self.config.stage1_model,
                "n_valid":    len(self.stage1._valid_indices),
            },
            "stage2": {
                "smearing_factor": self.stage2.smearing_factor,
                "n_features":      len(self.stage2._feature_names),
                "n_impatience_spline_basis": self.stage2._n_ispline_basis,
                "impatience_clip": self.stage2._impatience_clip,
            },
        }

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        joblib.dump({"config": self.config, "stage1": self.stage1, "stage2": self.stage2}, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BitcoinFeeEstimator":
        data = joblib.load(path)
        estimator = cls(data["config"])
        estimator.stage1    = data["stage1"]
        estimator.stage2    = data["stage2"]
        estimator._is_fitted = True
        logger.info("Model loaded from %s", path)
        return estimator


# ---------------------------------------------------------------------------
# 14. Export / diagnostic helpers
# ---------------------------------------------------------------------------

def save_plots(
    estimator: BitcoinFeeEstimator,
    predictions: pd.DataFrame,
    output_dir: str,
    se_summary: Optional[pd.DataFrame] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = estimator.df_prepared
    stage1 = estimator.stage1
    stage2 = estimator.stage2

    sample_size = min(10000, len(predictions))
    sample = predictions.sample(sample_size, random_state=42)
    df_sample = df.loc[sample.index] if len(df) > sample_size else df

    def _save(fig, name):
        p = os.path.join(plots_dir, name)
        fig.tight_layout()
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved: %s", p)

    # =================================================================
    # 1. Stage 1: Actual vs Predicted Delay
    # =================================================================
    fig, ax = plt.subplots(figsize=(7, 6))
    valid_mask = df_sample["W_hat"].notna() & df_sample["log_waittime"].notna()
    if valid_mask.sum() > 0:
        ax.scatter(
            df_sample.loc[valid_mask, "log_waittime"],
            df_sample.loc[valid_mask, "W_hat"],
            alpha=0.1, s=1,
        )
        ax.plot([0, 15], [0, 15], "r--", lw=2, label="45° line")
        ax.set_xlabel("Actual log(waittime)")
        ax.set_ylabel("Predicted log(delay)")
        ax.set_title("Stage 1: Actual vs Predicted Delay")
        ax.legend()
    _save(fig, "stage1_actual_vs_predicted.png")

    # =================================================================
    # 1b. Fee Rate vs Wait Time (convexity)
    # =================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    valid_mask = (
        df["fee_rate"].notna()
        & df["waittime"].notna()
        & (df["fee_rate"] > 0)
        & (df["waittime"] >= 0)
        & df["mempool_tx_count"].notna()
    )
    if valid_mask.sum() > 0:
        ds = df[valid_mask].copy()
        ds["waittime_minutes"] = ds["waittime"] / 60.0

        # --- Left panel: overall relationship with binned medians ---
        ax0 = axes[0]

        # Scatter subsample for visual clarity
        scatter_n = min(20000, len(ds))
        ds_scatter = ds.sample(scatter_n, random_state=42)
        ax0.scatter(
            ds_scatter["fee_rate"], ds_scatter["waittime_minutes"],
            alpha=0.04, s=1, c="#888888", zorder=1,
        )

        # Binned medians on a log-spaced fee_rate grid
        fee_min = max(ds["fee_rate"].quantile(0.01), 0.1)
        fee_max = ds["fee_rate"].quantile(0.99)
        bin_edges = np.geomspace(fee_min, fee_max, 31)
        ds["fee_bin"] = pd.cut(ds["fee_rate"], bins=bin_edges, labels=False)
        bin_stats = ds.groupby("fee_bin", observed=True).agg(
            fee_median=("fee_rate", "median"),
            wait_median=("waittime_minutes", "median"),
            wait_q25=("waittime_minutes", lambda x: x.quantile(0.25)),
            wait_q75=("waittime_minutes", lambda x: x.quantile(0.75)),
            n=("waittime_minutes", "size"),
        )
        bin_stats = bin_stats[bin_stats["n"] >= 20]

        ax0.plot(
            bin_stats["fee_median"], bin_stats["wait_median"],
            "o-", color="#b2182b", markersize=4, linewidth=2,
            label="Binned median", zorder=3,
        )
        ax0.fill_between(
            bin_stats["fee_median"],
            bin_stats["wait_q25"], bin_stats["wait_q75"],
            alpha=0.15, color="#b2182b", label="IQR", zorder=2,
        )

        ax0.set_xscale("log")
        ax0.set_xlabel("Fee rate (sat/vB, log scale)")
        ax0.set_ylabel("Wait time (minutes)")
        ax0.set_title("Fee Rate vs Wait Time — Convexity")
        ax0.legend(loc="upper right", fontsize=9)

        # --- Right panel: split by congestion regime ---
        ax1 = axes[1]
        cong_median = ds["mempool_tx_count"].median()
        regimes = [
            (ds["mempool_tx_count"] <= cong_median, "Low congestion", "#2166ac"),
            (ds["mempool_tx_count"] > cong_median,  "High congestion", "#b2182b"),
        ]
        for mask, label, color in regimes:
            sub = ds[mask]
            sub_bin = sub.groupby(
                pd.cut(sub["fee_rate"], bins=bin_edges, labels=False), observed=True
            ).agg(
                fee_median=("fee_rate", "median"),
                wait_median=("waittime_minutes", "median"),
                n=("waittime_minutes", "size"),
            )
            sub_bin = sub_bin[sub_bin["n"] >= 20]
            if len(sub_bin) > 0:
                ax1.plot(
                    sub_bin["fee_median"], sub_bin["wait_median"],
                    "o-", color=color, markersize=4, linewidth=2, label=label,
                )

        ax1.set_xscale("log")
        ax1.set_xlabel("Fee rate (sat/vB, log scale)")
        ax1.set_ylabel("Wait time (minutes)")
        ax1.set_title("Fee Rate vs Wait Time — by Congestion")
        ax1.legend(loc="upper right", fontsize=9)

    _save(fig, "feerate_vs_waittime_convexity.png")

    # =================================================================
    # 2. Stage 1: Delay vs Priority by Congestion
    # =================================================================
    fig, ax = plt.subplots(figsize=(7, 6))
    delay_col = "W_monotone" if "W_monotone" in df_sample.columns else "W_hat"
    valid_mask = (
        df_sample[delay_col].notna()
        & df_sample["fee_rate_percentile"].notna()
        & df_sample["mempool_tx_count"].notna()
    )
    if valid_mask.sum() > 0:
        df_plot = df_sample[valid_mask].copy()
        congestion_median = df_plot["mempool_tx_count"].median()
        high_cong = df_plot["mempool_tx_count"] > congestion_median
        low_cong = ~high_cong

        ax.scatter(
            df_plot.loc[low_cong, "fee_rate_percentile"],
            df_plot.loc[low_cong, delay_col],
            alpha=0.1, s=1, c="blue", label="Low congestion",
        )
        ax.scatter(
            df_plot.loc[high_cong, "fee_rate_percentile"],
            df_plot.loc[high_cong, delay_col],
            alpha=0.1, s=1, c="red", label="High congestion",
        )

        for mask, color in [(low_cong, "blue"), (high_cong, "red")]:
            if mask.sum() > 100:
                bins = pd.cut(df_plot.loc[mask, "fee_rate_percentile"], bins=10, labels=False)
                bin_means = df_plot.loc[mask].groupby(bins)[delay_col].mean()
                bin_centers = df_plot.loc[mask].groupby(bins)["fee_rate_percentile"].mean()
                ax.plot(bin_centers, bin_means, color=color, linewidth=2, marker="o", markersize=4)

        ax.set_xlabel("Priority (percentile)")
        ax.set_ylabel(f"Predicted Delay ({delay_col})")
        ax.set_title("Stage 1: Delay vs Priority by Congestion")
        ax.legend(loc="upper right", fontsize=8, markerscale=8)
    _save(fig, "stage1_delay_vs_priority.png")

    # =================================================================
    # 3. Stage 1: Local Slope Distribution
    # =================================================================
    fig, ax = plt.subplots(figsize=(7, 6))
    valid_mask = df_sample["Wprime_hat"].notna()
    if valid_mask.sum() > 0:
        log_wp = np.log(df_sample.loc[valid_mask, "Wprime_hat"].clip(lower=1e-10))
        in_sup = sample.loc[valid_mask, "in_model_support"] if "in_model_support" in sample.columns else None
        if in_sup is not None:
            ax.hist(log_wp[~in_sup], bins=50, edgecolor="black", alpha=0.4,
                    color="#999999", label=f"Trimmed ({(~in_sup).sum():,})")
            ax.hist(log_wp[in_sup], bins=50, edgecolor="black", alpha=0.7,
                    color="#2166ac", label=f"In support ({in_sup.sum():,})")
            ax.legend(fontsize=9)
        else:
            ax.hist(log_wp, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("log(W')")
        ax.set_ylabel("Frequency")
        ax.set_title("Stage 1: Local Slope Distribution")
    _save(fig, "stage1_slope_distribution.png")

    # =================================================================
    # 4. Stage 1: Feature Importance
    # =================================================================
    if stage1.model is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        importances = stage1.model.feature_importances_
        idx = np.argsort(importances)[::-1]
        ax.barh(range(len(importances)), importances[idx])
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([stage1.FEATURES[i] for i in idx])
        ax.set_xlabel("Importance")
        ax.set_title("Stage 1: Feature Importance")
        _save(fig, "stage1_feature_importance.png")

    # =================================================================
    # 5. Stage 2: Predicted vs Actual
    # =================================================================
    fig, ax = plt.subplots(figsize=(7, 6))
    pos_mask = sample["actual_fee_rate"] > 0
    if pos_mask.sum() > 0:
        actual = np.log(sample.loc[pos_mask, "actual_fee_rate"])
        predicted = sample.loc[pos_mask, "log_fee_hat"]
        in_sup = sample.loc[pos_mask, "in_model_support"] if "in_model_support" in sample.columns else None
        if in_sup is not None:
            ax.scatter(actual[~in_sup], predicted[~in_sup], alpha=0.06, s=1,
                       c="#999999", label=f"Extrapolated ({(~in_sup).sum():,})", zorder=1)
            ax.scatter(actual[in_sup], predicted[in_sup], alpha=0.15, s=1,
                       c="#2166ac", label=f"In support ({in_sup.sum():,})", zorder=2)
        else:
            ax.scatter(actual, predicted, alpha=0.1, s=1, zorder=1)
        lo = min(actual.quantile(0.001), predicted.quantile(0.001))
        hi = max(actual.quantile(0.999), predicted.quantile(0.999))
        ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="45° line", zorder=3)
        ax.set_xlabel("Actual log(fee_rate)")
        ax.set_ylabel("Predicted log(fee_rate)")
        ax.set_title("Stage 2: Predicted vs Actual")
        ax.legend(loc="upper left", fontsize=8, markerscale=5)
    _save(fig, "stage2_predicted_vs_actual.png")

    # =================================================================
    # 7. Stage 2: Delay Gradient Effect
    # =================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_mask = (
        df_sample["log_Wprime"].notna()
        & df_sample["log_fee_rate"].notna()
    )
    if valid_mask.sum() > 0:
        ds = df_sample[valid_mask].copy()
        in_sup = sample.loc[valid_mask, "in_model_support"] if "in_model_support" in sample.columns else None
        if in_sup is not None:
            ax.scatter(
                ds.loc[~in_sup.values, "log_Wprime"], ds.loc[~in_sup.values, "log_fee_rate"],
                alpha=0.06, s=1, c="#999999", label=f"Trimmed ({(~in_sup).sum():,})",
            )
            ds_sup = ds[in_sup.values]
            ax.scatter(
                ds_sup["log_Wprime"], ds_sup["log_fee_rate"],
                alpha=0.15, s=1, c="#2166ac", label=f"In support ({in_sup.sum():,})",
            )
            if len(ds_sup) > 100:
                bins = pd.qcut(ds_sup["log_Wprime"], q=20, duplicates="drop")
                bin_means_y = ds_sup.groupby(bins, observed=True)["log_fee_rate"].mean()
                bin_means_x = ds_sup.groupby(bins, observed=True)["log_Wprime"].mean()
                ax.plot(bin_means_x, bin_means_y, "o-", color="#b2182b",
                        markersize=4, linewidth=2, label="Binned mean (in support)")
        else:
            ax.scatter(
                ds["log_Wprime"], ds["log_fee_rate"],
                alpha=0.10, s=1, c="#2166ac", label=f"N = {len(ds):,}",
            )
            if len(ds) > 100:
                bins = pd.qcut(ds["log_Wprime"], q=20, duplicates="drop")
                bin_means_y = ds.groupby(bins, observed=True)["log_fee_rate"].mean()
                bin_means_x = ds.groupby(bins, observed=True)["log_Wprime"].mean()
                ax.plot(bin_means_x, bin_means_y, "o-", color="#b2182b",
                        markersize=4, linewidth=2, label="Binned mean")

        ax.set_xlabel("log(W')")
        ax.set_ylabel("log(fee_rate)")
        ax.set_title("Delay Gradient vs Fee Rate")
        ax.legend(loc="upper left", fontsize=8, markerscale=5)
    _save(fig, "stage2_delay_gradient_effect.png")

    # =================================================================
    # 7b. Delay Gradient Distribution
    # =================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    valid_mask = df_sample["log_Wprime"].notna()
    if valid_mask.sum() > 0:
        ds = df_sample[valid_mask].copy()
        in_sup = sample.loc[valid_mask, "in_model_support"] if "in_model_support" in sample.columns else None

        ax0 = axes[0]
        if in_sup is not None:
            ds_sup = ds[in_sup.values]
            n_trimmed = (~in_sup).sum()
            ax0.hist(ds_sup["log_Wprime"].dropna(), bins=50,
                     edgecolor="black", alpha=0.7, color="#2166ac",
                     label=f"In support ({in_sup.sum():,})")
            ax0.set_title(f"Distribution of log(W') — In Support\n"
                          f"({n_trimmed:,} trimmed observations excluded)")
            ax0.legend(fontsize=8)
        else:
            ax0.hist(ds["log_Wprime"].dropna(), bins=50, edgecolor="black",
                     alpha=0.7, color="#2166ac")
            ax0.axvline(ds["log_Wprime"].median(), color="#b2182b", ls="--",
                        label=f"Median = {ds['log_Wprime'].median():.2f}")
            ax0.set_title("Distribution of log(W')")
            ax0.legend(fontsize=9)
        ax0.set_xlabel("log(W')")
        ax0.set_ylabel("Count")

        ax1 = axes[1]
        if in_sup is not None:
            ds_sup = ds[in_sup.values]
            fee_valid = ds_sup["log_fee_rate"].notna()
        else:
            ds_sup = ds
            fee_valid = ds_sup["log_fee_rate"].notna()
        if fee_valid.sum() > 100:
            ax1.hexbin(ds_sup.loc[fee_valid, "log_Wprime"],
                       ds_sup.loc[fee_valid, "log_fee_rate"],
                       gridsize=30, cmap="Blues", mincnt=1)
            ax1.set_xlabel("log(W')")
            ax1.set_ylabel("log(fee_rate)")
            ax1.set_title("log(W') vs log(fee_rate) — In Support Only")
            plt.colorbar(ax1.collections[0], ax=ax1, label="Count")

    _save(fig, "stage2_slope_regime_summary.png")

    # =================================================================
    # 8. Stage 2: Residual Distribution
    # =================================================================
    fig, ax = plt.subplots(figsize=(7, 6))
    pos_sample = sample[sample["actual_fee_rate"] > 0]
    if len(pos_sample) > 0:
        residuals = np.log(pos_sample["actual_fee_rate"].values) - pos_sample["log_fee_hat"].values
        residuals = residuals[np.isfinite(residuals)]
        q_lo, q_hi = np.percentile(residuals, [0.5, 99.5])
        residuals_trimmed = residuals[(residuals >= q_lo) & (residuals <= q_hi)]
        ax.hist(residuals_trimmed, bins=50, edgecolor="black", alpha=0.7, density=True)
        x_grid = np.linspace(q_lo, q_hi, 100)
        ax.plot(x_grid, stats.norm.pdf(x_grid, residuals_trimmed.mean(), residuals_trimmed.std()), "r-", lw=2)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Density")
        ax.set_title("Stage 2: Residual Distribution")
    _save(fig, "stage2_residual_distribution.png")

    # =================================================================
    # 9. Stage 2: Residuals vs Fitted
    # =================================================================
    fig, ax = plt.subplots(figsize=(7, 6))
    if len(pos_sample) > 0:
        residuals = np.log(pos_sample["actual_fee_rate"].values) - pos_sample["log_fee_hat"].values
        ax.scatter(pos_sample["log_fee_hat"], residuals, alpha=0.1, s=1)
        ax.axhline(y=0, color="r", linestyle="--")
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Stage 2: Residuals vs Fitted")
    _save(fig, "stage2_residuals_vs_fitted.png")

    # =================================================================
    # 10. Stage 2: Impatience I-spline effect
    # =================================================================
    if stage2._impatience_spline is not None:
        effect_df = compute_impatience_spline_effect(stage2, df)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(effect_df["impatience"], effect_df["spline_effect"], color="#2166ac", lw=2.5)
        ax.set_xlabel("Impatience")
        ax.set_ylabel("s(iota)")
        ax.set_title("Stage 2: Aggregate Impatience I-spline Effect")

        ax = axes[1]
        contrib_cols = [c for c in effect_df.columns if c.startswith("contribution_")]
        for col in contrib_cols:
            ax.plot(effect_df["impatience"], effect_df[col], lw=1.2, alpha=0.8, label=col)
        ax.plot(
            effect_df["impatience"], effect_df["spline_effect"],
            color="black", lw=2.2, label="aggregate",
        )
        ax.set_xlabel("Impatience")
        ax.set_ylabel("Contribution to log(fee_rate)")
        ax.set_title("Stage 2: I-spline Basis Contributions")
        ax.legend(fontsize=7, loc="upper left")
        _save(fig, "stage2_impatience_spline_effect.png")

    # =================================================================
    # Stage 2 coefficient CI plot
    # =================================================================
    # Human-readable names for the coefficient CI plot
    DISPLAY_NAMES = {
        "log_Wprime":               "log(W')",
        "has_rbf":                  "has_rbf (0/1)",
        "is_cpfp_package":          "is_cpfp_package (0/1)",
        "log_total_output":         "log(total_output)",
        "log_n_in":                 "log(n_inputs)",
        "log_n_out":                "log(n_outputs)",
        "has_op_return":            "has_op_return (0/1)",
        "has_inscription":          "has_inscription (0/1)",
        "blockspace_utilization":   "blockspace_utilization",
        "log_time_since_last_block": "log(time_since_last_block)",
        "log_size":                 "log(mempool_size)",
    }

    if se_summary is not None:
        base_summary = se_summary[
            (se_summary["feature"] != "intercept")
            & ~se_summary["feature"].str.startswith("epoch_")
            & ~se_summary["feature"].str.startswith("impatience_spl_")
            & ~se_summary["feature"].str.contains("impatience", case=False, na=False)
        ].copy()
        plot_df = base_summary[
            base_summary["se_clustered"].notna()
        ].copy()
        plot_df = plot_df.sort_values("coef", ascending=True).reset_index(drop=True)

        if len(plot_df) > 0:
            fig3, ax3 = plt.subplots(figsize=(8, max(4, 0.6 * len(plot_df))))
            y_pos = np.arange(len(plot_df))
            colors = ["#2166ac" if c > 0 else "#b2182b" for c in plot_df["coef"]]

            ax3.barh(y_pos, plot_df["coef"], height=0.6, color=colors, alpha=0.7, zorder=2)
            ax3.errorbar(
                plot_df["coef"], y_pos,
                xerr=np.vstack([
                    plot_df["coef"] - plot_df["ci_lower"],
                    plot_df["ci_upper"] - plot_df["coef"],
                ]),
                fmt="none", ecolor="black", elinewidth=1.2, capsize=3, zorder=3,
            )
            ax3.axvline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)

            labels = []
            for _, row in plot_df.iterrows():
                name = DISPLAY_NAMES.get(row["feature"], row["feature"])
                stars = row["sig"]
                labels.append(f"{name} {stars}" if stars else name)

            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(labels, fontsize=10)
            ax3.set_xlabel("Coefficient (with 95% CI, epoch-clustered SEs)", fontsize=11)
            ax3.set_title("Stage 2 \u2014 Coefficient Estimates", fontsize=13)
            ax3.grid(axis="x", alpha=0.3)

            fig3.tight_layout()
            path3 = os.path.join(plots_dir, "stage2_coefficient_ci.png")
            fig3.savefig(path3, dpi=150, bbox_inches="tight")
            plt.close(fig3)
            logger.info("Plot saved: %s", path3)


def export_results(
    estimator: BitcoinFeeEstimator,
    predictions: pd.DataFrame,
    output_dir: str,
    skip_plots: bool = False,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths: Dict[str, str] = {}

    model_path = os.path.join(output_dir, f"fee_estimator_{timestamp}.pkl")
    estimator.save(model_path)
    paths["model"] = model_path

    pred_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
    predictions.to_csv(pred_path, index=False)
    paths["predictions"] = pred_path
    logger.info("Predictions saved to: %s", pred_path)

    coef_path = os.path.join(output_dir, f"coefficients_{timestamp}.csv")
    estimator.stage2.get_coefficient_summary().to_csv(coef_path, index=False)
    paths["coefficients"] = coef_path
    logger.info("Coefficients saved to: %s", coef_path)

    if estimator.stage2._impatience_spline is not None:
        imp_effect_path = os.path.join(output_dir, f"impatience_spline_effect_{timestamp}.csv")
        compute_impatience_spline_effect(estimator.stage2, estimator.df_prepared).to_csv(
            imp_effect_path, index=False
        )
        paths["impatience_spline_effect"] = imp_effect_path
        logger.info("Impatience spline effect saved to: %s", imp_effect_path)

    summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write("Bitcoin Fee Estimation Pipeline - Summary\n")
        f.write("=" * 50 + "\n\n")
        for section, values in estimator.summary().items():
            f.write(f"{section}:\n")
            if isinstance(values, dict):
                for k, v in values.items():
                    f.write(f"  {k}: {v}\n")
            f.write("\n")
    paths["summary"] = summary_path
    logger.info("Summary saved to: %s", summary_path)

    se_summary = compute_intensive_margin_se(estimator.stage2, estimator.df_prepared)
    se_path = os.path.join(output_dir, f"clustered_se_{timestamp}.csv")
    se_summary.to_csv(se_path, index=False)
    paths["clustered_se"] = se_path
    logger.info("Clustered SEs saved to: %s", se_path)

    if not skip_plots:
        save_plots(estimator, predictions, output_dir, se_summary=se_summary)

    return paths


# ---------------------------------------------------------------------------
# 15. main()
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bitcoin fee estimation pipeline (VCG two-stage structural model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        default=EstimationConfig.db_path,
        help="Path to source SQLite database",
    )
    parser.add_argument(
        "--annotation-path",
        default=EstimationConfig.annotation_path,
        help="Path to transaction annotations pickle",
    )
    parser.add_argument(
        "--block-limit", type=int, default=None,
        help="Number of blocks to sample (None = all available)",
    )
    parser.add_argument(
        "--epoch-mode", default="time", choices=["time", "block", "fixed_size"],
        help="Epoch assignment strategy",
    )
    parser.add_argument(
        "--epoch-duration", type=int, default=30,
        help="Minutes per epoch (only for --epoch-mode time)",
    )
    parser.add_argument(
        "--stage1-model",
        default="rf",
        choices=["rf", "xgb_monotone"],
        help="First-stage learner: random forest with post-hoc isotonic, or monotone XGBoost",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Cross-validation folds",
    )
    parser.add_argument(
        "--rf-estimators", type=int, default=200,
        help="Number of trees in Stage 1 Random Forest",
    )
    parser.add_argument(
        "--rf-max-depth", type=int, default=15,
        help="Max depth of Stage 1 Random Forest",
    )
    parser.add_argument(
        "--xgb-estimators", type=int, default=400,
        help="Number of trees in Stage 1 monotone XGBoost",
    )
    parser.add_argument(
        "--xgb-max-depth", type=int, default=6,
        help="Max depth of Stage 1 monotone XGBoost",
    )
    parser.add_argument(
        "--xgb-learning-rate", type=float, default=0.05,
        help="Learning rate for Stage 1 monotone XGBoost",
    )
    parser.add_argument(
        "--xgb-subsample", type=float, default=0.8,
        help="Row subsample rate for Stage 1 monotone XGBoost",
    )
    parser.add_argument(
        "--xgb-colsample-bytree", type=float, default=0.8,
        help="Column subsample rate for Stage 1 monotone XGBoost",
    )
    parser.add_argument(
        "--xgb-min-child-weight", type=float, default=10.0,
        help="Minimum child weight for Stage 1 monotone XGBoost",
    )
    parser.add_argument(
        "--output-dir", default="model_outputs",
        help="Directory for output files",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory to save/load stage checkpoints for resumability",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip matplotlib output",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--temporal-variance", action="store_true",
        help="Run temporal variance analysis after the main pipeline",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Path to a pre-built Parquet dataset (skips all data loading)",
    )
    parser.add_argument(
        "--mempool-db",
        default="/home/kristian/notebooks/mempool_space_data.db",
        help="Path to mempool.space SQLite database",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = EstimationConfig(
        db_path=args.db_path,
        annotation_path=args.annotation_path,
        block_limit=args.block_limit,
        epoch_mode=args.epoch_mode,
        epoch_duration_minutes=args.epoch_duration,
        stage1_model=args.stage1_model,
        n_folds=args.n_folds,
        rf_n_estimators=args.rf_estimators,
        rf_max_depth=args.rf_max_depth,
        xgb_n_estimators=args.xgb_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_min_child_weight=args.xgb_min_child_weight,
    )

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_features_path = os.path.join(ckpt_dir, CHECKPOINT_FEATURES) if ckpt_dir else None
    ckpt_stage1_path = (
        os.path.join(ckpt_dir, f"checkpoint_stage1_{config.stage1_model}.pkl")
        if ckpt_dir else None
    )

    # ------------------------------------------------------------------
    # Step 1: Load or build feature DataFrame
    # ------------------------------------------------------------------
    if args.dataset:
        logger.info("Loading pre-built dataset from %s", args.dataset)
        df = pd.read_parquet(args.dataset)
        logger.info("Loaded %d transactions from dataset", len(df))
    elif ckpt_features_path and os.path.exists(ckpt_features_path):
        logger.info("Loading features from checkpoint: %s", ckpt_features_path)
        with open(ckpt_features_path, "rb") as f:
            df = pickle.load(f)
        logger.info("Checkpoint loaded: %d transactions", len(df))
    else:
        logger.info("Building dataset from raw sources...")
        df = build_analysis_dataset(
            db_path=config.db_path,
            annotation_path=config.annotation_path,
            block_limit=config.block_limit,
            mempool_space_db=args.mempool_db,
            **config.feature_params,
        )

        if ckpt_features_path:
            logger.info("Saving feature checkpoint to %s", ckpt_features_path)
            with open(ckpt_features_path, "wb") as f:
                pickle.dump(df, f)

    logger.info("Dataset ready: %d transactions", len(df))

    # ------------------------------------------------------------------
    # Step 2: Stage 1 — load from checkpoint or fit
    # ------------------------------------------------------------------
    if ckpt_stage1_path and os.path.exists(ckpt_stage1_path):
        logger.info("Loading Stage 1 from checkpoint: %s", ckpt_stage1_path)
        with open(ckpt_stage1_path, "rb") as f:
            stage1 = pickle.load(f)
    else:
        stage1 = DelayTechnologyEstimator(config)
        stage1.fit(df)
        if ckpt_stage1_path:
            logger.info("Saving Stage 1 checkpoint to %s", ckpt_stage1_path)
            with open(ckpt_stage1_path, "wb") as f:
                pickle.dump(stage1, f)

    # Compute Stage 1 outputs
    logger.info("Computing Stage 1 outputs (monotone + finite diff slopes)...")
    W_hat      = stage1.predict_W_hat(df)
    W_monotone = stage1.predict_delay_monotone_per_obs(df, W_hat)
    Wprime_hat = stage1.compute_slope_finite_diff_per_obs(df, W_monotone, delta=0.05)

    df["W_hat"]      = np.nan
    df["W_monotone"] = np.nan
    df["Wprime_hat"] = np.nan
    df.loc[stage1._valid_indices, "W_hat"]      = W_hat
    df.loc[stage1._valid_indices, "W_monotone"] = W_monotone
    df.loc[stage1._valid_indices, "Wprime_hat"] = Wprime_hat
    df["log_Wprime"] = np.log(df["Wprime_hat"].clip(lower=1e-6))

    # ------------------------------------------------------------------
    # Step 3: Stage 2 — always fit (fast)
    # ------------------------------------------------------------------
    stage2 = HurdleFeeModel(config, use_ridge=False)
    stage2.fit(df)

    # ------------------------------------------------------------------
    # Step 4: Assemble estimator, get predictions, export
    # ------------------------------------------------------------------
    estimator              = BitcoinFeeEstimator(config)
    estimator.stage1       = stage1
    estimator.stage2       = stage2
    estimator.df_prepared  = df
    estimator._is_fitted   = True

    logger.info("Generating predictions...")
    predictions = estimator.predict()
    logger.info("Predictions generated: %d rows", len(predictions))

    output_paths = export_results(estimator, predictions, args.output_dir, args.skip_plots)
    logger.info("Output files:")
    for key, path in output_paths.items():
        logger.info("  %s: %s", key, path)

    # ------------------------------------------------------------------
    # Optional: Temporal Variance Analysis
    # ------------------------------------------------------------------
    if args.temporal_variance:
        from temporal_variance_analysis import (
            run_temporal_variance_analysis,
            plot_temporal_variance_analysis,
        )
        logger.info("Running temporal variance analysis...")
        tva_results = run_temporal_variance_analysis(df, stage2, config)
        plot_dir = str(Path(args.output_dir) / "plots")
        tva_fig = plot_temporal_variance_analysis(tva_results, config, output_dir=plot_dir)
        logger.info("Temporal variance analysis complete: %s", tva_fig)

    logger.info("Pipeline complete.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

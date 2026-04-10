#!/usr/bin/env python3
"""
Dataset builder for the Bitcoin fee estimation pipeline.

Loads raw transaction data from SQLite, merges mempool.space weight/fee
corrections and transaction annotations, collapses CPFP packages, and
engineers features for the two-stage structural model.

Usage:
    # Build and export the analysis-ready dataset
    python build_dataset.py \\
        --db-path /path/to/source.db \\
        --annotation-path /path/to/annotations.pkl \\
        --output data/analysis_ready.parquet

    # With block limit and custom epochs
    python build_dataset.py \\
        --db-path /path/to/source.db \\
        --annotation-path /path/to/annotations.pkl \\
        --block-limit 1000 --epoch-duration 30 \\
        --output data/analysis_ready.parquet

    # Load an existing dataset (verify / re-export)
    python build_dataset.py --load data/analysis_ready.parquet --info
"""

import argparse
import gc
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MEMPOOL_SPACE_DB = "/home/kristian/notebooks/mempool_space_data.db"


# ---------------------------------------------------------------------------
# Column descriptions for metadata export
# ---------------------------------------------------------------------------

COLUMN_DESCRIPTIONS = {
    # Identifiers
    "tx_id": "Transaction ID (hex string)",
    "conf_block_hash": "Block hash where transaction was confirmed",

    # Timestamps
    "found_at": "Timestamp when transaction was first seen in mempool",
    "mined_at": "Timestamp when transaction was confirmed in a block",

    # Core transaction attributes (mempool.space-corrected, CPFP-adjusted)
    "fee_rate": "Fee rate in sat/vB (corrected from mempool.space; CPFP package rate for parents)",
    "fee_rate_original": "Original fee rate from source database before correction",
    "weight": "Transaction weight in WU (corrected; CPFP package weight for parents)",
    "weight_original": "Original weight from source database",
    "size": "Transaction size in bytes (corrected)",
    "size_original": "Original size from source database",
    "vsize": "Virtual size in vB (corrected; CPFP package vsize for parents)",
    "absolute_fee": "Total fee in satoshis (corrected)",
    "absolute_fee_original": "Original fee from source database",
    "has_corrected_weight": "Whether mempool.space correction was available (bool)",

    # Wait time and mempool state
    "waittime": "Seconds between first mempool observation and block confirmation",
    "min_respend_blocks": "Minimum blocks until any input can be re-spent (-1 if unknown)",
    "mempool_tx_count": "Number of transactions in mempool when tx was observed",
    "mempool_size": "Mempool size in bytes when tx was observed",
    "total_output_amount": "Total output value in satoshis",

    # RBF / CPFP
    "child_txid": "CPFP child transaction ID (if this tx is a CPFP parent)",
    "rbf_fee_total": "Total fee paid via RBF replacement (if applicable)",
    "has_rbf": "1 if transaction used replace-by-fee",
    "is_cpfp_package": "1 if tx is a CPFP parent (fee/weight reflect package totals)",

    # Annotations
    "label": "Transaction type (consolidation, coinjoin, datacarrying, simple, etc.)",
    "n_in": "Number of inputs",
    "n_out": "Number of outputs",
    "has_op_return": "1 if transaction contains OP_RETURN output",
    "has_inscription": "1 if transaction contains an ordinals inscription",

    # Block state features (from mempool.space block data)
    "block_median_feerate": "Median fee rate in the confirmation block (sat/vB)",
    "block_mean_feerate": "Mean fee rate in the confirmation block",
    "block_p10_feerate": "10th percentile fee rate in the confirmation block",
    "block_p90_feerate": "90th percentile fee rate in the confirmation block",
    "block_weight": "Total weight of the confirmation block",
    "block_n_tx": "Number of transactions in the confirmation block",
    "time_since_last_block": "Seconds since the previous block",
    "ema_feerate_3block": "3-block exponential moving average of median fee rate",
    "ema_feerate_6block": "6-block exponential moving average of median fee rate",
    "tx_arrival_rate_10m": "Transaction arrival rate over 10-minute window",
    "tx_arrival_rate_30m": "Transaction arrival rate over 30-minute window",

    # Epoch assignment
    "epoch_id": "Time-based epoch identifier (30-minute windows by default)",

    # Engineered features
    "fee_rate_percentile": "Tie-aware within-epoch fee rate percentile (priority, 0-1)",
    "impatience": "Impatience proxy: 1 / (min_respend_blocks + eps), truncated",
    "cumulative_weight": "Cumulative weight of txs in the block up to this tx",
    "blockspace_utilization": "Fraction of max block weight used (cumulative_weight / 4M WU)",

    # Log transforms
    "log_fee_rate": "log(1 + fee_rate)",
    "log_waittime": "log(1 + waittime)",
    "log_weight": "log(1 + weight)",
    "log_congestion": "log(1 + mempool_tx_count)",
    "log_size": "log(1 + mempool_size)",
    "log_total_output": "log(1 + total_output_amount)",
    "log_impatience": "log(impatience)",
    "log_n_in": "log(1 + n_in)",
    "log_n_out": "log(1 + n_out)",
    "log_block_median_feerate": "log(1 + block_median_feerate)",
    "log_block_p90_feerate": "log(1 + block_p90_feerate)",
    "log_ema_feerate_3block": "log(1 + ema_feerate_3block)",
    "log_time_since_last_block": "log(1 + time_since_last_block)",
    "log_tx_arrival_rate_10m": "log(1 + tx_arrival_rate_10m)",

    # Temporal features
    "hour_sin": "sin(2*pi*hour/24) — cyclical hour encoding",
    "hour_cos": "cos(2*pi*hour/24) — cyclical hour encoding",
    "dow_sin": "sin(2*pi*day_of_week/7) — cyclical day encoding",
    "dow_cos": "cos(2*pi*day_of_week/7) — cyclical day encoding",
    "is_weekend": "1 if Saturday or Sunday",

    # Derived type indicators
    "is_consolidation": "1 if label == 'consolidation'",
    "is_coinjoin": "1 if label == 'coinjoin'",
    "is_datacarrying": "1 if label == 'datacarrying'",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def get_valid_blocks_from_mempool_space(
    mempool_space_db: str = DEFAULT_MEMPOOL_SPACE_DB,
) -> List[str]:
    conn = sqlite3.connect(mempool_space_db)
    blocks = pd.read_sql_query(
        "SELECT DISTINCT block_hash FROM transactions", conn
    )["block_hash"].tolist()
    conn.close()
    return blocks


def load_corrected_weights(
    mempool_space_db: str = DEFAULT_MEMPOOL_SPACE_DB,
) -> pd.DataFrame:
    conn = sqlite3.connect(mempool_space_db)
    df = pd.read_sql_query(
        """
        SELECT
            txid,
            weight  AS weight_corrected,
            size    AS size_corrected,
            vsize   AS vsize_corrected,
            fee     AS fee_corrected,
            fee_rate AS fee_rate_corrected
        FROM transactions
        """,
        conn,
    )
    conn.close()
    return df


def load_data_from_sqlite(
    db_path: str,
    block_limit: Optional[int] = None,
    mempool_space_db: str = DEFAULT_MEMPOOL_SPACE_DB,
) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("Loading data from SQLite (with corrected weights)")
    logger.info("=" * 60)

    valid_blocks = get_valid_blocks_from_mempool_space(mempool_space_db)
    logger.info("Blocks with corrected weight data: %d", len(valid_blocks))

    conn = sqlite3.connect(db_path)
    total_count = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM mempool_transactions", conn
    )["count"].iloc[0]
    logger.info("Total transactions in source database: %d", total_count)

    if block_limit and block_limit < len(valid_blocks):
        logger.info("Sampling %d blocks from %d available...", block_limit, len(valid_blocks))

        placeholders = ",".join(["?" for _ in valid_blocks])
        all_blocks_query = f"""
            SELECT DISTINCT
                conf_block_hash,
                strftime('%Y-%m-%d %H', mined_at) as hour_bin
            FROM mempool_transactions
            WHERE conf_block_hash IN ({placeholders})
              AND mined_at IS NOT NULL
        """
        all_blocks_df = pd.read_sql_query(all_blocks_query, conn, params=valid_blocks)
        logger.info("Found %d blocks with timestamps", len(all_blocks_df))

        all_blocks_df = all_blocks_df.sample(frac=1, random_state=42)
        all_blocks_df["rank_in_hour"] = all_blocks_df.groupby("hour_bin").cumcount()

        n_hours = all_blocks_df["hour_bin"].nunique()
        blocks_per_hour = max(1, block_limit // n_hours)

        sampled_df = all_blocks_df[all_blocks_df["rank_in_hour"] < blocks_per_hour]
        if len(sampled_df) < block_limit:
            remaining = all_blocks_df[all_blocks_df["rank_in_hour"] >= blocks_per_hour]
            sampled_df = pd.concat(
                [sampled_df, remaining.head(block_limit - len(sampled_df))]
            )
        sampled_blocks = sampled_df["conf_block_hash"].head(block_limit).tolist()
        logger.info("Sampled %d blocks with temporal balancing", len(sampled_blocks))
    else:
        sampled_blocks = valid_blocks
        logger.info("Using all %d available blocks", len(sampled_blocks))

    placeholders = ",".join(["?" for _ in sampled_blocks])
    query = f"""
        SELECT
            tx_id, conf_block_hash, found_at, mined_at,
            fee_rate AS fee_rate_original,
            waittime, weight AS weight_original, size AS size_original,
            min_respend_blocks, child_txid, rbf_fee_total,
            mempool_tx_count, mempool_size, total_output_amount,
            absolute_fee AS absolute_fee_original
        FROM mempool_transactions
        WHERE conf_block_hash IN ({placeholders})
    """
    logger.info("Loading transactions from source database...")
    df = pd.read_sql_query(query, conn, params=sampled_blocks)
    conn.close()
    logger.info("Loaded %d transactions from source", len(df))

    logger.info("Merging corrected weight data from mempool.space...")
    corrected_df = load_corrected_weights(mempool_space_db)
    logger.info("Corrected weight data available for %d transactions", len(corrected_df))

    df = df.merge(corrected_df, left_on="tx_id", right_on="txid", how="left")
    df["weight"] = df["weight_corrected"].fillna(df["weight_original"])
    df["size"] = df["size_corrected"].fillna(df["size_original"])
    df["fee_rate"] = df["fee_rate_corrected"].fillna(df["fee_rate_original"])
    df["absolute_fee"] = df["fee_corrected"].fillna(df["absolute_fee_original"])
    df["vsize"] = df["vsize_corrected"].fillna(df["weight"] / 4)
    df["has_corrected_weight"] = df["weight_corrected"].notna()
    df = df.drop(
        columns=["txid", "weight_corrected", "size_corrected",
                 "vsize_corrected", "fee_corrected", "fee_rate_corrected"]
    )

    corrected_count = df["has_corrected_weight"].sum()
    logger.info(
        "Loaded %d transactions — %d with corrected weights (%.1f%%)",
        len(df), corrected_count, 100 * corrected_count / len(df),
    )
    return df


def load_and_merge_annotations(df: pd.DataFrame, annotation_path: str) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("Loading Transaction Annotations")
    logger.info("=" * 60)

    annot = pd.read_pickle(annotation_path)
    logger.info("Loaded %d annotations", len(annot))

    n_errors = (annot["label"] == "error").sum()
    if n_errors > 0:
        annot = annot[annot["label"] != "error"].copy()
        logger.info("Removed %d error annotations", n_errors)

    annot["has_op_return"]  = annot["tags"].apply(lambda t: int("op_return"   in t))
    annot["has_inscription"] = annot["tags"].apply(lambda t: int("inscription" in t))

    annot_merge = annot[
        ["tx_id", "label", "n_in", "n_out", "has_op_return", "has_inscription"]
    ].copy()

    n_before = len(df)
    df = df.merge(annot_merge, on="tx_id", how="left")
    n_matched = df["label"].notna().sum()
    logger.info(
        "Annotations merged: %d / %d matched (%.1f%%)",
        n_matched, n_before, 100 * n_matched / n_before,
    )

    df["label"]           = df["label"].fillna("unknown")
    df["n_in"]            = df["n_in"].fillna(1).astype(int)
    df["n_out"]           = df["n_out"].fillna(2).astype(int)
    df["has_op_return"]   = df["has_op_return"].fillna(0).astype(int)
    df["has_inscription"] = df["has_inscription"].fillna(0).astype(int)
    return df


def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("Data Quality Check")
    logger.info("=" * 60)

    critical_cols = ["fee_rate", "waittime", "weight", "mempool_tx_count"]
    df_clean = df.dropna(subset=critical_cols).copy()
    logger.info(
        "Clean dataset: %d transactions (%.1f%% retained)",
        len(df_clean), len(df_clean) / len(df) * 100,
    )
    return df_clean


def load_block_state_features(
    mempool_space_db: str = DEFAULT_MEMPOOL_SPACE_DB,
) -> pd.DataFrame:
    conn = sqlite3.connect(mempool_space_db)
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    if "block_state_features" not in tables:
        logger.warning(
            "block_state_features table not found. "
            "Run: python scripts/compute_block_state_features.py"
        )
        conn.close()
        return pd.DataFrame()
    df = pd.read_sql_query("SELECT * FROM block_state_features", conn)
    conn.close()
    logger.info("Loaded block state features for %d blocks", len(df))
    return df


# ---------------------------------------------------------------------------
# CPFP helpers
# ---------------------------------------------------------------------------

def fetch_child_transaction_data(
    db_path: str,
    child_txids: List[str],
    mempool_space_db: str = DEFAULT_MEMPOOL_SPACE_DB,
) -> pd.DataFrame:
    if not child_txids:
        return pd.DataFrame(
            columns=["tx_id", "fee_rate", "weight", "vsize", "absolute_fee", "size", "waittime"]
        )

    chunk_size = 900

    corrected_conn = sqlite3.connect(mempool_space_db)
    corrected_list = []
    for i in range(0, len(child_txids), chunk_size):
        chunk = child_txids[i : i + chunk_size]
        ph = ",".join(["?" for _ in chunk])
        q = f"""
            SELECT txid AS tx_id,
                   weight AS weight_corrected, vsize AS vsize_corrected,
                   fee AS fee_corrected, fee_rate AS fee_rate_corrected,
                   size AS size_corrected
            FROM transactions WHERE txid IN ({ph})
        """
        corrected_list.append(pd.read_sql_query(q, corrected_conn, params=chunk))
    corrected_conn.close()
    corrected_df = (
        pd.concat(corrected_list, ignore_index=True) if corrected_list else pd.DataFrame()
    )

    conn = sqlite3.connect(db_path)
    children_list = []
    for i in range(0, len(child_txids), chunk_size):
        chunk = child_txids[i : i + chunk_size]
        ph = ",".join(["?" for _ in chunk])
        q = f"""
            SELECT tx_id, fee_rate AS fee_rate_original,
                   weight AS weight_original, absolute_fee AS absolute_fee_original,
                   size AS size_original, waittime
            FROM mempool_transactions WHERE tx_id IN ({ph})
        """
        children_list.append(pd.read_sql_query(q, conn, params=chunk))
    conn.close()

    if not children_list:
        return pd.DataFrame(
            columns=["tx_id", "fee_rate", "weight", "vsize", "absolute_fee", "size", "waittime"]
        )

    children_df = pd.concat(children_list, ignore_index=True)

    if len(corrected_df) > 0:
        children_df = children_df.merge(corrected_df, on="tx_id", how="left")
        children_df["weight"]       = children_df["weight_corrected"].fillna(children_df["weight_original"])
        children_df["size"]         = children_df["size_corrected"].fillna(children_df["size_original"])
        children_df["fee_rate"]     = children_df["fee_rate_corrected"].fillna(children_df["fee_rate_original"])
        children_df["absolute_fee"] = children_df["fee_corrected"].fillna(children_df["absolute_fee_original"])
        children_df["vsize"]        = children_df["vsize_corrected"].fillna(children_df["weight"] / 4)
        drop_cols = [c for c in children_df.columns if c.endswith("_corrected") or c.endswith("_original")]
        children_df = children_df.drop(columns=drop_cols, errors="ignore")
    else:
        children_df["weight"]       = children_df["weight_original"]
        children_df["size"]         = children_df["size_original"]
        children_df["fee_rate"]     = children_df["fee_rate_original"]
        children_df["absolute_fee"] = children_df["absolute_fee_original"]
        children_df["vsize"]        = children_df["weight"] / 4
        drop_cols = [c for c in children_df.columns if c.endswith("_original")]
        children_df = children_df.drop(columns=drop_cols, errors="ignore")

    return children_df


def collapse_cpfp_packages(
    df: pd.DataFrame,
    db_path: str,
    mempool_space_db: str = DEFAULT_MEMPOOL_SPACE_DB,
) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("CPFP PACKAGE COLLAPSING")
    logger.info("=" * 60)

    cpfp_mask = df["child_txid"].notna() & (df["child_txid"] != "")
    n_parents = cpfp_mask.sum()
    logger.info("CPFP parents found: %d (%.2f%%)", n_parents, n_parents / len(df) * 100)

    if n_parents == 0:
        logger.info("No CPFP transactions to collapse.")
        return df

    child_txids = df.loc[cpfp_mask, "child_txid"].unique().tolist()
    logger.info("Unique child transactions: %d", len(child_txids))

    children_df = fetch_child_transaction_data(db_path, child_txids, mempool_space_db)
    logger.info("Found %d child transactions in database", len(children_df))

    if len(children_df) == 0:
        logger.warning("No child transactions found in database. Skipping collapse.")
        return df

    if "absolute_fee" not in df.columns or df["absolute_fee"].isna().any():
        df["absolute_fee"] = df["absolute_fee"].fillna(df["fee_rate"] * df["weight"] / 4)

    if children_df["absolute_fee"].isna().any():
        children_df["absolute_fee"] = children_df["absolute_fee"].fillna(
            children_df["fee_rate"] * children_df["weight"] / 4
        )

    child_data = children_df.set_index("tx_id")[
        ["fee_rate", "weight", "vsize", "absolute_fee", "size"]
    ].to_dict("index")

    package_fee_rates, package_weights, package_vsizes = [], [], []
    matched_count = unmatched_count = 0

    for idx in df[cpfp_mask].index:
        child_txid  = df.loc[idx, "child_txid"]
        parent_fee  = df.loc[idx, "absolute_fee"]
        parent_vsize = df.loc[idx, "vsize"]
        parent_weight = df.loc[idx, "weight"]

        if child_txid in child_data:
            ci = child_data[child_txid]
            total_fee   = parent_fee + ci["absolute_fee"]
            total_vsize = parent_vsize + ci["vsize"]
            total_weight = parent_weight + ci["weight"]
            package_fee_rates.append(total_fee / total_vsize)
            package_weights.append(total_weight)
            package_vsizes.append(total_vsize)
            matched_count += 1
        else:
            package_fee_rates.append(df.loc[idx, "fee_rate"])
            package_weights.append(df.loc[idx, "weight"])
            package_vsizes.append(df.loc[idx, "vsize"])
            unmatched_count += 1

    logger.info("Matched parent-child pairs: %d", matched_count)
    logger.info("Unmatched (child not in DB): %d", unmatched_count)

    df.loc[cpfp_mask, "fee_rate"] = package_fee_rates
    df.loc[cpfp_mask, "weight"]   = package_weights
    df.loc[cpfp_mask, "vsize"]    = package_vsizes
    df.loc[cpfp_mask, "is_cpfp_package"] = 1
    df.loc[~cpfp_mask, "is_cpfp_package"] = 0

    child_txid_set = set(child_txids)
    is_child = df["tx_id"].isin(child_txid_set)
    n_children_in_data = is_child.sum()
    df_collapsed = df[~is_child].copy()

    logger.info(
        "CPFP collapse: %d → %d transactions (%d children removed)",
        len(df), len(df_collapsed), n_children_in_data,
    )
    return df_collapsed


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def assign_epochs(
    df: pd.DataFrame,
    epoch_mode: str = "block",
    blocks_per_epoch: int = 3,
    target_epoch_size: int = 1000,
    epoch_duration_minutes: int = 30,
    min_epoch_fullness: float = 0.25,
) -> pd.DataFrame:
    logger.info("Assigning epochs (mode=%s)...", epoch_mode)

    df["found_at"] = pd.to_datetime(df["found_at"])
    df["mined_at"] = pd.to_datetime(df["mined_at"])

    if epoch_mode == "block":
        block_times = df.groupby("conf_block_hash")["mined_at"].min().sort_values()
        unique_blocks = block_times.index.tolist()
        block_to_seq = {b: i for i, b in enumerate(unique_blocks)}
        df["block_seq"] = df["conf_block_hash"].map(block_to_seq)
        df["epoch_id"]  = df["block_seq"] // blocks_per_epoch

    elif epoch_mode == "fixed_size":
        df = df.sort_values("found_at").copy()
        n_transactions = len(df)
        n_epochs = max(1, n_transactions // target_epoch_size)
        df["epoch_id"] = (np.arange(n_transactions) // target_epoch_size).clip(0, n_epochs - 1)

    elif epoch_mode == "time":
        df["found_at_ts"] = df["found_at"].astype("int64") // 10**9
        start_time    = df["found_at_ts"].min()
        epoch_seconds = epoch_duration_minutes * 60
        num_epochs    = int(np.ceil((df["found_at_ts"].max() - start_time) / epoch_seconds)) + 1
        bins = np.linspace(start_time, start_time + num_epochs * epoch_seconds, num_epochs + 1)
        df["epoch_id"] = pd.cut(df["found_at_ts"], bins=bins, labels=False, include_lowest=True)
    else:
        raise ValueError(f"Unknown epoch_mode: {epoch_mode}")

    epoch_counts = df.groupby("epoch_id").size()
    median_size  = epoch_counts.median()
    threshold    = max(20, int(median_size * min_epoch_fullness))
    sparse_mask  = epoch_counts < threshold
    n_sparse     = sparse_mask.sum()
    if n_sparse > 0:
        logger.info("Dropping %d sparse epoch(s) below %d txs", n_sparse, threshold)
    full_ids = epoch_counts[~sparse_mask].index
    df = df[df["epoch_id"].isin(full_ids)]

    logger.info(
        "Epochs assigned: %d full epochs, %d txs",
        df["epoch_id"].nunique(), len(df),
    )
    return df


def compute_fee_rate_percentile(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing tie-aware fee-rate percentiles...")

    def tie_aware_percentile(group):
        r = group["fee_rate"].to_numpy()
        N = len(group)
        if N == 0:
            return pd.Series(dtype=float)
        if N == 1:
            return pd.Series([0.5], index=group.index)
        vc = pd.Series(r).value_counts().sort_index()
        cum_less = vc.cumsum().shift(1, fill_value=0)
        less = pd.Series(r).map(cum_less).to_numpy()
        eq   = pd.Series(r).map(vc).to_numpy()
        return pd.Series((less + 0.5 * eq) / N, index=group.index)

    df["fee_rate_percentile"] = df.groupby("epoch_id", group_keys=False).apply(
        tie_aware_percentile
    )
    df["fee_rate_percentile"] = df["fee_rate_percentile"].fillna(0.5).clip(0.001, 0.999)
    logger.info(
        "Percentiles computed: [%.4f, %.4f]",
        df["fee_rate_percentile"].min(), df["fee_rate_percentile"].max(),
    )
    return df


def compute_impatience_proxy(
    df: pd.DataFrame, truncation_blocks: int, epsilon: float
) -> pd.DataFrame:
    logger.info("Computing impatience proxy...")
    df["min_respend_blocks"] = df["min_respend_blocks"].fillna(-1)
    valid_respend = df["min_respend_blocks"] >= 0
    respend_truncated = df["min_respend_blocks"].clip(upper=truncation_blocks)
    df["impatience"] = np.nan
    df.loc[valid_respend, "impatience"] = 1 / (respend_truncated.loc[valid_respend] + epsilon)
    median_impatience = df.loc[valid_respend, "impatience"].median()
    df["impatience"] = df["impatience"].fillna(median_impatience)
    return df


def compute_blockspace_utilization(df: pd.DataFrame, max_block_weight: int) -> pd.DataFrame:
    logger.info("Computing blockspace utilization...")
    df = df.sort_values(["conf_block_hash", "found_at"]).copy()
    df["cumulative_weight"]      = df.groupby("conf_block_hash")["weight"].cumsum()
    df["blockspace_utilization"] = (df["cumulative_weight"] / max_block_weight).clip(0, 1)
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating derived features...")

    df["log_weight"]       = np.log1p(df["weight"])
    df["log_waittime"]     = np.log1p(df["waittime"])
    df["log_fee_rate"]     = np.log1p(df["fee_rate"])
    df["log_congestion"]   = np.log1p(df["mempool_tx_count"])
    df["log_size"]         = np.log1p(df["mempool_size"])
    df["log_total_output"] = np.log1p(df["total_output_amount"])

    if "block_median_feerate" in df.columns:
        df["log_block_median_feerate"]   = np.log1p(df["block_median_feerate"])
        df["log_block_p90_feerate"]      = np.log1p(df["block_p90_feerate"])
        df["log_ema_feerate_3block"]     = np.log1p(df["ema_feerate_3block"])
        df["log_time_since_last_block"]  = np.log1p(df["time_since_last_block"].clip(lower=0))
        df["log_tx_arrival_rate_10m"]    = np.log1p(df["tx_arrival_rate_10m"].fillna(0))

    df["has_rbf"] = (df["rbf_fee_total"].notna() & (df["rbf_fee_total"] > 0)).astype(int)

    if "is_cpfp_package" not in df.columns:
        df["is_cpfp_package"] = df["child_txid"].notna().astype(int)
    df["is_cpfp_package"] = df["is_cpfp_package"].fillna(0).astype(int)

    df["log_impatience"] = np.log(df["impatience"].clip(lower=1e-10))

    if "found_at" in df.columns:
        found_dt = pd.to_datetime(df["found_at"])
        hour = found_dt.dt.hour
        dow  = found_dt.dt.dayofweek
        df["hour_sin"]   = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"]   = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"]    = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"]    = np.cos(2 * np.pi * dow / 7)
        df["is_weekend"] = dow.isin([5, 6]).astype(int)

    if "n_in" in df.columns:
        df["log_n_in"]  = np.log1p(df["n_in"])
        df["log_n_out"] = np.log1p(df["n_out"])

    if "label" in df.columns:
        df["is_consolidation"] = (df["label"] == "consolidation").astype(int)
        df["is_coinjoin"]      = (df["label"] == "coinjoin").astype(int)
        df["is_datacarrying"]  = (df["label"] == "datacarrying").astype(int)

    return df


def prepare_features(
    df: pd.DataFrame,
    epoch_mode: str = "time",
    blocks_per_epoch: int = 2,
    target_epoch_size: int = 1000,
    epoch_duration_minutes: int = 30,
    respend_truncation_blocks: int = 14,
    epsilon: float = 1e-6,
    max_block_weight: int = 4_000_000,
) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("FEATURE PREPARATION PIPELINE")
    logger.info("=" * 60)

    df = assign_epochs(
        df,
        epoch_mode=epoch_mode,
        blocks_per_epoch=blocks_per_epoch,
        target_epoch_size=target_epoch_size,
        epoch_duration_minutes=epoch_duration_minutes,
    )
    df = compute_fee_rate_percentile(df)
    df = compute_impatience_proxy(df, respend_truncation_blocks, epsilon)
    df = compute_blockspace_utilization(df, max_block_weight)
    df = create_derived_features(df)

    logger.info("Feature preparation complete: %d transactions, %d epochs",
                len(df), df["epoch_id"].nunique())
    return df


# ---------------------------------------------------------------------------
# Dataset export
# ---------------------------------------------------------------------------

def export_dataset(
    df: pd.DataFrame,
    output_path: str,
    write_metadata: bool = True,
    build_params: Optional[Dict] = None,
) -> str:
    """Export DataFrame to Parquet with an optional column metadata JSON sidecar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    size_mb = output_path.stat().st_size / 1e6
    logger.info("Dataset exported: %s (%d rows, %.1f MB)", output_path, len(df), size_mb)

    if write_metadata:
        metadata = {
            "description": (
                "Analysis-ready dataset for Bitcoin fee estimation "
                "(VCG two-stage structural model)"
            ),
            "exported_at": datetime.now().isoformat(),
            "n_transactions": len(df),
            "n_columns": len(df.columns),
        }

        if "epoch_id" in df.columns:
            metadata["n_epochs"] = int(df["epoch_id"].nunique())
        if "found_at" in df.columns:
            metadata["time_range"] = {
                "start": str(df["found_at"].min()),
                "end": str(df["found_at"].max()),
            }
        if "conf_block_hash" in df.columns:
            metadata["n_blocks"] = int(df["conf_block_hash"].nunique())

        if build_params:
            metadata["build_params"] = build_params

        # Column-level metadata: dtype + description for each column present
        columns = {}
        for col in df.columns:
            entry = {"dtype": str(df[col].dtype)}
            if col in COLUMN_DESCRIPTIONS:
                entry["description"] = COLUMN_DESCRIPTIONS[col]
            columns[col] = entry
        metadata["columns"] = columns

        meta_path = output_path.with_suffix(".metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Column metadata: %s", meta_path)

    return str(output_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_analysis_dataset(
    db_path: str,
    annotation_path: str,
    block_limit: Optional[int] = None,
    mempool_space_db: str = DEFAULT_MEMPOOL_SPACE_DB,
    epoch_mode: str = "time",
    epoch_duration_minutes: int = 30,
    blocks_per_epoch: int = 2,
    target_epoch_size: int = 1000,
    respend_truncation_blocks: int = 14,
    epsilon: float = 1e-6,
    max_block_weight: int = 4_000_000,
) -> pd.DataFrame:
    """Build the complete analysis-ready dataset from raw sources.

    Loads transaction data from the source SQLite database, merges corrected
    weights/fees from the mempool.space database, merges transaction annotations,
    collapses CPFP packages, and engineers all features needed for the two-stage
    structural fee model.
    """
    df_raw = load_data_from_sqlite(db_path, block_limit, mempool_space_db)
    df_raw = load_and_merge_annotations(df_raw, annotation_path)

    block_features = load_block_state_features(mempool_space_db)
    if len(block_features) > 0:
        n_before = len(df_raw)
        df_raw = df_raw.merge(
            block_features, left_on="conf_block_hash", right_on="block_hash", how="left"
        )
        if "block_hash" in df_raw.columns:
            df_raw = df_raw.drop(columns=["block_hash"])
        matched = df_raw["block_median_feerate"].notna().sum()
        logger.info("Block state features merged: %d / %d matched", matched, n_before)

    df = check_data_quality(df_raw)
    del df_raw
    gc.collect()

    df = collapse_cpfp_packages(df, db_path, mempool_space_db)
    df = prepare_features(
        df,
        epoch_mode=epoch_mode,
        blocks_per_epoch=blocks_per_epoch,
        target_epoch_size=target_epoch_size,
        epoch_duration_minutes=epoch_duration_minutes,
        respend_truncation_blocks=respend_truncation_blocks,
        epsilon=epsilon,
        max_block_weight=max_block_weight,
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build analysis-ready dataset for the Bitcoin fee estimation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        default="/home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db",
        help="Path to source SQLite database",
    )
    parser.add_argument(
        "--annotation-path",
        default="/home/armin/datalake/data-samples/tx-annotations-15m-2-7-2026.pkl",
        help="Path to transaction annotations pickle",
    )
    parser.add_argument(
        "--mempool-db",
        default=DEFAULT_MEMPOOL_SPACE_DB,
        help="Path to mempool.space SQLite database",
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
        "--output", "-o", default="data/analysis_ready.parquet",
        help="Output path for Parquet file",
    )
    parser.add_argument(
        "--no-metadata", action="store_true",
        help="Skip writing the JSON metadata sidecar",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    build_params = {
        "db_path": args.db_path,
        "annotation_path": args.annotation_path,
        "mempool_space_db": args.mempool_db,
        "block_limit": args.block_limit,
        "epoch_mode": args.epoch_mode,
        "epoch_duration_minutes": args.epoch_duration,
    }

    df = build_analysis_dataset(
        db_path=args.db_path,
        annotation_path=args.annotation_path,
        block_limit=args.block_limit,
        mempool_space_db=args.mempool_db,
        epoch_mode=args.epoch_mode,
        epoch_duration_minutes=args.epoch_duration,
    )

    logger.info("Dataset ready: %d transactions", len(df))

    export_dataset(
        df,
        output_path=args.output,
        write_metadata=not args.no_metadata,
        build_params=build_params,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()

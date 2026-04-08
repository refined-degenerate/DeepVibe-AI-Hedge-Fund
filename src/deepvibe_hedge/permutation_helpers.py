"""Small subset of permutation utilities used by ``mad.permutation_test`` (no vectorbt / Donchian deps)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def block_shuffle(returns: np.ndarray, block_size: int) -> np.ndarray:
    """Block bootstrap: shuffle contiguous blocks, preserving local autocorrelation within each block."""
    blocks = [returns[i : i + block_size] for i in range(0, len(returns), block_size)]
    np.random.shuffle(blocks)
    shuffled = np.concatenate(blocks)
    return shuffled[: len(returns)]


def _verdict(p_value: float, alpha: float) -> tuple[str, str]:
    if p_value < alpha:
        return (
            f"PASS  (p = {p_value:.4f} < α = {alpha})\n"
            "This combo's IS Profit Factor is unlikely to arise by chance.\n"
            "Proceed to walk-forward testing.",
            "#00e676",
        )
    return (
        f"FAIL  (p = {p_value:.4f} ≥ α = {alpha})\n"
        "IS Profit Factor is consistent with random chance.\n"
        "Do not proceed to live trading.",
        "#ff4d4d",
    )


def _normalize_optim_split(raw: str) -> str:
    val = str(raw).strip().lower()
    if val == "avg":
        return "avg"
    try:
        return str(int(val))
    except Exception as exc:
        raise ValueError(f"Invalid --optim-split value {raw!r}. Use 'avg' or an integer split id.") from exc


def _available_best_rows(db_path: Path, columns: str = "*", table: str = "best_per_split") -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql(f"SELECT {columns} FROM {table}", con)

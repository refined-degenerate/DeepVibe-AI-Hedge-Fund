"""Data directories under the project ``data/`` tree (OHLCV SQLite, MAD optimiser outputs)."""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = _PROJECT_ROOT / "data"
OHLCV_DIR = DATA_ROOT / "ohlcv"
MAD_DATA_DIR = DATA_ROOT / "mad"


def ensure_data_dirs() -> None:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    MAD_DATA_DIR.mkdir(parents=True, exist_ok=True)

"""
Assigns walk-forward splits and pre-calculates indicators on the existing OHLCV dataset.

Split 0  — warmup period covering the longest enabled precompute (MA, Donchian, ADX, etc.).
           Used only to seed indicators; never used for training or testing.

Split 1..NUM_SPLITS — remaining candles divided into equal chunks for walk-forward
                      optimisation and out-of-sample testing.

Run after alpaca_fetcher. Processes every symbol in ``config.ohlcv_pipeline_tickers()``; each DB/CSV is updated in place.

OHLCV should be downloaded with ``config.ALPACA_BAR_ADJUSTMENT`` (default ``split``) so prices are
split-normalized before indicators and returns are computed. If you change adjustment or fix legacy
raw DBs, re-run the fetcher then this splitter.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from deepvibe_hedge import config
from deepvibe_hedge.paths import OHLCV_DIR


def _progress(done: int, total: int, label: str) -> None:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    pct = 100.0 * done / total
    bar_len = 24
    filled = int(round(bar_len * done / total))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"[{bar}] {done}/{total} ({pct:6.2f}%)  {label}")


def print_loaded_config() -> None:
    cfg_path = Path(getattr(config, "__file__", "unknown")).resolve()
    print(
        "Loaded config:\n"
        f"  file               : {cfg_path}\n"
        f"  SPLITTER_MA_START  : {config.SPLITTER_MA_START}\n"
        f"  SPLITTER_MA_STOP   : {config.SPLITTER_MA_STOP}\n"
        f"  SPLITTER_MA_STEP   : {config.SPLITTER_MA_STEP}\n"
        f"  SPLITTER_DONCHIAN_START : {getattr(config, 'SPLITTER_DONCHIAN_START', 'n/a')}\n"
        f"  SPLITTER_DONCHIAN_STOP  : {getattr(config, 'SPLITTER_DONCHIAN_STOP', 'n/a')}\n"
        f"  SPLITTER_DONCHIAN_STEP  : {getattr(config, 'SPLITTER_DONCHIAN_STEP', 'n/a')}\n"
        f"  SPLITTER_NUM_SPLITS: {config.SPLITTER_NUM_SPLITS}\n"
        f"  SPLITTER_ENABLE_SPLIT_ASSIGNMENT : {getattr(config, 'SPLITTER_ENABLE_SPLIT_ASSIGNMENT', True)}\n"
        f"  SPLITTER_ENABLE_MA_PRECOMPUTE    : {getattr(config, 'SPLITTER_ENABLE_MA_PRECOMPUTE', True)}\n"
        f"  SPLITTER_ENABLE_DONCHIAN_PRECOMPUTE: {getattr(config, 'SPLITTER_ENABLE_DONCHIAN_PRECOMPUTE', True)}\n"
        f"  SPLITTER_ADX_START : {getattr(config, 'SPLITTER_ADX_START', 'n/a')}\n"
        f"  SPLITTER_ADX_STOP  : {getattr(config, 'SPLITTER_ADX_STOP', 'n/a')}\n"
        f"  SPLITTER_ADX_STEP  : {getattr(config, 'SPLITTER_ADX_STEP', 'n/a')}\n"
        f"  SPLITTER_ENABLE_ADX_PRECOMPUTE: {getattr(config, 'SPLITTER_ENABLE_ADX_PRECOMPUTE', True)}\n"
        f"  MOD_DONCHAIAN_DAILY_SMA_REGIME_ENABLED: {getattr(config, 'MOD_DONCHAIAN_DAILY_SMA_REGIME_ENABLED', False)}\n"
        f"  DAILY_REGIME_SMA_PERIODS (count): {len(_daily_regime_sma_periods())}\n"
        f"  DAILY_REGIME_WARMUP_UNIQUE_CAL_DAYS: {daily_regime_warmup_min_unique_calendar_days()}\n"
        f"  SPLITTER_WARMUP_BARS (effective): {_required_warmup_bars()}\n"
        f"  OHLCV pipeline tickers: {', '.join(config.ohlcv_pipeline_tickers())}\n"
    )


def _filename(ticker: str | None = None) -> str:
    t = str(ticker or config.TARGET_TICKER).strip().upper()
    return f"{t}_{config.TARGET_CANDLE_GRANULARITY}"


def load_ohlcv(ticker: str | None = None) -> pd.DataFrame:
    path = OHLCV_DIR / f"{_filename(ticker)}.db"
    if not path.exists():
        raise FileNotFoundError(
            f"No DB found at {path} — run: PYTHONPATH=src python -m deepvibe_hedge.alpaca_fetcher"
        )
    with sqlite3.connect(path) as con:
        cols = [row[1] for row in con.execute("PRAGMA table_info(ohlcv)").fetchall()]
        wanted = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in cols]
        if "timestamp" not in wanted:
            raise RuntimeError("ohlcv table is missing required 'timestamp' column.")
        query = f"SELECT {', '.join(wanted)} FROM ohlcv"
        df = pd.read_sql(query, con, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    warmup = _required_warmup_bars(df)   # split 0 = longest enabled indicator period
    min_days = int(getattr(config, "SPLITTER_MIN_WARMUP_DAYS", 0))
    if min_days > 0:
        warmup = max(warmup, _warmup_bars_for_min_days(df, min_days))

    if n <= warmup:
        raise ValueError(
            f"Only {n} rows of data — need more than {warmup} bars for warmup."
        )

    remaining = n - warmup
    # evenly distribute remaining rows across SPLITTER_NUM_SPLITS
    indices = np.arange(remaining)
    split_labels = (indices * config.SPLITTER_NUM_SPLITS // remaining) + 1  # 1..NUM_SPLITS

    splits = np.zeros(n, dtype=int)   # warmup rows = split 0
    splits[warmup:] = split_labels

    df["split"] = splits
    return df


def _warmup_bars_for_min_days(df: pd.DataFrame, min_days: int) -> int:
    if min_days <= 0:
        return 0
    day_series = pd.Series(df.index.normalize(), index=df.index)
    unique_days = day_series.drop_duplicates()
    if len(unique_days) <= min_days:
        # Not enough unique days to leave any non-warmup rows.
        return len(df)
    # Warmup ends at the first bar of day (min_days + 1),
    # so warmup contains at least `min_days` unique daily bars.
    day_cutoff = unique_days.iloc[min_days]
    day_vals = day_series.to_numpy()
    idx = np.flatnonzero(day_vals == day_cutoff)
    first_test_pos = int(idx[0]) if len(idx) else len(df)
    return max(1, first_test_pos)


def _daily_regime_sma_periods() -> list[int]:
    if not bool(getattr(config, "MOD_DONCHAIAN_DAILY_SMA_REGIME_ENABLED", False)):
        return []
    start = int(getattr(config, "MOD_DONCHAIAN_DAILY_SMA_DAYS_START", 2))
    stop = int(getattr(config, "MOD_DONCHAIAN_DAILY_SMA_DAYS_STOP", 200))
    step = int(getattr(config, "MOD_DONCHAIAN_DAILY_SMA_DAYS_STEP", 10))
    if step <= 0 or stop < start:
        return []
    start = max(2, start)
    return list(range(start, stop + 1, step))


def daily_regime_warmup_min_unique_calendar_days() -> int:
    """
    Distinct calendar dates that must fall in split 0 so every ``daily_regime_sma_<n>`` column can be
    non-NaN before split 1: ``max(n) + 1`` (rolling n on daily closes plus ``shift(1)``), plus
    ``MOD_DONCHAIAN_DAILY_SMA_WARMUP_EXTRA_CALENDAR_DAYS``. Returns 0 when the daily regime feature is off.
    """
    periods = _daily_regime_sma_periods()
    if not periods:
        return 0
    extra = max(0, int(getattr(config, "MOD_DONCHAIAN_DAILY_SMA_WARMUP_EXTRA_CALENDAR_DAYS", 0)))
    return int(max(periods)) + 1 + extra


def _required_warmup_bars(df: pd.DataFrame | None = None) -> int:
    warmups: list[int] = []
    if bool(getattr(config, "SPLITTER_ENABLE_MA_PRECOMPUTE", True)):
        warmups.append(int(config.SPLITTER_MA_STOP))
    if bool(getattr(config, "SPLITTER_ENABLE_DONCHIAN_PRECOMPUTE", True)):
        warmups.append(int(getattr(config, "SPLITTER_DONCHIAN_STOP", 200)))
    if bool(getattr(config, "SPLITTER_ENABLE_ADX_PRECOMPUTE", True)):
        adx_ps = _adx_periods()
        if adx_ps:
            pmax = max(adx_ps)
            # First bar with valid ADX is index 2*P − 2 (0-based); split 1 starts at index >= 2*P − 1.
            warmups.append(2 * pmax - 1)
    nd_daily = daily_regime_warmup_min_unique_calendar_days()
    if nd_daily > 0:
        if df is not None and not df.empty:
            warmups.append(_warmup_bars_for_min_days(df, nd_daily))
        else:
            bars_per_day = {
                "1m": 390.0,
                "5m": 78.0,
                "15m": 26.0,
                "1h": 6.5,
                "4h": 2.0,
                "1d": 1.0,
                "1w": 1.0 / 5.0,
                "1mo": 1.0 / 21.0,
            }.get(str(getattr(config, "TARGET_CANDLE_GRANULARITY", "1d")).lower(), 1.0)
            warmups.append(max(1, int(np.ceil(float(nd_daily) * float(bars_per_day)))))
    if not warmups:
        # Fallback to MA warmup if all precompute features are disabled.
        warmups.append(int(config.SPLITTER_MA_STOP))
    min_days = int(getattr(config, "SPLITTER_MIN_WARMUP_DAYS", 0))
    if min_days > 0:
        if df is not None and not df.empty:
            per_day = pd.Series(1, index=df.index).groupby(df.index.normalize()).size()
            bars_per_day = float(per_day.median()) if len(per_day) else 1.0
        else:
            bars_per_day = {
                "1m": 390.0,
                "5m": 78.0,
                "15m": 26.0,
                "1h": 6.5,
                "4h": 2.0,
                "1d": 1.0,
                "1w": 1.0 / 5.0,
                "1mo": 1.0 / 21.0,
            }.get(str(getattr(config, "TARGET_CANDLE_GRANULARITY", "1d")).lower(), 1.0)
        warmup_from_days = int(np.ceil(float(min_days) * float(bars_per_day)))
        warmups.append(max(1, warmup_from_days))
    return max(warmups)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    price_col = "close"
    periods = range(
        config.SPLITTER_MA_START,
        config.SPLITTER_MA_STOP + 1,
        config.SPLITTER_MA_STEP,
    )
    new_cols: dict[str, pd.Series] = {}
    for period in periods:
        new_cols[f"sma_{period}"] = df[price_col].rolling(period).mean().round(4)
        new_cols[f"wma_{period}"] = weighted_moving_average(df[price_col], period).round(4)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    n_periods = len(list(periods))
    print(
        f"Calculated {n_periods} SMAs and {n_periods} WMAs from {price_col}: "
        f"{config.SPLITTER_MA_START} → {config.SPLITTER_MA_STOP} (step {config.SPLITTER_MA_STEP})"
    )
    return df


def _donchian_lookbacks() -> list[int]:
    if all(hasattr(config, k) for k in ("SPLITTER_DONCHIAN_START", "SPLITTER_DONCHIAN_STOP", "SPLITTER_DONCHIAN_STEP")):
        start = int(config.SPLITTER_DONCHIAN_START)
        stop = int(config.SPLITTER_DONCHIAN_STOP)
        step = int(config.SPLITTER_DONCHIAN_STEP)
        if start < 2:
            start = 2
        if step <= 0:
            raise ValueError(f"SPLITTER_DONCHIAN_STEP must be > 0 (got {step}).")
        if stop < start:
            raise ValueError(f"SPLITTER_DONCHIAN_STOP must be >= start ({start}), got {stop}.")
        return list(range(start, stop + 1, step))
    # Backward-compat fallback.
    if all(hasattr(config, k) for k in ("DONCHIAN_SWEEP_START", "DONCHIAN_SWEEP_STOP", "DONCHIAN_SWEEP_STEP")):
        return list(
            range(
                int(config.DONCHIAN_SWEEP_START),
                int(config.DONCHIAN_SWEEP_STOP) + 1,
                int(config.DONCHIAN_SWEEP_STEP),
            )
        )
    # Last-resort fallback.
    return [int(getattr(config, "SPLITTER_DONCHIAN_STOP", 72))]


def add_daily_regime_smas(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each period n in MOD_DONCHAIAN_DAILY_SMA_DAYS_*: SMA of last daily close, shifted by one
    trading day, expanded to every bar (constant within a session day). Column: daily_regime_sma_n.
    """
    periods = _daily_regime_sma_periods()
    if not periods:
        return df
    price_col = "close"
    close = df[price_col].astype(float)
    day_series = pd.Series(df.index.normalize(), index=df.index)
    daily_close = close.groupby(day_series, sort=False).last()
    new_cols: dict[str, pd.Series] = {}
    for n in periods:
        sma_d = daily_close.rolling(int(n), min_periods=int(n)).mean().shift(1)
        mapped = sma_d.reindex(day_series.values)
        new_cols[f"daily_regime_sma_{int(n)}"] = pd.Series(mapped.to_numpy(), index=df.index).round(4)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    step = periods[1] - periods[0] if len(periods) > 1 else 1
    print(
        f"Calculated {len(periods)} daily regime SMA series (shift-1, no same-day leak): "
        f"{periods[0]} → {periods[-1]} (step {step})"
    )
    return df


def add_donchian_channels(df: pd.DataFrame) -> pd.DataFrame:
    lookbacks = _donchian_lookbacks()
    lagged_close = df["close"].shift(1)
    new_cols: dict[str, pd.Series] = {}
    for lookback in lookbacks:
        window = int(lookback) - 1
        if window <= 0:
            continue
        new_cols[f"donchian_upper_{lookback}"] = lagged_close.rolling(
            window=window, min_periods=window
        ).max().round(4)
        new_cols[f"donchian_lower_{lookback}"] = lagged_close.rolling(
            window=window, min_periods=window
        ).min().round(4)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    print(
        f"Calculated {len(lookbacks)} Donchian channel pairs from close: "
        f"{lookbacks[0]} → {lookbacks[-1]} (step {lookbacks[1]-lookbacks[0] if len(lookbacks) > 1 else 1})"
    )
    return df


def weighted_moving_average(series: pd.Series, period: int) -> pd.Series:
    """
    Weighted moving average where newer bars have higher linear weight.

    WMA_t(period) = sum(price_i * i, i=1..period) / sum(i, i=1..period)
    """
    weights = np.arange(1, period + 1, dtype=float)
    denom = float(weights.sum())
    return series.rolling(period).apply(
        lambda window, w=weights, d=denom: float(np.dot(window, w) / d),
        raw=True,
    )


def _adx_periods() -> list[int]:
    if not all(
        hasattr(config, k)
        for k in ("SPLITTER_ADX_START", "SPLITTER_ADX_STOP", "SPLITTER_ADX_STEP")
    ):
        return [14]
    start = int(config.SPLITTER_ADX_START)
    stop = int(config.SPLITTER_ADX_STOP)
    step = int(config.SPLITTER_ADX_STEP)
    if start < 2:
        start = 2
    if step <= 0:
        raise ValueError(f"SPLITTER_ADX_STEP must be > 0 (got {step}).")
    if stop < start:
        return []
    return list(range(start, stop + 1, step))


def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder / RMA smoothing: first value is mean of first `period` points."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    out[period - 1] = float(np.nanmean(arr[:period]))
    for i in range(period, n):
        out[i] = out[i - 1] - (out[i - 1] / period) + arr[i]
    return out


def _wilder_smooth_dx(dx: np.ndarray, period: int, first_dx_idx: int) -> np.ndarray:
    """Wilder smooth of DX; first meaningful DX is at `first_dx_idx` (typically period - 1)."""
    n = len(dx)
    out = np.full(n, np.nan, dtype=np.float64)
    start = first_dx_idx + period - 1
    if start >= n or first_dx_idx < 0:
        return out
    out[start] = float(np.mean(dx[first_dx_idx : first_dx_idx + period]))
    for i in range(start + 1, n):
        out[i] = out[i - 1] - (out[i - 1] / period) + dx[i]
    return out


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index (+DI, −DI, ADX) via Wilder's method.

    Returns (plus_di, minus_di, adx) aligned to the input index.
    """
    if period < 2:
        raise ValueError(f"ADX period must be >= 2, got {period}.")
    h = high.to_numpy(dtype=np.float64, copy=False)
    l = low.to_numpy(dtype=np.float64, copy=False)
    c = close.to_numpy(dtype=np.float64, copy=False)
    n = len(c)
    prev_c = np.empty(n, dtype=np.float64)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]

    tr = np.maximum(np.maximum(h - l, np.abs(h - prev_c)), np.abs(l - prev_c))
    up = np.empty(n, dtype=np.float64)
    up[0] = 0.0
    up[1:] = h[1:] - h[:-1]
    down = np.empty(n, dtype=np.float64)
    down[0] = 0.0
    down[1:] = l[:-1] - l[1:]

    plus_dm = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0.0), down, 0.0)

    atr = _wilder_smooth(tr, period)
    sm_plus = _wilder_smooth(plus_dm, period)
    sm_minus = _wilder_smooth(minus_dm, period)

    eps = 1e-12
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    m_tr = atr > eps
    np.divide(100.0 * sm_plus, atr, out=plus_di, where=m_tr)
    np.divide(100.0 * sm_minus, atr, out=minus_di, where=m_tr)
    den = plus_di + minus_di
    dx = np.zeros(n, dtype=np.float64)
    np.divide(100.0 * np.abs(plus_di - minus_di), den, out=dx, where=den > eps)
    adx = _wilder_smooth_dx(dx, period, period - 1)

    idx = high.index
    return (
        pd.Series(plus_di, index=idx),
        pd.Series(minus_di, index=idx),
        pd.Series(adx, index=idx),
    )


def add_adx_indicators(df: pd.DataFrame) -> pd.DataFrame:
    periods = _adx_periods()
    if not periods:
        print("ADX precompute skipped: SPLITTER_ADX_STOP < SPLITTER_ADX_START.")
        return df
    new_cols: dict[str, pd.Series] = {}
    for p in periods:
        pdi, mdi, adx = compute_adx(df["high"], df["low"], df["close"], int(p))
        new_cols[f"plus_di_{p}"] = pdi.round(4)
        new_cols[f"minus_di_{p}"] = mdi.round(4)
        new_cols[f"adx_{p}"] = adx.round(4)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    step = periods[1] - periods[0] if len(periods) > 1 else 1
    print(
        f"Calculated ADX (+DI/−DI) for {len(periods)} period(s): "
        f"{periods[0]} → {periods[-1]} (step {step})"
    )
    return df


def save_back(df: pd.DataFrame, ticker: str | None = None) -> None:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OHLCV_DIR / f"{_filename(ticker)}.db"
    retries = int(getattr(config, "SPLITTER_DB_WRITE_RETRIES", 6))
    retry_sec = float(getattr(config, "SPLITTER_DB_WRITE_RETRY_SEC", 5))
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with sqlite3.connect(db_path, timeout=max(5.0, retry_sec)) as con:
                con.execute(f"PRAGMA busy_timeout = {int(max(5000, retry_sec * 1000))}")
                df.reset_index().to_sql("ohlcv", con, if_exists="replace", index=False)
            print(f"Updated DB  → {db_path}")
            last_err = None
            break
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            if "database is locked" in msg and attempt < retries:
                wait = retry_sec * attempt
                print(
                    f"DB write locked (attempt {attempt}/{retries}); retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
                continue
            raise
    if last_err is not None:
        raise RuntimeError(
            "Failed to save ohlcv due to persistent DB lock. "
            "Close other processes using this DB and retry."
        ) from last_err

    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OHLCV_DIR / f"{_filename(ticker)}.csv"
    df.to_csv(csv_path)
    print(f"Updated CSV → {csv_path}")


def print_summary(df: pd.DataFrame, *, ticker_label: str = "") -> None:
    label = f" [{ticker_label}]" if ticker_label else ""
    counts = df.groupby("split").size()
    print(f"\n{'Split':<10} {'Bars':<8} {'From':<28} {'To'}{label}")
    print("-" * 70)
    for split_num, count in counts.items():
        chunk = df[df["split"] == split_num]
        split_lbl = f"0 (warmup)" if split_num == 0 else str(split_num)
        print(f"{split_lbl:<10} {count:<8} {str(chunk.index.min()):<28} {chunk.index.max()}")


def run_pipeline_for_ticker(ticker: str) -> None:
    """Load one OHLCV DB, assign splits, precompute indicators, save back."""
    sym = str(ticker).strip().upper()
    do_split = bool(getattr(config, "SPLITTER_ENABLE_SPLIT_ASSIGNMENT", True))
    do_ma = bool(getattr(config, "SPLITTER_ENABLE_MA_PRECOMPUTE", True))
    do_don = bool(getattr(config, "SPLITTER_ENABLE_DONCHIAN_PRECOMPUTE", True))
    do_adx = bool(getattr(config, "SPLITTER_ENABLE_ADX_PRECOMPUTE", True))
    do_daily_regime = bool(getattr(config, "MOD_DONCHAIAN_DAILY_SMA_REGIME_ENABLED", False))

    steps = [
        ("Load OHLCV", True),
        ("Assign splits", do_split),
        ("Precompute MA indicators", do_ma),
        ("Precompute Donchian channels", do_don),
        ("Precompute ADX", do_adx),
        ("Precompute daily regime SMAs", do_daily_regime),
        ("Save DB/CSV", True),
        ("Print summary", True),
    ]
    total_steps = sum(1 for _name, enabled in steps if enabled)
    done_steps = 0

    _progress(done_steps, total_steps, f"[{sym}] Starting data splitter")
    df = load_ohlcv(sym)
    done_steps += 1
    _progress(done_steps, total_steps, f"[{sym}] Loaded {len(df):,} OHLCV rows")

    if do_split:
        df = assign_splits(df)
        done_steps += 1
        _progress(done_steps, total_steps, f"[{sym}] Assigned walk-forward splits")
    else:
        print(f"[{sym}] Skipping split assignment (SPLITTER_ENABLE_SPLIT_ASSIGNMENT=False)")

    if do_ma:
        df = add_indicators(df)
        done_steps += 1
        _progress(done_steps, total_steps, f"[{sym}] Precomputed SMA/WMA indicators")
    else:
        print(f"[{sym}] Skipping MA precompute (SPLITTER_ENABLE_MA_PRECOMPUTE=False)")

    if do_don:
        df = add_donchian_channels(df)
        done_steps += 1
        _progress(done_steps, total_steps, f"[{sym}] Precomputed Donchian channels")
    else:
        print(f"[{sym}] Skipping Donchian precompute (SPLITTER_ENABLE_DONCHIAN_PRECOMPUTE=False)")

    if do_adx:
        df = add_adx_indicators(df)
        done_steps += 1
        _progress(done_steps, total_steps, f"[{sym}] Precomputed ADX")
    else:
        print(f"[{sym}] Skipping ADX precompute (SPLITTER_ENABLE_ADX_PRECOMPUTE=False)")

    if do_daily_regime:
        df = add_daily_regime_smas(df)
        done_steps += 1
        _progress(done_steps, total_steps, f"[{sym}] Precomputed daily regime SMA columns")
    else:
        print(f"[{sym}] Skipping daily regime SMA precompute (MOD_DONCHAIAN_DAILY_SMA_REGIME_ENABLED=False)")

    save_back(df, sym)
    done_steps += 1
    _progress(done_steps, total_steps, f"[{sym}] Saved updated dataset to DB/CSV")
    if "split" in df.columns:
        print_summary(df, ticker_label=sym)
    else:
        print(f"\n[{sym}] No split column present; split summary skipped.")
    done_steps += 1
    _progress(done_steps, total_steps, f"[{sym}] Completed")


if __name__ == "__main__":
    print_loaded_config()
    tickers = config.ohlcv_pipeline_tickers()
    print(f"\nData splitter — {len(tickers)} symbol(s): {', '.join(tickers)}\n")
    for sym in tickers:
        print(f"{'=' * 16} {sym} {'=' * 16}")
        run_pipeline_for_ticker(sym)
        print()

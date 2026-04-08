"""
MAD / MRAT standalone configuration (data pipeline, splitter, strategy, live).

Typical flow
------------
1. Set universe, bar size, and date range below.
2. ``PYTHONPATH=src python -m deepvibe_hedge.alpaca_fetcher``
3. ``PYTHONPATH=src python -m deepvibe_hedge.data_splitter``
4. ``PYTHONPATH=src python -m deepvibe_hedge.mad.backtester``
5. Live: ``PYTHONPATH=src python -m deepvibe_hedge.mad.live_bot`` (after OHLCV is current).
"""
from __future__ import annotations

from datetime import datetime, timezone

from deepvibe_hedge.nasdaq100 import nasdaq100

# -----------------------------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------------------------

MAD_UNIVERSE_TICKERS = nasdaq100

TARGET_TICKER = "BIL"

OHLCV_PIPELINE_MODE = "mad_universe"  # "mad_universe" | "target_only"


def ohlcv_pipeline_tickers() -> tuple[str, ...]:
    """Symbols for ``alpaca_fetcher`` / ``data_splitter`` (one ``data/ohlcv/{SYM}_*.db`` each)."""
    mode = OHLCV_PIPELINE_MODE.strip().lower()
    if mode == "target_only":
        base = (TARGET_TICKER.strip().upper(),)
    elif mode in ("mad_universe", "universe", "mad"):
        raw = MAD_UNIVERSE_TICKERS
        base = (
            (raw.strip().upper(),)
            if isinstance(raw, str)
            else tuple(str(x).strip().upper() for x in raw if str(x).strip())
        )
    else:
        raise ValueError(
            f"Invalid OHLCV_PIPELINE_MODE={mode!r}. Use 'mad_universe' or 'target_only'."
        )
    if MAD_REGIME_MA_ENABLED:
        rt = (MAD_REGIME_TICKER or "").strip().upper()
        if rt and rt not in base:
            base = (*base, rt)
    return base


TARGET_CANDLE_GRANULARITY = "1d"

TARGET_START_DATE = "2010-01-01"
TARGET_END_DATE = "now"
OHLCV_DOWNLOAD_END_MODE = "utc_now"


def ohlcv_download_start_utc() -> datetime:
    s = str(globals().get("TARGET_START_DATE", "2010-01-01")).strip()
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def ohlcv_download_end_utc() -> datetime:
    mode = str(globals().get("OHLCV_DOWNLOAD_END_MODE", "fixed")).strip().lower()
    if mode in ("utc_now", "now", "live"):
        return datetime.now(timezone.utc)
    if mode == "fixed":
        e = str(globals().get("TARGET_END_DATE", "2099-12-31")).strip()
        return datetime.fromisoformat(e).replace(tzinfo=timezone.utc)
    raise ValueError(
        f"Invalid OHLCV_DOWNLOAD_END_MODE={mode!r}. Use 'fixed' or 'utc_now'."
    )


ALPACA_BAR_ADJUSTMENT = "split"
BACKTEST_FEE_RATE = 0.001

# Alpaca market data feed for historical bars (``alpaca_fetcher``).
LIVE_BOT_DATA_FEED = "iex"
# Paper vs live key resolution fallbacks (``alpaca_asset``, ``alpaca_live``).
LIVE_BOT_PAPER = True
LIVE_BOT_ALLOW_SHORT = True

# -----------------------------------------------------------------------------
# Data splitter (walk-forward splits + indicator precompute on OHLCV SQLite)
# -----------------------------------------------------------------------------

SPLITTER_NUM_SPLITS = 10

SPLITTER_MA_START = 2
SPLITTER_MA_STOP = 256
SPLITTER_MA_STEP = 2

SPLITTER_DONCHIAN_START = 2
SPLITTER_DONCHIAN_STOP = 200
SPLITTER_DONCHIAN_STEP = 1

SPLITTER_ADX_START = 14
SPLITTER_ADX_STOP = 14
SPLITTER_ADX_STEP = 1

SPLITTER_ENABLE_SPLIT_ASSIGNMENT = True
SPLITTER_ENABLE_MA_PRECOMPUTE = True
SPLITTER_ENABLE_DONCHIAN_PRECOMPUTE = True
SPLITTER_ENABLE_ADX_PRECOMPUTE = True

SPLITTER_MIN_WARMUP_DAYS = 0

SPLITTER_DB_WRITE_RETRIES = 6
SPLITTER_DB_WRITE_RETRY_SEC = 5

SPLIT_PLAN_IN_SAMPLE = (1, 3, 5, 7, 9)
SPLIT_PLAN_OUT_OF_SAMPLE = (2, 4, 6, 8, 10)

# Optional daily regime SMA columns (off for MAD-only installs; splitter skips when False).
MOD_DONCHAIAN_DAILY_SMA_REGIME_ENABLED = False
MOD_DONCHAIAN_DAILY_SMA_DAYS_START = 2
MOD_DONCHAIAN_DAILY_SMA_DAYS_STOP = 200
MOD_DONCHAIAN_DAILY_SMA_DAYS_STEP = 10
MOD_DONCHAIAN_DAILY_SMA_WARMUP_EXTRA_CALENDAR_DAYS = 5

# -----------------------------------------------------------------------------
# MAD / MRAT
# -----------------------------------------------------------------------------

MAD_DIRECTION_MODE = "long_only"

MAD_SMA_SHORT = 21
MAD_SMA_LONG = 200
MAD_SMA_SHORT_GRID = (21,)
MAD_SMA_LONG_GRID = (200,)

MAD_EXIT_MA_ENABLED = False
MAD_EXIT_MA_PERIOD = 0
MAD_EXIT_MA_GRID = (0, 50, 100, 150, 200)

MAD_REGIME_MA_ENABLED = True
MAD_REGIME_TICKER = "QQQ"
MAD_REGIME_MA_GRID = (0, 50, 100, 150, 200)

MAD_LONG_SIGMA_MULT = 1.0
MAD_SHORT_SIGMA_MULT = 1.0
MAD_SYMMETRIC_SHORT_SIGMA = False

MAD_LONG_DECILE_MIN = 10
MAD_SHORT_DECILE_MAX = 1
MAD_MIN_HISTORY_BARS = 252
MAD_AGGREGATE_TO_DAILY = True

MAD_EVAL_ALL_SPLITS = True

MAD_IS_SPLITS = 6
MAD_OOS_SPLITS = 4

MAD_DASHBOARD_PORT = 8063
MAD_WF_DASHBOARD_PORT = 8064
MAD_WF_OPTIM_SPLIT = "avg"
MAD_WF_OOS_SPLIT = "all"

MAD_PERM_N = 10_000
MAD_PERM_ALPHA = 0.05
MAD_PERM_BLOCK_SIZE = 5
MAD_PERM_PORT = 8065
MAD_PERM_OPTIM_SPLIT = "avg"
MAD_PERM_IS_SPLITS = MAD_IS_SPLITS

MAD_LIVE_PAPER = True
MAD_LIVE_POLL_SECONDS = 300
MAD_LIVE_TRADE_ONLY_WHEN_MARKET_OPEN = False
MAD_LIVE_LOAD_PARAMS_FROM_DB = True

MAD_LIVE_SMA_SHORT = None
MAD_LIVE_SMA_LONG = None
MAD_LIVE_EXIT_MA = None
MAD_LIVE_REGIME_MA = 200
MAD_LIVE_REGIME_TICKER = None

MAD_LIVE_EQUITY_FRACTION = 0.98
MAD_LIVE_MAX_GROSS_USD = None
MAD_LIVE_MIN_ORDER_USD = 1.0

MAD_LIVE_OHLCV_HEALTH_CHECK = True
MAD_LIVE_HEALTH_REFERENCE_TICKER = None
MAD_LIVE_OHLCV_RECENT_REF_BARS = 60
MAD_LIVE_OHLCV_MAX_STALE_CALENDAR_DAYS = 1
MAD_LIVE_ABORT_ON_OHLCV_ISSUES = False

MAD_LIVE_REFRESH_SPLITTER_DB = False
MAD_LIVE_REFRESH_SPLITTER_ONCE_PER_UTC_DAY = True
MAD_LIVE_REFRESH_SPLITTER_ON_STARTUP = True

MAD_LIVE_EXTENDED_HOURS_ORDERS = True
MAD_LIVE_REGIME_OFF_PROXY_TICKER = "BIL"
MAD_LIVE_REGIME_OFF_CLOSE_ALL_NON_PROXY = True
MAD_LIVE_REGIME_OFF_EQUITY_FRACTION = 0.995

MAD_LIVE_ALPACA_CONNECT_RETRIES = 5
MAD_LIVE_ALPACA_CONNECT_RETRY_SEC = 2.0

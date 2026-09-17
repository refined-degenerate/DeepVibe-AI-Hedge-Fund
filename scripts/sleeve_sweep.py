"""Ad-hoc research sweep: which sleeves + weighting schemes worked best.

Loads the MAD daily panel once, then evaluates a matrix of:
  (A) sleeve compositions through the multi-index allocator (fixed schemes),
  (B) single-index SP500 / QQQ momentum baselines with and without the
      risk-off trend gate (the "original" no-sleeve momentum book), and
  (C) a weighting-scheme sweep on the best multi-slot composition.

Eval window honors config.MAD_BACKTEST_START_DATE / END_DATE.
Run:  PYTHONPATH=src python scripts/sleeve_sweep.py
"""
from __future__ import annotations

import sys
import pandas as pd

import deepvibe_hedge.config as config
from deepvibe_hedge.paths import OHLCV_DIR
from deepvibe_hedge.sp500 import sp500 as SP500
from deepvibe_hedge.nasdaq100 import nasdaq100 as NDX
from deepvibe_hedge.mad.backtester import (
    build_panel_long,
    aggregate_panel_to_daily,
    evaluate_mad,
    evaluate_mad_multi_index,
    mad_reference_ticker,
    mad_universe_tickers,
    mad_calendar_key,
    _bars_per_year_for_mad,
)

GRAN = str(config.TARGET_CANDLE_GRANULARITY)
DAILY_AGG = bool(getattr(config, "MAD_AGGREGATE_TO_DAILY", True)) and GRAN.lower() != "1d"
SHORT_W = int(config.MAD_SMA_SHORT)
LONG_W = int(config.MAD_SMA_LONG)
MIN_PRICE = 5.0
MIN_HIST = int(getattr(config, "MAD_MIN_HISTORY_BARS", 252))
FEE = float(getattr(config, "BACKTEST_FEE_RATE", 0.001))
DIR_MODE = getattr(config, "MAD_DIRECTION_MODE", "both")
EXIT_MA = int(getattr(config, "MAD_EXIT_MA_PERIOD", 0) or 0)

print("[load] building panel ...", flush=True)
_panel = build_panel_long(mad_universe_tickers(), GRAN, mad_reference_ticker(), OHLCV_DIR)
if DAILY_AGG:
    DAILY = aggregate_panel_to_daily(_panel)
else:
    DAILY = _panel.assign(date=pd.to_datetime(_panel["timestamp"], utc=True).dt.normalize()).drop(
        columns=["timestamp"], errors="ignore"
    )
BPY = _bars_per_year_for_mad(GRAN, DAILY_AGG or GRAN.lower() == "1d")

# Eval window -> eval_dates set for the single-index baselines.
_start = pd.Timestamp(config.MAD_BACKTEST_START_DATE, tz="UTC") if config.MAD_BACKTEST_START_DATE else None
_end = pd.Timestamp(config.MAD_BACKTEST_END_DATE, tz="UTC") if config.MAD_BACKTEST_END_DATE else None
_dates = pd.to_datetime(DAILY["date"], utc=True)
_mask = pd.Series(True, index=DAILY.index)
if _start is not None:
    _mask &= _dates >= _start
if _end is not None:
    _mask &= _dates <= _end
EVAL_DATES = {mad_calendar_key(d) for d in pd.DatetimeIndex(sorted(set(_dates[_mask])))}
print(f"[load] panel ready: {DAILY['ticker'].nunique()} names, eval_dates={len(EVAL_DATES)}", flush=True)

RESULTS: list[dict] = []


def _avg_weights(eval_df: pd.DataFrame) -> dict[str, float]:
    cols = [c for c in eval_df.columns if c.startswith("alloc_")]
    if not cols:
        return {}
    m = eval_df[cols].mean(axis=0)
    return {c.replace("alloc_", "").replace("__risk_off__", "RiskOff"): float(m[c]) for c in cols}


def run_multi(name: str, enabled, stock="mrat_distance", index="mrat_breadth_risk_parity",
              breadth_exp=1.0, cache=None, want_cache=False, redistribute=False):
    config.MAD_INDEX_ENABLED_ETFS = tuple(enabled)
    config.MAD_WEIGHTING_SCHEME = stock
    config.MAD_INDEX_WEIGHTING_SCHEME = index
    config.MAD_INDEX_BREADTH_EXPONENT = breadth_exp
    config.MAD_INDEX_REDISTRIBUTE_TO_SURVIVORS = redistribute
    print(f"\n=== {name}  enabled={enabled} stock={stock} index={index} redist={redistribute} ===", flush=True)
    try:
        m, ev = evaluate_mad_multi_index(
            DAILY, short_w=SHORT_W, long_w=LONG_W, min_price=MIN_PRICE, min_history=MIN_HIST,
            fee_rate=FEE, direction_mode=DIR_MODE, eval_dates=None, bars_per_year_local=BPY,
            exit_ma_period=EXIT_MA, granularity=GRAN, aggregate_to_daily=DAILY_AGG,
            _per_slot_cache=cache, _return_per_slot_cache=want_cache, _verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    !! FAILED: {exc}", flush=True)
        return None
    w = _avg_weights(ev)
    RESULTS.append({
        "name": name, "ret": m["net_total_return"], "sharpe": m["sharpe_ratio"],
        "sortino": m["sortino_ratio"], "pf": m["profit_factor"], "weights": w,
    })
    print(f"    -> ret={m['net_total_return']:+.2%} sharpe={m['sharpe_ratio']:.2f} "
          f"sortino={m['sortino_ratio']:.2f} pf={m['profit_factor']:.2f} | "
          f"{ {k: round(v,3) for k,v in w.items()} }", flush=True)
    return m.get("_per_slot_cache")


def run_single(name: str, universe, regime_ma: int, stock="mrat_distance", regime_ticker="SPY"):
    config.MAD_WEIGHTING_SCHEME = stock
    sub = DAILY[DAILY["ticker"].isin(set(universe))].copy()
    print(f"\n=== {name}  (single-index, regime_ma={regime_ma}, n={sub['ticker'].nunique()}) ===", flush=True)
    try:
        m, _ = evaluate_mad(
            sub, short_w=SHORT_W, long_w=LONG_W, min_price=MIN_PRICE, min_history=MIN_HIST,
            fee_rate=FEE, direction_mode=DIR_MODE, eval_dates=EVAL_DATES, bars_per_year_local=BPY,
            exit_ma_period=EXIT_MA, regime_ma_period=regime_ma, regime_ticker=regime_ticker,
            granularity=GRAN, aggregate_to_daily=DAILY_AGG,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    !! FAILED: {exc}", flush=True)
        return
    RESULTS.append({
        "name": name, "ret": m["net_total_return"], "sharpe": m["sharpe_ratio"],
        "sortino": m["sortino_ratio"], "pf": m["profit_factor"], "weights": {},
    })
    print(f"    -> ret={m['net_total_return']:+.2%} sharpe={m['sharpe_ratio']:.2f} "
          f"sortino={m['sortino_ratio']:.2f} pf={m['profit_factor']:.2f}", flush=True)


# ---- Reference baselines ----
run_single("SP500 momentum, NO risk-off (regime=0)", SP500, regime_ma=0)

# ---- (D) Rotate-to-survivors: thematic sleeves, share goes to ON sleeves not cash ----
# Hold index scheme at the strong mrat_distance_inv_vol; compare cash vs survivors.
IDX = "mrat_distance_inv_vol"
for comp in (("SPY", "QQQ"), ("SPY", "UFO", "PGJ"), ("SPY", "QQQ", "UFO", "PGJ")):
    tag = "+".join(comp)
    run_multi(f"{tag} · CASH (classic)", comp, index=IDX, redistribute=False)
    run_multi(f"{tag} · SURVIVORS (rotate)", comp, index=IDX, redistribute=True)
run_multi("SPY+QQQ+UFO+PGJ · SURVIVORS · breadth_rp", ("SPY", "QQQ", "UFO", "PGJ"),
          index="mrat_breadth_risk_parity", redistribute=True)

# ---- Leaderboard ----
print("\n\n" + "=" * 100, flush=True)
print("  SWEEP LEADERBOARD (sorted by Sharpe)", flush=True)
print("=" * 100, flush=True)
RESULTS.sort(key=lambda r: r["sharpe"], reverse=True)
print(f"  {'config':<42}{'return':>10}{'sharpe':>8}{'sortino':>9}{'PF':>6}", flush=True)
print("  " + "-" * 96, flush=True)
for r in RESULTS:
    print(f"  {r['name']:<42}{r['ret']:>+10.2%}{r['sharpe']:>8.2f}{r['sortino']:>9.2f}{r['pf']:>6.2f}", flush=True)
print("=" * 100, flush=True)

"""
OHLCV SQLite checks for MAD live: staleness vs reference calendar and recent missing bars.

MRAT in ``compute_mad_live_snapshot`` is driven by **closes** in each symbol DB (not precomputed
``sma_*`` columns). This module verifies panel symbols share the same **last bar date** as the
reference ticker and have no holes on the reference calendar over a recent tail window. For
``Nd`` granularities, bars are matched by **UTC calendar day** (so mixed timestamp times do not
false-flag gaps), and ref dates after the symbol's last bar are ignored when counting missing rows
(so a ref that is one session ahead does not mark every name as 1 bar short).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class SymbolOhlcvCheck:
    symbol: str
    db_path: Path
    exists: bool
    n_bars: int = 0
    last_ts: pd.Timestamp | None = None
    stale_calendar_days_vs_ref: int = 0
    missing_in_recent_tail: int = 0
    issues: list[str] = field(default_factory=list)


@dataclass
class MadOhlcvHealthReport:
    ref_ticker: str
    granularity: str
    ref_last: pd.Timestamp | None
    ref_n_bars: int
    symbols: list[SymbolOhlcvCheck]
    ok: bool
    messages: list[str]


def _read_ts_close(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with sqlite3.connect(path) as con:
        df = pd.read_sql(
            "SELECT timestamp, close FROM ohlcv ORDER BY timestamp",
            con,
            parse_dates=["timestamp"],
        )
    if df.empty:
        return df
    df = df.set_index("timestamp").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _granularity_is_daily_bar(gran: str) -> bool:
    g = str(gran).strip().lower()
    return len(g) >= 2 and g[-1] == "d" and g[:-1].isdigit()


def _close_by_utc_date(df: pd.DataFrame) -> pd.Series:
    """One row per UTC calendar day (last duplicate wins). Aligns mixed bar timestamps for daily DBs."""
    s = df["close"].astype(float)
    idx = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True)).normalize()
    out = pd.Series(s.values, index=idx)
    return out[~out.index.duplicated(keep="last")].sort_index()


def audit_mad_ohlcv_panel(
    *,
    ohlcv_dir: Path,
    granularity: str,
    ref_ticker: str,
    panel_symbols: tuple[str, ...],
    extra_symbols: tuple[str, ...] = (),
    recent_ref_bars: int = 60,
    max_stale_calendar_days: int = 1,
) -> MadOhlcvHealthReport:
    """
    Compare every symbol's last bar to the reference DB and count missing closes on the reference
    index over the last ``recent_ref_bars`` rows (daily: trading days in ref DB).

    ``max_stale_calendar_days``: allow symbol last date to lag ref by at most this many calendar days
    (0 = must match ref last UTC date).
    """
    ref = str(ref_ticker).strip().upper()
    gran = str(granularity).strip()
    ref_path = ohlcv_dir / f"{ref}_{gran}.db"
    messages: list[str] = []
    ref_df = _read_ts_close(ref_path)
    if ref_df.empty:
        return MadOhlcvHealthReport(
            ref_ticker=ref,
            granularity=gran,
            ref_last=None,
            ref_n_bars=0,
            symbols=[],
            ok=False,
            messages=[f"Reference OHLCV missing or empty: {ref_path}"],
        )

    daily = _granularity_is_daily_bar(gran)
    if daily:
        ref_close = _close_by_utc_date(ref_df)
        ref_last = pd.Timestamp(ref_close.index[-1])
        recent_axis = ref_close.index[-max(1, int(recent_ref_bars)) :]
    else:
        ref_last = pd.Timestamp(ref_df.index[-1]).tz_convert("UTC").normalize()
        recent_axis = ref_df.index[-max(1, int(recent_ref_bars)) :]

    want = sorted(set(str(s).strip().upper() for s in panel_symbols if str(s).strip()) | set(extra_symbols))
    checks: list[SymbolOhlcvCheck] = []
    ok = True

    for sym in want:
        p = ohlcv_dir / f"{sym}_{gran}.db"
        chk = SymbolOhlcvCheck(symbol=sym, db_path=p, exists=p.exists())
        if not chk.exists:
            chk.issues.append("DB file missing")
            ok = False
            checks.append(chk)
            continue

        sdf = _read_ts_close(p)
        if sdf.empty:
            chk.issues.append("ohlcv table empty")
            ok = False
            checks.append(chk)
            continue

        chk.n_bars = len(sdf)
        if daily:
            sym_close = _close_by_utc_date(sdf)
            last = pd.Timestamp(sym_close.index[-1])
        else:
            last = pd.Timestamp(sdf.index[-1]).tz_convert("UTC").normalize()
        chk.last_ts = last
        stale = int((ref_last - last).days)
        chk.stale_calendar_days_vs_ref = max(0, stale)
        if stale > int(max_stale_calendar_days):
            chk.issues.append(
                f"stale vs ref: last={last.date()} ref_last={ref_last.date()} (Δ {stale} cal days)"
            )
            ok = False

        if daily:
            # Only ref sessions on or before the symbol's last bar count as "expected" (ref can be 1 bar ahead).
            recent_ok = recent_axis[recent_axis <= last]
            aligned = sym_close.reindex(recent_ok)
            compare_n = len(recent_ok)
        else:
            close = sdf["close"].astype(float)
            aligned = close.reindex(recent_axis)
            compare_n = len(recent_axis)
        miss = int(aligned.isna().sum())
        chk.missing_in_recent_tail = miss
        if miss > 0:
            chk.issues.append(
                f"{miss} missing bar(s) on ref calendar in last {compare_n} comparable ref row(s)"
            )
            ok = False

        checks.append(chk)

    if ok:
        messages.append(
            f"OHLCV OK: ref={ref} last={ref_last.date()} | checked {len(checks)} symbol(s) "
            f"(staleness ≤{max_stale_calendar_days} cal day(s), no gaps in last {len(recent_axis)} ref bars)."
        )
    else:
        bad = [c for c in checks if c.issues]
        messages.append(f"OHLCV issues on {len(bad)} / {len(checks)} symbol(s) — run fetcher + splitter.")
        for c in bad[:25]:
            messages.append(f"  {c.symbol}: {'; '.join(c.issues)}")
        if len(bad) > 25:
            messages.append(f"  ... +{len(bad) - 25} more")

    return MadOhlcvHealthReport(
        ref_ticker=ref,
        granularity=gran,
        ref_last=ref_last,
        ref_n_bars=len(ref_df),
        symbols=checks,
        ok=ok,
        messages=messages,
    )


def print_health_report(report: MadOhlcvHealthReport) -> None:
    for line in report.messages:
        print(line)

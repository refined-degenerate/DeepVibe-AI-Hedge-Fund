"""
Alpaca order helpers for MAD live trading (extracted from the parent project's Donchian live bot).

MAD only needs net-quantity reconcile, market clock, latest trade price, and shortability checks.
"""
from __future__ import annotations

import math

from alpaca.common.exceptions import APIError
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from deepvibe_hedge import config
from deepvibe_hedge.alpaca_asset import _alpaca_trading_keys

_SYMBOL_SELL_SHORT_BLOCKED: set[str] = set()


def _alpaca_short_sale_forbidden(exc: BaseException) -> bool:
    raw = str(exc)
    low = raw.lower()
    return "42210000" in raw or "cannot be sold short" in low


def _get_current_qty(trading_client: TradingClient, symbol: str) -> float:
    try:
        pos = trading_client.get_open_position(symbol)
    except APIError as exc:
        if "position does not exist" in str(exc).lower():
            return 0.0
        raise
    return float(pos.qty)


def _apply_live_short_constraints(
    trading_client: TradingClient,
    symbol: str,
    desired_qty_net: int,
) -> tuple[int, str]:
    if desired_qty_net >= 0:
        return desired_qty_net, ""
    sym = symbol.strip().upper()
    if sym in _SYMBOL_SELL_SHORT_BLOCKED:
        print(
            f"  [{symbol}] short target {desired_qty_net:+d} suppressed "
            "(Alpaca rejected sell-to-open earlier this session; restart bot to retry)."
        )
        return 0, " | broker_no_short→0"
    if not bool(getattr(config, "LIVE_BOT_ALLOW_SHORT", True)):
        print(
            f"  [{symbol}] net target {desired_qty_net:+d} (short); LIVE_BOT_ALLOW_SHORT=False → flat."
        )
        return 0, " | short_disabled→0"
    try:
        asset = trading_client.get_asset(symbol)
    except APIError as exc:
        print(f"  [{symbol}] get_asset failed ({exc}); short order may still be attempted.")
        return desired_qty_net, ""
    if not bool(asset.shortable):
        print(
            f"  [{symbol}] net target {desired_qty_net:+d} (short) skipped: Alpaca asset.shortable=false "
            f"for {symbol!r}. Paper/live still require a shortable ticker for sell-to-open."
        )
        return 0, " | not_shortable→0"
    if not bool(asset.easy_to_borrow):
        print(
            f"  [{symbol}] short warning: easy_to_borrow=false; broker may reject the sell anyway."
        )
    return desired_qty_net, ""


def _latest_stock_trade_price(symbol: str, *, paper: bool) -> float:
    sym = symbol.strip().upper()
    key, secret = _alpaca_trading_keys(paper=paper)
    dc = StockHistoricalDataClient(api_key=key, secret_key=secret)
    out = dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=sym))
    if isinstance(out, dict):
        tr = out.get(sym)
        if tr is None and out:
            tr = next(iter(out.values()))
    else:
        tr = out
    if tr is None:
        raise RuntimeError(f"No latest trade returned for {sym!r}")
    return float(tr.price)


def _extended_hours_limit_price(reference_price: float, *, buy: bool) -> float:
    r = float(reference_price)
    if not math.isfinite(r) or r <= 0:
        raise ValueError(f"invalid reference_price for extended-hours limit: {reference_price!r}")
    if buy:
        return round(r * 1.01, 2)
    return round(max(r * 0.99, 0.01), 2)


def _submit_delta_order(
    trading_client: TradingClient,
    symbol: str,
    delta_qty: int,
    *,
    extended_hours: bool = False,
    reference_price: float | None = None,
    paper: bool | None = None,
) -> None:
    if delta_qty == 0:
        return
    side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
    if extended_hours:
        pb = bool(getattr(config, "LIVE_BOT_PAPER", True)) if paper is None else paper
        ref = reference_price
        if ref is None or not math.isfinite(float(ref)) or float(ref) <= 0:
            ref = _latest_stock_trade_price(symbol, paper=pb)
        limit_px = _extended_hours_limit_price(float(ref), buy=(delta_qty > 0))
        order = LimitOrderRequest(
            symbol=symbol,
            qty=abs(int(delta_qty)),
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_px,
            extended_hours=True,
        )
    else:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=abs(int(delta_qty)),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
    try:
        trading_client.submit_order(order_data=order)
    except APIError as exc:
        if delta_qty < 0:
            print(
                "  [order] SELL rejected (sell-to-open or larger sell). Typical causes: symbol not "
                "shortable, not easy to borrow, shorting disabled on the account, or insufficient "
                f"buying power for margin. Raw API error: {exc}"
            )
        else:
            print(f"  [order] BUY rejected: {exc}")
        raise


def _reconcile_symbol_net_qty(
    trading_client: TradingClient,
    symbol: str,
    desired_qty_net: int,
    *,
    _short_retry: bool = True,
    extended_hours: bool = False,
    reference_price: float | None = None,
    paper: bool | None = None,
) -> tuple[int, int, int]:
    current_qty = _get_current_qty(trading_client, symbol)
    current_qty_int = int(round(current_qty))
    delta_qty = int(desired_qty_net) - current_qty_int
    try:
        _submit_delta_order(
            trading_client,
            symbol,
            delta_qty,
            extended_hours=extended_hours,
            reference_price=reference_price,
            paper=paper,
        )
    except APIError as exc:
        if (
            _short_retry
            and delta_qty < 0
            and _alpaca_short_sale_forbidden(exc)
        ):
            _SYMBOL_SELL_SHORT_BLOCKED.add(symbol.strip().upper())
            print(
                f"  [{symbol}] sell-to-open rejected by Alpaca ({exc}). "
                "Symbol is cached as no-short for this session; targeting flat/long only."
            )
            return _reconcile_symbol_net_qty(
                trading_client,
                symbol,
                max(0, int(desired_qty_net)),
                _short_retry=False,
                extended_hours=extended_hours,
                reference_price=reference_price,
                paper=paper,
            )
        raise
    return current_qty_int, int(desired_qty_net), delta_qty


def _market_is_open(trading_client: TradingClient) -> bool:
    try:
        return bool(trading_client.get_clock().is_open)
    except Exception:
        return False

"""
Map ``config.ALPACA_BAR_ADJUSTMENT`` to Alpaca ``Adjustment`` for historical bars.

Alpaca applies vendor-side factors so the series is continuous across splits (and optionally
dividends). Use ``split`` for research so MA/returns are not distorted by mechanical price jumps.
"""
from __future__ import annotations

from alpaca.data.enums import Adjustment

from deepvibe_hedge import config

_ADJUSTMENT_ALIASES: dict[str, Adjustment] = {
    "raw": Adjustment.RAW,
    "none": Adjustment.RAW,
    "unadjusted": Adjustment.RAW,
    "split": Adjustment.SPLIT,
    "splits": Adjustment.SPLIT,
    "dividend": Adjustment.DIVIDEND,
    "dividends": Adjustment.DIVIDEND,
    "all": Adjustment.ALL,
}


def historical_bar_adjustment() -> Adjustment:
    raw = str(getattr(config, "ALPACA_BAR_ADJUSTMENT", "split")).strip().lower()
    adj = _ADJUSTMENT_ALIASES.get(raw)
    if adj is None:
        allowed = ", ".join(sorted({k for k in _ADJUSTMENT_ALIASES}))
        raise ValueError(
            f"Invalid ALPACA_BAR_ADJUSTMENT={raw!r}. Use one of: {allowed}"
        )
    return adj

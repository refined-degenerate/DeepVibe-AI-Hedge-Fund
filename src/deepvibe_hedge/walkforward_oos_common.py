from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


def read_explicit_split_plan_from_config() -> tuple[list[int] | None, list[int] | None]:
    """
    When SPLIT_PLAN_IN_SAMPLE and SPLIT_PLAN_OUT_OF_SAMPLE are both non-empty, every
    strategy should use those split ids. When both are empty, return (None, None) for
    legacy *._IS_SPLITS + trailing *._OOS_SPLITS behavior.
    """
    from deepvibe_hedge import config

    raw_is = getattr(config, "SPLIT_PLAN_IN_SAMPLE", ()) or ()
    raw_oos = getattr(config, "SPLIT_PLAN_OUT_OF_SAMPLE", ()) or ()
    if not raw_is and not raw_oos:
        return None, None
    if not raw_is or not raw_oos:
        raise ValueError(
            "SPLIT_PLAN_IN_SAMPLE and SPLIT_PLAN_OUT_OF_SAMPLE must both be non-empty "
            "tuples/lists, or both empty (). Partial configuration is not allowed."
        )
    is_ids = sorted({int(x) for x in raw_is})
    oos_ids = sorted({int(x) for x in raw_oos})
    return is_ids, oos_ids


def normalize_selector(raw: str, *, allow_avg: bool, allow_all: bool) -> str:
    val = str(raw).strip().lower()
    if allow_avg and val == "avg":
        return "avg"
    if allow_all and val == "all":
        return "all"
    try:
        return str(int(val))
    except Exception as exc:
        opts: list[str] = []
        if allow_avg:
            opts.append("'avg'")
        if allow_all:
            opts.append("'all'")
        opts.append("an integer split id")
        raise ValueError(f"Invalid split selector {raw!r}. Use {', '.join(opts)}.") from exc


@dataclass(frozen=True)
class SplitPlan:
    is_splits: list[int]
    reserved_oos_splits: list[int]


def resolve_split_plan(
    df: pd.DataFrame,
    *,
    is_target: int,
    oos_reserved: int,
    label: str,
    explicit_is: Sequence[int] | None = None,
    explicit_oos: Sequence[int] | None = None,
) -> SplitPlan:
    if "split" not in df.columns:
        raise RuntimeError("No split column found in dataset. Run data_splitter.py first.")
    available = sorted(int(x) for x in df[df["split"] > 0]["split"].unique().tolist())
    if not available:
        raise RuntimeError("No non-warmup splits found in dataset.")

    if explicit_is is not None and explicit_oos is not None:
        is_ids = sorted({int(x) for x in explicit_is})
        oos_ids = sorted({int(x) for x in explicit_oos})
        if not is_ids or not oos_ids:
            raise ValueError(f"{label}: explicit split plan lists must be non-empty.")
        avail_set = set(available)
        miss_is = sorted(x for x in is_ids if x not in avail_set)
        miss_oos = sorted(x for x in oos_ids if x not in avail_set)
        if miss_is or miss_oos:
            raise ValueError(
                f"{label}: explicit split ids missing from OHLCV non-warmup splits. "
                f"missing IS={miss_is or 'none'}, missing OOS={miss_oos or 'none'}; "
                f"available={available}"
            )
        overlap = sorted(set(is_ids) & set(oos_ids))
        if overlap:
            raise ValueError(
                f"{label}: SPLIT_PLAN_IN_SAMPLE and SPLIT_PLAN_OUT_OF_SAMPLE must be disjoint; "
                f"overlap={overlap}"
            )
        union = set(is_ids) | set(oos_ids)
        extra = sorted(avail_set - union)
        if extra:
            print(
                f"[{label}] Note: explicit IS/OOS plan leaves {len(extra)} non-warmup split(s) "
                f"unassigned (not in IS or OOS): {extra}"
            )
        return SplitPlan(is_splits=is_ids, reserved_oos_splits=oos_ids)

    if is_target <= 0:
        raise ValueError(f"{label}_IS_SPLITS must be >= 1 (got {is_target}).")
    if oos_reserved < 0:
        raise ValueError(f"{label}_OOS_SPLITS must be >= 0 (got {oos_reserved}).")
    if oos_reserved >= len(available):
        raise ValueError(
            f"{label}_OOS_SPLITS={oos_reserved} leaves no in-sample data; "
            f"available non-warmup splits={len(available)}."
        )

    reserved_oos = available[-oos_reserved:] if oos_reserved > 0 else []
    is_pool = available[:-oos_reserved] if oos_reserved > 0 else available
    if not is_pool:
        raise ValueError("No in-sample splits remain after OOS reservation.")

    if is_target > len(is_pool):
        print(
            f"Requested {label}_IS_SPLITS={is_target}, but only {len(is_pool)} split(s) "
            f"are available after reserving {oos_reserved} OOS split(s). Using {len(is_pool)}."
        )
    is_splits = is_pool[: min(is_target, len(is_pool))]
    return SplitPlan(is_splits=is_splits, reserved_oos_splits=reserved_oos)


def select_oos_splits(reserved_oos_splits: list[int], selected: str) -> list[int]:
    if not reserved_oos_splits:
        raise ValueError("No reserved OOS splits configured; set *_OOS_SPLITS > 0 first.")
    key = normalize_selector(selected, allow_avg=False, allow_all=True)
    if key == "all":
        return list(reserved_oos_splits)
    split_id = int(key)
    if split_id not in set(int(s) for s in reserved_oos_splits):
        raise ValueError(
            f"Requested OOS split {split_id} is not reserved. Reserved OOS splits: {reserved_oos_splits}"
        )
    return [split_id]

"""Small aggregation helpers shared by executable notebook cells."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def mean_ci95(
    values: Sequence[float],
    *,
    lower: float = 0.0,
    upper: float | None = None,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(array.mean())
    if array.size == 1:
        return mean, mean, mean
    half_width = 1.96 * float(array.std(ddof=1)) / np.sqrt(array.size)
    low = max(lower, mean - half_width)
    high = mean + half_width if upper is None else min(upper, mean + half_width)
    return mean, low, high


def summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "count": int(array.size),
    }


def off_diagonal(dimension: int) -> np.ndarray:
    return ~np.eye(dimension, dtype=bool)


def trajectory_metrics(
    estimate: dict,
    truth: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> dict[str, float]:
    point = np.asarray(estimate["information_flow"], dtype=float)
    critical = np.asarray(estimate["statistics"]["p95_critical_value"], dtype=float)
    diagonal = np.eye(truth.shape[0], dtype=bool)
    null = (np.abs(truth) <= tolerance) & ~diagonal
    active = (np.abs(truth) > tolerance) & ~diagonal
    rejected = np.abs(point) > critical
    covered = (truth >= point - critical) & (truth <= point + critical)
    return {
        "type1_error": float(rejected[null].mean()),
        "null_coverage": float(covered[null].mean()),
        "power": float(rejected[active].mean()),
        "rmse": float(np.sqrt(np.mean((point[active] - truth[active]) ** 2))),
    }


def autocorrelation_diagnostics(
    series: np.ndarray,
    segments: list[list[int]],
) -> dict[str, float]:
    lag_one = []
    for block in segments:
        block_mean = series[:, block].mean(axis=1)
        if np.std(block_mean[:-1]) < 1e-12 or np.std(block_mean[1:]) < 1e-12:
            lag_one.append(0.0)
        else:
            lag_one.append(float(np.corrcoef(block_mean[:-1], block_mean[1:])[0, 1]))
    cross = np.corrcoef(series[:-1].T, series[1:].T)
    dimension = series.shape[1]
    return {
        "mean_seg_lag1_autocorr": float(np.mean(lag_one)),
        "max_abs_cross_lag1_corr": float(
            np.max(np.abs(cross[:dimension, dimension:]))
        ),
    }


def summarize_time_series_rows(
    rows: list[dict],
    grouping: tuple[str, ...],
) -> list[dict]:
    metric_keys = (
        "type1_error",
        "null_coverage",
        "power",
        "rmse",
        "mean_seg_lag1_autocorr",
        "max_abs_cross_lag1_corr",
        "hac_max_lag_used",
    )
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in grouping), []).append(row)
    result = []
    for group_values, selected in grouped.items():
        entry = dict(zip(grouping, group_values))
        entry["n"] = len(selected)
        for key in metric_keys:
            values = [row[key] for row in selected if row.get(key) is not None]
            if not values:
                continue
            entry[key] = summary(values)
            by_repetition: dict[int, list[float]] = {}
            for row in selected:
                if row.get(key) is not None:
                    by_repetition.setdefault(int(row["rep"]), []).append(float(row[key]))
            entry[f"{key}_across_repeats"] = summary(
                np.mean(repetition_values)
                for repetition_values in by_repetition.values()
            )
        result.append(entry)
    return result


__all__ = [
    "autocorrelation_diagnostics",
    "mean_ci95",
    "off_diagonal",
    "summary",
    "summarize_time_series_rows",
    "trajectory_metrics",
]

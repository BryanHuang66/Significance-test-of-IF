"""Experiment-only plugins kept outside the maintained :mod:`lkif` package."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov
from scipy.special import ndtr


def to_index_segments(
    segments: Sequence[Sequence[int] | tuple[int, int]],
) -> list[list[int]]:
    """Convert legacy half-open tuples to explicit variable-index lists."""
    converted: list[list[int]] = []
    for segment in segments:
        if isinstance(segment, tuple):
            if len(segment) != 2:
                raise ValueError(f"Tuple segment must be (start, stop): {segment}")
            start, stop = map(int, segment)
            if start < 0 or stop <= start:
                raise ValueError(f"Invalid half-open segment: {segment}")
            indices = list(range(start, stop))
        else:
            indices = [int(index) for index in segment]
            if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
                raise ValueError(f"Invalid index segment: {segment}")
        converted.append(indices)
    if not converted:
        raise ValueError("At least one segment is required")
    return converted


def _blocks(matrix: np.ndarray, segments: list[list[int]]) -> np.ndarray:
    result = np.empty((len(segments), len(segments)), dtype=object)
    for i, rows in enumerate(segments):
        for j, cols in enumerate(segments):
            result[i, j] = np.asarray(matrix)[np.ix_(rows, cols)]
    return result


def _block_inverses(blocks: np.ndarray) -> np.ndarray:
    result = np.empty(len(blocks), dtype=object)
    for i in range(len(blocks)):
        diagonal = (blocks[i, i] + blocks[i, i].T) / 2
        result[i] = np.linalg.pinv(diagonal)
    return result


def _inverse_with_ridge(
    matrix: np.ndarray,
    ridge_lambda: float | str = 0.0,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if isinstance(ridge_lambda, str):
        if ridge_lambda.lower() != "auto":
            raise ValueError(f"Unknown ridge setting: {ridge_lambda}")
        if np.linalg.cond(matrix) <= 1e3:
            ridge_lambda = 0.0
        else:
            scale = np.max(np.abs(np.diag(matrix))) or 1.0
            ridge_lambda = 1.0
            for candidate in np.logspace(-8, 0, 200):
                regularized = matrix + candidate * scale * np.eye(matrix.shape[0])
                if np.linalg.cond(regularized) <= 1e3:
                    ridge_lambda = float(candidate)
                    break
    if float(ridge_lambda) > 0:
        scale = np.max(np.diag(matrix))
        if scale == 0 or not np.isfinite(scale):
            scale = 1.0
        matrix = matrix + float(ridge_lambda) * scale * np.eye(matrix.shape[0])
    inverse = np.linalg.pinv(matrix)
    return (inverse + inverse.T) / 2


def _true_information_flow_and_std(
    A: np.ndarray,
    B: np.ndarray,
    Sigma: np.ndarray,
    segments: list[list[int]],
    dt: float,
    deg_freedom: int,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    cov_blocks = _blocks(Sigma, segments)
    diag_inv_cov = _block_inverses(cov_blocks)
    inv_cov = _blocks(_inverse_with_ridge(Sigma, ridge_lambda), segments)
    drift = _blocks(A, segments)
    errors = _blocks(B @ B.T / dt, segments)
    information_flow = np.zeros((len(segments), len(segments)), dtype=float)
    std = np.zeros_like(information_flow)

    for i in range(len(segments)):
        for j in range(len(segments)):
            information_flow[i, j] = np.trace(
                drift[i, j] @ cov_blocks[j, i] @ diag_inv_cov[i]
            )
            temp = cov_blocks[i, j].T @ diag_inv_cov[i]
            variance_cov = np.trace(
                drift[i, j].T
                @ diag_inv_cov[i]
                @ drift[i, j]
                @ (
                    cov_blocks[j, j]
                    - cov_blocks[j, i] @ diag_inv_cov[i] @ cov_blocks[i, j]
                )
            )
            variance_reg = np.trace(
                temp.T @ inv_cov[j, j] @ temp @ errors[i, i]
            )
            variance = float(variance_cov + variance_reg) / deg_freedom
            std[i, j] = np.sqrt(max(variance, 0.0))
    return information_flow, std


def _origin_std_from_truth(
    A: np.ndarray,
    Sigma: np.ndarray,
    errors: np.ndarray,
    segments: list[list[int]],
    deg_freedom: int,
    ridge_lambda: float,
) -> np.ndarray:
    cov_blocks = _blocks(Sigma, segments)
    diag_inv_cov = _block_inverses(cov_blocks)
    inv_cov = _blocks(_inverse_with_ridge(Sigma, ridge_lambda), segments)
    error_blocks = _blocks(errors, segments)
    result = np.zeros((len(segments), len(segments)), dtype=float)
    for i in range(len(segments)):
        for j in range(len(segments)):
            temp = cov_blocks[i, j].T @ diag_inv_cov[i]
            variance = np.trace(
                temp.T @ inv_cov[j, j] @ temp @ error_blocks[i, i]
            )
            result[i, j] = np.sqrt(max(float(variance) / deg_freedom, 0.0))
    return result


def _compute_origin_standard_error(
    model,
    segments: list[list[int]],
    ridge_lambda: float | str = 1e-6,
) -> np.ndarray:
    """Historical regression-only standard error used by the paper."""
    covariance = np.asarray(model.cov, dtype=float)
    cov_blocks = _blocks(covariance, segments)
    diag_inv_cov = _block_inverses(cov_blocks)
    inv_cov = _blocks(_inverse_with_ridge(covariance, ridge_lambda), segments)
    errors = _blocks(np.asarray(model.error_square_mean, dtype=float), segments)
    result = np.zeros((len(segments), len(segments)), dtype=float)
    for i in range(len(segments)):
        for j in range(len(segments)):
            temp = cov_blocks[i, j].T @ diag_inv_cov[i]
            variance = np.trace(temp.T @ inv_cov[j, j] @ temp @ errors[i, i])
            result[i, j] = np.sqrt(max(float(variance) / model.deg_freedom, 0.0))
    return result


def _prepare_dataset(
    series_list: list[np.ndarray],
    lag_list: Sequence[int],
    dt: float,
    segments: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    lags = sorted(int(lag) for lag in lag_list)
    if not lags or lags[0] < 1:
        raise ValueError("lag_list must contain positive integers")
    lag_max = lags[-1]
    deltas: list[np.ndarray] = []
    regressors: list[np.ndarray] = []
    for series in series_list:
        series = np.asarray(series, dtype=float)
        if series.ndim != 2 or series.shape[0] <= lag_max:
            raise ValueError("Each series must have shape (T, d) with enough rows")
        deltas.append((series[lag_max:] - series[lag_max - 1 : -1]) / dt)
        regressors.append(
            np.hstack([series[lag_max - lag : -lag] for lag in lags])
        )
    n_variables = series_list[0].shape[1]
    processed_segments = [list(segment) for segment in segments]
    for lag_index in range(1, len(lags)):
        offset = n_variables * lag_index
        processed_segments.extend(
            [[index + offset for index in segment] for segment in segments]
        )
    return np.vstack(deltas), np.vstack(regressors), processed_segments


def bootstrap_estimate(
    ts_data_list: list[np.ndarray] | np.ndarray,
    dt: float = 1.0,
    lag_list: Sequence[int] = (1,),
    segments: Sequence[Sequence[int] | tuple[int, int]] | None = None,
    bootstrap_num: int = 1000,
    output_all: bool = False,
    seed: int | None = None,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> dict:
    """Bootstrap LK information flow using iid lagged-row resampling."""
    if not isinstance(ts_data_list, list):
        ts_data_list = [np.asarray(ts_data_list)]
    n_variables = ts_data_list[0].shape[1]
    if segments is None:
        segments = [(index, index + 1) for index in range(n_variables)]
    index_segments = to_index_segments(segments)
    delta, regressors, processed_segments = _prepare_dataset(
        ts_data_list, lag_list, dt, index_segments
    )
    centered = regressors - np.mean(regressors, axis=0, keepdims=True)
    n_rows = centered.shape[0]
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")
    random_source = rng if rng is not None else np.random.default_rng(seed)
    flow_list: list[np.ndarray] = []

    for _ in range(int(bootstrap_num)):
        indices = random_source.choice(n_rows, size=n_rows, replace=True)
        x_sample = centered[indices]
        delta_sample = delta[indices]
        covariance = x_sample.T @ x_sample / (n_rows - 1)
        drift = np.linalg.lstsq(x_sample, delta_sample, rcond=None)[0].T
        cov_blocks = _blocks(covariance, processed_segments)
        drift_blocks = _blocks(drift, processed_segments)[: len(index_segments), :]
        diag_inv_cov = _block_inverses(cov_blocks)
        flow = np.zeros((len(index_segments), len(processed_segments)), dtype=float)
        for i in range(len(index_segments)):
            for j in range(len(processed_segments)):
                flow[i, j] = np.trace(
                    drift_blocks[i, j] @ cov_blocks[j, i] @ diag_inv_cov[i]
                )
        flow_list.append(flow)

    if not flow_list:
        raise ValueError("bootstrap_num must be positive")
    flow_array = np.asarray(flow_list)
    return {
        "bootstrap_information_flow_mean": np.mean(flow_array, axis=0),
        "bootstrap_information_flow_std": np.std(flow_array, axis=0),
        "bootstrap_information_flow_list": flow_list if output_all else None,
    }


def real_information_flow_linear_case(
    det_mat: np.ndarray,
    sto_mat: np.ndarray,
    deg_freedom: int | None = None,
    segments: Sequence[Sequence[int] | tuple[int, int]] | None = None,
    discrete_lyapunov: bool = True,
    dt: float = 1.0,
    ridge_lambda: float = 0.0,
) -> dict:
    """Compute analytical LK information flow for a linear stochastic system."""
    A = np.asarray(det_mat, dtype=float)
    B = np.asarray(sto_mat, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or B.shape != A.shape:
        raise ValueError("det_mat and sto_mat must be square matrices of equal shape")
    if segments is None:
        segments = [(index, index + 1) for index in range(A.shape[0])]
    index_segments = to_index_segments(segments)
    Q = B @ B.T
    if discrete_lyapunov:
        Sigma = solve_discrete_lyapunov(np.eye(A.shape[0]) + dt * A, dt * Q)
    else:
        Sigma = solve_continuous_lyapunov(A, -Q)

    degrees = int(deg_freedom) if deg_freedom is not None else 1
    information_flow, std = _true_information_flow_and_std(
        A, B, Sigma, index_segments, dt, degrees, ridge_lambda
    )
    cov_blocks = _blocks(Sigma, index_segments)
    diag_inv_cov = _block_inverses(cov_blocks)
    q_blocks = _blocks(Q, index_segments)
    noise = np.zeros((len(index_segments), 1), dtype=float)
    for i in range(len(index_segments)):
        noise[i, 0] = 0.5 * np.trace(q_blocks[i, i] @ diag_inv_cov[i])
    normalizer = np.sum(np.abs(information_flow), axis=1, keepdims=True) + np.abs(noise)
    result = {
        "information_flow": information_flow,
        "normalized_information_flow": information_flow / normalizer,
        "segments": index_segments,
        "lag_list": [1],
    }
    if deg_freedom is not None:
        result.update(
            information_flow_std=std,
            information_flow_std_origin=_origin_std_from_truth(
                A, Sigma, Q / dt, index_segments, degrees, ridge_lambda
            ),
            statistics={
                "p99_critical_value": std * 2.5758293035489004,
                "p95_critical_value": std * 1.959963984540054,
                "p90_critical_value": std * 1.6448536269514722,
                "p": 2.0
                * (1.0 - ndtr(np.abs(information_flow) / np.maximum(std, 1e-15))),
            },
        )
    return result


__all__ = ["bootstrap_estimate", "real_information_flow_linear_case"]

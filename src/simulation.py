"""Authoritative linear-SDE simulation helpers used by the paper notebook.

The system builder and simulator preserve the algorithms in the archived
``exp/dynamics_builder.py`` and ``exp/simulator.py``.  They live here only so
the notebook has no dependency on the author's archival TeX directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


FlowKind = Literal["self", "structural_zero", "cancellation_zero", "active"]


def gaussian_samples(
    rng: np.random.Generator,
    mean: np.ndarray,
    covariance: np.ndarray,
    size: int | None = None,
) -> np.ndarray:
    """Draw with the same NumPy routine used by the archived experiments.

    The exact factorization matters for seeded Monte Carlo reproduction: a
    mathematically equivalent eigendecomposition consumes the same number of
    random values but produces a different realized sample.
    """
    return rng.multivariate_normal(mean, covariance, size=size)


@dataclass
class DynamicalSystem:
    A: np.ndarray
    S: np.ndarray
    Sigma: np.ndarray
    subspace_information_flow: np.ndarray
    flow_kinds: list[list[FlowKind]]
    segments: list[list[int]]


def _normalize_segments(dimension: int, segments: list[list[int]]) -> list[list[int]]:
    normalized = [sorted(int(index) for index in block) for block in segments]
    flat = [index for block in normalized for index in block]
    if sorted(flat) != list(range(dimension)):
        raise ValueError("segments must partition range(dimension) exactly")
    return normalized


def subspace_information_flow(
    A: np.ndarray,
    S: np.ndarray,
    segments: list[list[int]],
) -> np.ndarray:
    blocks = [np.asarray(block, dtype=int) for block in segments]
    flows = np.zeros((len(blocks), len(blocks)), dtype=float)
    for i, block_i in enumerate(blocks):
        sii_inverse = np.linalg.inv(S[np.ix_(block_i, block_i)])
        for j, block_j in enumerate(blocks):
            if i != j:
                flows[i, j] = float(
                    np.trace(
                        A[np.ix_(block_i, block_j)]
                        @ S[np.ix_(block_j, block_i)]
                        @ sii_inverse
                    )
                )
    return flows


def build_dynamical_system(
    dimension: int,
    segments: list[list[int]],
    *,
    seed: int | None = None,
    max_tries: int = 5000,
    structural_zero_probability: float = 0.35,
    cancellation_zero_probability: float = 0.30,
    diagonal_damping: tuple[float, float] = (0.9, 1.6),
    cross_scale: float = 0.16,
    covariance_jitter: float = 0.5,
    sigma_min_eigenvalue: float = 1e-8,
    stability_margin: float = 0.08,
    min_active_flow_magnitude: float = 0.0,
) -> DynamicalSystem:
    """Build the exact structural/cancellation/active system family from exp/."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    segments = _normalize_segments(dimension, segments)
    if structural_zero_probability + cancellation_zero_probability >= 1.0:
        raise ValueError("zero-flow probabilities must sum to less than one")

    rng = np.random.default_rng(seed)
    blocks = [np.asarray(block, dtype=int) for block in segments]
    for _ in range(max_tries):
        raw = rng.normal(size=(dimension, dimension))
        base = raw @ raw.T + covariance_jitter * np.eye(dimension)
        shared = rng.normal(size=(dimension, max(2, len(segments))))
        S = base + 0.35 * (shared @ shared.T)
        S = 0.5 * (S + S.T)
        S += covariance_jitter * np.eye(dimension)

        A = np.zeros((dimension, dimension), dtype=float)
        kinds: list[list[FlowKind]] = [
            ["self" for _ in blocks] for _ in blocks
        ]
        for block_i in blocks:
            local_basis = rng.normal(size=(len(block_i), len(block_i)))
            local_spd = local_basis @ local_basis.T + np.eye(len(block_i))
            damping = rng.uniform(*diagonal_damping)
            A[np.ix_(block_i, block_i)] = -(
                damping * np.eye(len(block_i)) + 0.12 * local_spd
            )

        for i, block_i in enumerate(blocks):
            sii_inverse = np.linalg.inv(S[np.ix_(block_i, block_i)])
            for j, block_j in enumerate(blocks):
                if i == j:
                    continue
                draw = rng.random()
                if draw < structural_zero_probability:
                    kinds[i][j] = "structural_zero"
                    continue
                reference = S[np.ix_(block_j, block_i)] @ sii_inverse
                if draw < structural_zero_probability + cancellation_zero_probability:
                    block = cross_scale * rng.normal(size=(len(block_i), len(block_j)))
                    denominator = float(np.sum(reference * reference))
                    if denominator <= 1e-14:
                        block = np.zeros_like(block)
                        kinds[i][j] = "structural_zero"
                    else:
                        block -= float(np.trace(block @ reference)) / denominator * reference.T
                        if np.linalg.norm(block) <= 1e-10:
                            block = np.zeros_like(block)
                            kinds[i][j] = "structural_zero"
                        else:
                            kinds[i][j] = "cancellation_zero"
                else:
                    block = cross_scale * rng.normal(size=(len(block_i), len(block_j)))
                    kinds[i][j] = "active"
                A[np.ix_(block_i, block_j)] = block

        maximum_real_part = float(np.max(np.real(np.linalg.eigvals(A))))
        if maximum_real_part >= -stability_margin:
            A -= (maximum_real_part + stability_margin) * np.eye(dimension)

        Sigma = -(A @ S + S @ A.T)
        Sigma = 0.5 * (Sigma + Sigma.T)
        if float(np.min(np.linalg.eigvalsh(Sigma))) < sigma_min_eigenvalue:
            continue

        flows = subspace_information_flow(A, S, segments)
        has_exact_zero = False
        has_active = False
        valid = True
        for i in range(len(blocks)):
            for j in range(len(blocks)):
                if i == j:
                    continue
                if kinds[i][j] in {"structural_zero", "cancellation_zero"}:
                    has_exact_zero = True
                    valid &= bool(np.isclose(flows[i, j], 0.0, atol=1e-12, rtol=0.0))
                else:
                    has_active = True
                    valid &= not bool(np.isclose(flows[i, j], 0.0, atol=1e-8, rtol=0.0))
                    valid &= abs(flows[i, j]) >= min_active_flow_magnitude
                if not valid:
                    break
            if not valid:
                break

        if valid and min_active_flow_magnitude > 0.0:
            active_values = [
                abs(flows[i, j])
                for i in range(len(blocks))
                for j in range(len(blocks))
                if i != j and kinds[i][j] == "active"
            ]
            if active_values and min(active_values) < min_active_flow_magnitude:
                valid = False
        if valid and has_exact_zero and has_active:
            return DynamicalSystem(A, S, Sigma, flows, kinds, segments)
    raise RuntimeError("Failed to construct a valid dynamical system within max_tries")


def van_loan_discretization(
    A: np.ndarray,
    Sigma_c: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = A.shape[0]
    block = np.zeros((2 * dimension, 2 * dimension))
    block[:dimension, :dimension] = A
    block[:dimension, dimension:] = Sigma_c
    block[dimension:, dimension:] = -A.T
    exponential = expm(block * dt)
    F = exponential[:dimension, :dimension]
    Q = exponential[:dimension, dimension:] @ F.T
    Q = 0.5 * (Q + Q.T)
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    Q = eigenvectors @ np.diag(np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    return F, Q


def required_burnin_steps(
    A: np.ndarray,
    sim_dt: float,
    n_half_lives: float = 10.0,
) -> int:
    slowest_rate = float(np.min(np.abs(np.real(np.linalg.eigvals(A)))))
    if slowest_rate < 1e-12:
        raise ValueError("A has a near-zero real eigenvalue")
    return max(int(np.ceil(n_half_lives / slowest_rate / sim_dt)), 500)


def simulate(
    A: np.ndarray,
    Sigma_c: np.ndarray,
    *,
    sim_dt: float,
    subsample_factor: int,
    n_obs: int,
    seed: int,
    n_half_lives: float = 10.0,
) -> np.ndarray:
    """Exact Van Loan simulation with the archived burn-in/indexing rule."""
    rng = np.random.default_rng(seed)
    dimension = A.shape[0]
    F, Q = van_loan_discretization(A, Sigma_c, sim_dt)
    burnin_steps = required_burnin_steps(A, sim_dt, n_half_lives)
    total_steps = burnin_steps + n_obs * subsample_factor

    try:
        S0 = solve_continuous_lyapunov(A, -Sigma_c)
        S0 = 0.5 * (S0 + S0.T)
        state = (
            gaussian_samples(rng, np.zeros(dimension), S0)
            if np.min(np.linalg.eigvalsh(S0)) > 1e-10
            else np.zeros(dimension)
        )
    except Exception:
        state = np.zeros(dimension)

    noise = gaussian_samples(rng, np.zeros(dimension), Q, size=total_steps)
    observations = []
    for step in range(total_steps):
        state = F @ state + noise[step]
        if step >= burnin_steps and (step - burnin_steps) % subsample_factor == 0:
            observations.append(state.copy())
            if len(observations) == n_obs:
                break
    return np.asarray(observations)


def generate_panel_data(
    N: int,
    A: np.ndarray,
    B: np.ndarray,
    f: np.ndarray,
    Sigma: np.ndarray,
    dt: float,
    seed: int,
) -> list[np.ndarray]:
    """Generate N independent two-time-point panels."""
    rng = np.random.default_rng(seed)
    X0 = gaussian_samples(rng, np.zeros(A.shape[0]), Sigma, size=N)
    epsilon = rng.standard_normal((N, A.shape[0]))
    X1 = X0 + dt * (X0 @ A.T + f) + np.sqrt(dt) * epsilon @ B.T
    return [np.vstack((X0[index], X1[index])) for index in range(N)]


def balanced_segments(dimension: int, n_subspaces: int) -> list[list[int]]:
    if not 0 < n_subspaces <= dimension:
        raise ValueError("invalid subspace count")
    base_size, remainder = divmod(dimension, n_subspaces)
    segments = []
    start = 0
    for index in range(n_subspaces):
        size = base_size + (1 if index < remainder else 0)
        segments.append(list(range(start, start + size)))
        start += size
    return segments


__all__ = [
    "DynamicalSystem",
    "balanced_segments",
    "build_dynamical_system",
    "generate_panel_data",
    "gaussian_samples",
    "required_burnin_steps",
    "simulate",
    "subspace_information_flow",
    "van_loan_discretization",
]

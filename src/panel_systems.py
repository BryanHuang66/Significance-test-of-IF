"""Six-dimensional independent-panel generator from statistical_validation.py."""

from __future__ import annotations

import numpy as np

from .plugins import real_information_flow_linear_case


def generate_random_stable_system(
    dimension: int,
    dt: float,
    noise_strength: float,
    seed: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    off_diagonal = ~np.eye(dimension, dtype=bool)

    def covariance_block(size: int) -> np.ndarray:
        raw = rng.standard_normal((size, size))
        covariance = raw @ raw.T + size * np.eye(size)
        scale = np.diag(1.0 / np.sqrt(np.diag(covariance)))
        covariance = scale @ covariance @ scale
        standard_deviations = rng.uniform(0.8, 1.8, size)
        return (
            np.diag(standard_deviations)
            @ covariance
            @ np.diag(standard_deviations)
        )

    for _ in range(30_000):
        if mode == "structural":
            Sigma = covariance_block(dimension)
        elif mode == "covariance":
            left = dimension // 2
            Sigma = np.zeros((dimension, dimension))
            Sigma[:left, :left] = covariance_block(left)
            Sigma[left:, left:] = covariance_block(dimension - left)
            permutation = rng.permutation(dimension)
            Sigma = Sigma[np.ix_(permutation, permutation)]
        else:
            raise ValueError("mode must be 'structural' or 'covariance'")

        A = np.zeros((dimension, dimension))
        np.fill_diagonal(A, -rng.uniform(0.65, 1.05, dimension))
        active = rng.random((dimension, dimension)) < 0.40
        np.fill_diagonal(active, False)
        entries = rng.choice([-1.0, 1.0], size=(dimension, dimension)) * rng.uniform(
            0.20, 0.45, size=(dimension, dimension)
        )
        A[active] = entries[active]

        if mode == "covariance":
            candidates = np.argwhere(
                np.isclose(Sigma, 0.0, atol=1e-14) & off_diagonal
            )
            if len(candidates) < 8:
                continue
            for chosen in rng.choice(len(candidates), size=8, replace=False):
                i, j = candidates[chosen]
                A[i, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 0.45)

        for i in range(dimension):
            row_off_diagonal = np.sum(np.abs(A[i])) - abs(A[i, i])
            A[i, i] = -max(abs(A[i, i]), row_off_diagonal + 0.25)

        transition = np.eye(dimension) + dt * A
        eigenvalues = np.abs(np.linalg.eigvals(transition))
        if eigenvalues.max() >= 0.97 or eigenvalues.min() <= 0.20:
            continue
        off_values = A[off_diagonal]
        if np.isclose(off_values, 0.0).sum() < 10:
            continue
        if (~np.isclose(off_values, 0.0)).sum() < 10:
            continue

        innovation = Sigma - transition @ Sigma @ transition.T
        innovation = 0.5 * (innovation + innovation.T)
        if np.min(np.linalg.eigvalsh(innovation)) <= 1e-9:
            continue
        B = np.linalg.cholesky(innovation) * noise_strength
        if mode == "covariance" and (
            np.max(np.abs(B)) > 0.95 or np.linalg.cond(Sigma) > 12
        ):
            continue
        f = rng.normal(0.0, 0.15, dimension)
        return A, B, f, Sigma
    raise RuntimeError("Failed to generate a valid panel system after 30000 attempts")


def accepted_panel_systems(
    count: int,
    *,
    dimension: int = 6,
    dt: float = 1.0,
    covariance_ratio: float = 0.2,
) -> list[dict]:
    off_diagonal = ~np.eye(dimension, dtype=bool)
    n_covariance = (
        max(1, min(count - 1, round(count * covariance_ratio))) if count > 1 else 0
    )
    modes = ["structural"] * (count - n_covariance) + ["covariance"] * n_covariance
    systems = []
    seed_cursor = 1000
    for mode in modes:
        while True:
            A, B, f, Sigma = generate_random_stable_system(
                dimension, dt, 1.0, seed_cursor, mode
            )
            seed_used = seed_cursor
            seed_cursor += 1
            true_if = np.asarray(
                real_information_flow_linear_case(
                    A,
                    B,
                    deg_freedom=200_000 - dimension,
                    segments=[(i, i + 1) for i in range(dimension)],
                    discrete_lyapunov=True,
                    dt=dt,
                )["information_flow"]
            )
            null = np.isclose(true_if, 0.0, atol=1e-10) & off_diagonal
            active = ~np.isclose(true_if, 0.0, atol=1e-10) & off_diagonal
            structural = np.isclose(A, 0.0, atol=1e-12) & null
            covariance = (
                ~np.isclose(A, 0.0, atol=1e-12)
                & np.isclose(Sigma, 0.0, atol=1e-12)
                & null
            )
            accepted = (
                structural.sum() >= 8 and null.sum() >= 10
                if mode == "structural"
                else covariance.sum() >= 4 and null.sum() >= 8
            )
            if accepted and active.sum() >= 6:
                systems.append(
                    {
                        "A": A,
                        "B": B,
                        "f": f,
                        "Sigma": Sigma,
                        "true_if": true_if,
                        "null_pairs": null,
                        "alt_pairs": active,
                        "struct_null": structural,
                        "cov_null": covariance,
                        "mode": mode,
                        "seed": seed_used,
                    }
                )
                break
    return systems


__all__ = ["accepted_panel_systems", "generate_random_stable_system"]

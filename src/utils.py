"""Maintained-LKIF adapter, provenance, and result serialization utilities."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .plugins import _compute_origin_standard_error, to_index_segments


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PAPER_SOURCE_ROOT = Path(
    "/Users/huangkewei/Tex 存档/Significance Test of IF/Significance Test of IF"
)
LKIF_ROOT = Path(
    os.environ.get(
        "LK_INFO_FLOW_ROOT",
        "/Users/huangkewei/Library/CloudStorage/Dropbox/PHD Code/LK_info_flow",
    )
).expanduser()

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_or_none(path: Path) -> str | None:
    """Hash a provenance file, tolerating its absence on remote runs."""
    try:
        return _sha256(path)
    except OSError:
        return None


def _git_metadata(path: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {"head": head.stdout.strip() or None, "dirty": bool(status.stdout.strip())}


def metadata() -> dict[str, Any]:
    paper_scripts = (
        "exp/exp2.py",
        "exp/exp3.py",
        "exp/dynamics_builder.py",
        "exp/simulator.py",
        "exp/plot_exp2.py",
        "exp/plot_exp3.py",
        "experiments/statistical_validation.py",
        "experiments/variance_estimation_1d.py",
        "experiments/variance_estimation_subspace.py",
        "experiments/experiment_subspace_varying_dim.py",
        "experiments/lorenz_z_to_x_test.py",
    )
    return {
        "project_root": str(PROJECT_ROOT),
        "paper_source_root": str(PAPER_SOURCE_ROOT),
        "external_lkif_root": str(LKIF_ROOT),
        "external_lkif_git": _git_metadata(LKIF_ROOT),
        "external_lkif_sources": {
            name: _sha256_or_none(LKIF_ROOT / "lkif" / name)
            for name in ("linear_causality.py", "utils.py")
        },
        "paper_source_hashes": {
            relative: _sha256_or_none(PAPER_SOURCE_ROOT / relative)
            for relative in paper_scripts
        },
    }


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def save_json(path: str | Path, payload: Any) -> Path:
    """Write a freshly computed result without mutating reference summaries."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return output


def load_linear_lkif():
    """Load the estimator from the maintained external checkout."""
    if not (LKIF_ROOT / "lkif" / "linear_causality.py").is_file():
        raise RuntimeError(f"Missing external LKIF checkout: {LKIF_ROOT}")
    root_text = str(LKIF_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    loaded = sys.modules.get("lkif")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", "") or ""
        if not str(Path(loaded_file).resolve()).startswith(str(LKIF_ROOT.resolve())):
            for name in list(sys.modules):
                if name == "lkif" or name.startswith("lkif."):
                    del sys.modules[name]
    module = importlib.import_module("lkif")
    return module.LinearLKInformationFlow


def estimate_information_flow(
    data: list[np.ndarray] | np.ndarray,
    segments: Sequence[Sequence[int] | tuple[int, int]],
    *,
    dt: float,
    data_mode: str,
    hac_max_lag: int | None = None,
    ridge_lambda: float | str = 1e-6,
) -> dict[str, Any]:
    """Run the maintained two-stage LKIF API with explicit experiment settings."""
    if data_mode not in {"panel", "time_series"}:
        raise ValueError(f"Unsupported data mode: {data_mode}")
    index_segments = to_index_segments(segments)
    model = load_linear_lkif()(dt=dt)
    model.data_init(
        data,
        lag_list=[1],
        segments=index_segments,
        significance_test=True,
        ridge_lambda=ridge_lambda,
    )
    model.causality_estimate(data_mode=data_mode, hac_max_lag=hac_max_lag)
    result = dict(model.get_dict())
    result["information_flow_std_origin"] = _compute_origin_standard_error(
        model, index_segments, ridge_lambda
    )
    hac_variance = getattr(model, "hac_cov_variance", None)
    if hac_variance is not None:
        result["hac_cov_variance"] = np.asarray(hac_variance)
    return result


__all__ = [
    "LKIF_ROOT",
    "OUTPUT_ROOT",
    "PROJECT_ROOT",
    "estimate_information_flow",
    "metadata",
    "save_json",
]

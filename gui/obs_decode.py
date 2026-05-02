"""
Decode sumo-rl DefaultObservationFunction vectors into named parts:
[phase_one_hot | min_green | densities... | queues...]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DecodedObservation:
    phase_one_hot: list[float]
    min_green_ok: float
    densities: list[float]
    queues: list[float]
    n_green_phases: int
    n_lanes: int


def infer_n_lanes(obs_dim: int, n_green_phases: int) -> int:
    """obs_dim = n_green_phases + 1 + 2 * n_lanes."""
    rest = obs_dim - n_green_phases - 1
    if rest < 0 or rest % 2 != 0:
        raise ValueError(
            f"Cannot infer lanes: obs_dim={obs_dim}, n_green_phases={n_green_phases}"
        )
    return rest // 2


def decode_default_observation(
    obs: np.ndarray,
    *,
    n_green_phases: int,
) -> DecodedObservation:
    """Split a single sumo-rl default observation vector."""
    flat = np.asarray(obs, dtype=np.float64).ravel()
    obs_dim = int(flat.shape[0])
    n_lanes = infer_n_lanes(obs_dim, n_green_phases)
    i = 0
    phase_one_hot = flat[i : i + n_green_phases].tolist()
    i += n_green_phases
    min_green_ok = float(flat[i])
    i += 1
    densities = flat[i : i + n_lanes].tolist()
    i += n_lanes
    queues = flat[i : i + n_lanes].tolist()
    return DecodedObservation(
        phase_one_hot=phase_one_hot,
        min_green_ok=min_green_ok,
        densities=densities,
        queues=queues,
        n_green_phases=n_green_phases,
        n_lanes=n_lanes,
    )


def lane_labels(n_lanes: int, prefix: str = "in") -> list[str]:
    return [f"{prefix}_{i}" for i in range(n_lanes)]

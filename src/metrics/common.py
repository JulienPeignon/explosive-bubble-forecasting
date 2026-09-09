"""Primitives shared across the metric families in src/metrics/."""

import numpy as np


def normalize(density, grid_y):
    """Renormalize each row to integrate to 1 over the finite grid."""
    mass = np.trapezoid(density, grid_y, axis=1)[:, None]
    return density / np.clip(mass, 1e-12, None)

"""Utils functions for flintpy."""

from typing import Tuple

import numpy as np


def logarithm_t_range(t_range: Tuple[float, float], kernel_dim: int) -> np.ndarray:
    """Generates a logarithmic time range."""
    return np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), num=kernel_dim)

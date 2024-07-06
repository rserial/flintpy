"""Utils functions for pyflint."""

from pathlib import Path
from typing import Tuple

import numpy as np


def logarithm_t_range(t_range: Tuple[float, float], kernel_dim: int) -> np.ndarray:
    """Generates a logarithmic time range."""
    return np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), num=kernel_dim)


# loading of data
def load_1d_decay(file_path: Path, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load 1D decay data from a file.

    Args:
        file_path (Path): The path to the file.
        file_name (str): The name of the file.

    Returns:
        tuple[np.ndarray, np.ndarray]: Time axis and decay data.
    """
    data = np.loadtxt(file_path / file_name, delimiter=",")
    time_axis = data[:, 0]
    decay_data = data[:, 1]
    return time_axis, decay_data

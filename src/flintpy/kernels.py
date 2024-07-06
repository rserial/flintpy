"""Kernel functions for Flint class."""

from typing import Any, Callable

import numpy as np


def kernel_t2(tau: np.ndarray, t2: np.ndarray) -> Any:  # noqa: ANN401
    """T2 exponential decay."""
    return np.exp(-np.outer(tau, 1 / t2))


def kernel_t1_ir(tau: np.ndarray, t1: np.ndarray) -> Any:  # noqa: ANN401
    """T1 exponential decay for Inversion Recovery experiments."""
    return 1 - 2 * np.exp(-np.outer(tau, 1 / t1))


def kernel_t1_sr(tau: np.ndarray, t1: np.ndarray) -> Any:  # noqa: ANN401
    """T1 exponential decay for Saturation Recovery experiments."""
    return 1 - 1 * np.exp(-np.outer(tau, 1 / t1))


def set_kernel(
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tau: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Sets a kernel for given tau and T arrays."""
    k = kernel(tau, t)
    return k

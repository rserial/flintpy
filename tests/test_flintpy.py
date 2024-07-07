"""Tests for `flintpy` module."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from pytest import LogCaptureFixture

from flintpy.flintpy import Flint, FlintSignal


@pytest.mark.parametrize(
    "data,t1_axis,t2_axis,kernel_shape,kernel_name,alpha,t1_range,t2_range,expected",
    [
        (
            np.ones((1, 1)),
            np.ones(1),
            None,
            [1, 1],
            "T1IR",
            0.1,
            (1.0, 1.0),
            None,
            "Lipschitz constant found: 0.33",
        ),
        (
            np.loadtxt(Path("./data/examples") / "T1IRT2.dat", delimiter=" "),
            np.loadtxt(Path("./data/examples") / "T1IRT2_t1axis.dat"),
            np.loadtxt(Path("./data/examples") / "T1IRT2_t2axis.dat"),
            [32, 32],
            "T1IRT2",
            1e-1,
            [1e-4, 10e0],
            [1e-4, 10e0],
            "Lipschitz constant found: 78646960.22",
        ),
        (
            np.loadtxt(Path("./data/examples") / "T2decay.dat")[:, 1],
            np.loadtxt(Path("./data/examples") / "T2decay.dat")[:, 0],
            None,
            [100, 1],
            "T2",
            1e-1,
            [1e-4, 10e0],
            None,
            "Lipschitz constant found: 4783.93",
        ),
    ],
)
def test_solve_flint(
    caplog: LogCaptureFixture,
    data: np.ndarray,
    t1_axis: np.ndarray,
    t2_axis: Optional[np.ndarray],
    kernel_shape: tuple[int, int],
    kernel_name: str,
    alpha: float,
    t1_range: tuple[float, float],
    t2_range: Optional[tuple[float, float]],
    expected: str,
) -> None:
    """Test solve flint."""
    with caplog.at_level(logging.INFO):
        signal = FlintSignal.load_from_data(data, t1_axis, t2_axis)
        flint = Flint(
            signal,
            kernel_shape=kernel_shape,
            kernel_name=kernel_name,
            alpha=alpha,
            t1range=t1_range,
            t2range=t2_range,
        )
        flint.solve_flint()
    assert expected in caplog.text

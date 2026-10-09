"""Small helpers shared by several modules."""
from __future__ import annotations

import pickle
from typing import Any

import numpy as np

try:  # numpy >= 2
    trapezoid = np.trapezoid
except AttributeError:  # pragma: no cover
    trapezoid = np.trapz


def gaussian(x, amp, mu, sigma):
    """Gaussian with peak ``amp``, centre ``mu`` and dispersion ``sigma``."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def jy_to_flambda(f_jy, wave_angstrom):
    """Convert Jy -> erg s^-1 cm^-2 A^-1."""
    return f_jy * 2.99792458e-5 / wave_angstrom ** 2


def progress(iterable, desc: str = "", disable: bool = False):
    """tqdm progress bar if available, otherwise the plain iterable."""
    if disable:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, leave=False)
    except ImportError:
        return iterable


def save(obj: Any, path: str) -> None:
    """Pickle any result (dicts of LineMaps, fits, decompositions...) so you can
    restart a session without re-running slow steps."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"saved -> {path}")


def load(path: str) -> Any:
    """Load an object written with :func:`save`."""
    with open(path, "rb") as f:
        return pickle.load(f)

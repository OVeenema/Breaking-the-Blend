"""
STEP 4 - Stack the per-line disc-fraction maps into one map (paper Sect. 3.4, Fig. 2).

Each line's f_disc map is interpolated onto a common grid of offsets from the
AGN in parsec (so different channels / pixel scales line up), then averaged
with a weight per line. The paper's weight peaks for lines that trace both
components equally (mean f_disc = 0.5) and falls linearly to zero at 0 and 1:

    w = 1 - 2 |<f_disc> - 0.5|

Only grid points covered by at least ``min_lines`` lines are kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .decompose import Decomposition


@dataclass
class StackedMap:
    """Final multi-line map on a common parsec grid centred on the AGN."""

    x_pc: np.ndarray                 # (nx,) grid x offsets [pc]
    y_pc: np.ndarray                 # (ny,) grid y offsets [pc]
    f_disc: np.ndarray               # (ny, nx) stacked disc fraction
    n_lines: np.ndarray              # (ny, nx) number of lines contributing
    weights: Dict[str, float]
    mean_f_disc: Dict[str, float]
    scheme: str
    per_line: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)  # each line on the grid

    @property
    def f_out(self) -> np.ndarray:
        """AGN outflow fraction (paper Fig. 2)."""
        return 1.0 - self.f_disc

    @property
    def extent(self):
        dx = np.diff(self.x_pc[:2])[0] / 2
        dy = np.diff(self.y_pc[:2])[0] / 2
        return [self.x_pc[0] - dx, self.x_pc[-1] + dx, self.y_pc[0] - dy, self.y_pc[-1] + dy]

    def sample(self, x_pc, y_pc, which: str = "f_disc") -> np.ndarray:
        """Interpolate the stacked map at arbitrary pc offsets (NaN outside)."""
        arr = getattr(self, which)
        f = RegularGridInterpolator((self.y_pc, self.x_pc), arr, bounds_error=False, fill_value=np.nan)
        pts = np.column_stack([np.ravel(y_pc), np.ravel(x_pc)])
        return f(pts).reshape(np.shape(x_pc))


def line_weight(mean_f_disc: float, scheme: str = "linear", width: float = 0.2) -> float:
    """Weight of a line from its mean disc fraction.

    ``"linear"`` (paper), ``"gaussian"`` (centred on 0.5, dispersion ``width``) or ``"equal"``.
    """
    if not np.isfinite(mean_f_disc):
        return 0.0
    if scheme == "linear":
        return float(np.clip(1.0 - 2.0 * abs(mean_f_disc - 0.5), 0, None))
    if scheme == "gaussian":
        return float(np.exp(-0.5 * ((mean_f_disc - 0.5) / width) ** 2))
    if scheme == "equal":
        return 1.0
    raise ValueError("scheme must be 'linear', 'gaussian' or 'equal'")


def to_grid(d: Decomposition, x_pc: np.ndarray, y_pc: np.ndarray, which: str = "f_disc") -> np.ndarray:
    """Interpolate one decomposition map onto the common (y_pc, x_pc) grid."""
    arr = getattr(d, which)
    ny, nx = arr.shape
    s = d.pixel_scale_pc
    xs = (np.arange(nx) - d.agn_xy[0]) * s
    ys = (np.arange(ny) - d.agn_xy[1]) * s
    f = RegularGridInterpolator((ys, xs), arr, bounds_error=False, fill_value=np.nan)
    X, Y = np.meshgrid(x_pc, y_pc)
    return f(np.column_stack([Y.ravel(), X.ravel()])).reshape(X.shape)


def stack_fraction_maps(decomps: Dict[str, Decomposition], names: Optional[Sequence[str]] = None,
                        scheme: str = "linear", combine: str = "mean", min_lines: int = 2,
                        weights: Optional[Dict[str, float]] = None, gaussian_width: float = 0.2,
                        grid_step_pc: Optional[float] = None, grid_halfsize_pc: Optional[float] = None,
                        verbose: bool = True) -> StackedMap:
    """Combine per-line disc fractions into one map.

    Parameters
    ----------
    decomps : ``{line: Decomposition}`` from Step 3.
    names : lines to include (default: all) - use this to test subsets.
    scheme : weighting scheme (see :func:`line_weight`).
    combine : ``"mean"`` (weighted mean, paper) or ``"median"`` (unweighted median).
    min_lines : minimum number of lines per grid point.
    weights : ``{line: w}`` to set the weights by hand (overrides ``scheme``).
    grid_step_pc : grid spacing; default = half the finest pixel scale.
    grid_halfsize_pc : grid half-size; default = largest extent of any map.
    """
    names = list(names or decomps)
    ds = [decomps[n] for n in names]
    if grid_step_pc is None:
        grid_step_pc = 0.5 * min(d.pixel_scale_pc for d in ds)
    if grid_halfsize_pc is None:
        grid_halfsize_pc = max(np.max(np.abs(d.extent_pc())) for d in ds)
    n = int(np.ceil(grid_halfsize_pc / grid_step_pc))
    grid = np.arange(-n, n + 1) * grid_step_pc

    w_used, means, per_line = {}, {}, {}
    for d in ds:
        means[d.line_name] = d.mean_f_disc
        w_used[d.line_name] = (weights or {}).get(d.line_name, line_weight(d.mean_f_disc, scheme, gaussian_width))
        per_line[d.line_name] = to_grid(d, grid, grid)

    cube = np.array([per_line[n_] for n_ in names])           # (nlines, ny, nx)
    finite = np.isfinite(cube)
    count = finite.sum(axis=0)
    if combine == "mean":
        w = np.array([w_used[n_] for n_ in names])[:, None, None] * finite
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.nansum(np.nan_to_num(cube) * w, axis=0) / w.sum(axis=0)
    elif combine == "median":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            f = np.nanmedian(cube, axis=0)
    else:
        raise ValueError("combine must be 'mean' or 'median'")
    f = np.where(count >= min_lines, f, np.nan)

    if verbose:
        print(f"{'line':>9s} {'<f_disc>':>9s} {'weight':>7s}")
        for d in ds:
            print(f"{d.label:>9s} {means[d.line_name]:9.2f} {w_used[d.line_name]:7.2f}")
        print(f"grid: {len(grid)}x{len(grid)} at {grid_step_pc:.1f} pc; "
              f"{np.isfinite(f).sum()} points with >= {min_lines} lines")
    return StackedMap(x_pc=grid, y_pc=grid, f_disc=f, n_lines=count, weights=w_used,
                      mean_f_disc=means, scheme=f"{combine}/{scheme}", per_line=per_line)


def line_to_line_scatter(stack: StackedMap) -> float:
    """Median over the grid of the standard deviation of f_disc between lines
    (where >= 2 lines overlap). The paper quotes 0.17 for NGC 5728."""
    import warnings
    cube = np.array(list(stack.per_line.values()))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(cube, axis=0)
    sd[np.isfinite(cube).sum(axis=0) < 2] = np.nan
    return float(np.nanmedian(sd))


def compare_stacks(stacks: Dict[str, StackedMap], reference: str) -> None:
    """Print the median |difference| of several stacked maps from a reference one
    (robustness test of paper Sect. 3.4)."""
    r = stacks[reference]
    X, Y = np.meshgrid(r.x_pc, r.y_pc)
    for k, s in stacks.items():
        if k == reference:
            continue
        diff = np.abs(s.sample(X, Y) - r.f_disc)
        print(f"{k:>25s}: median |delta f_disc| = {np.nanmedian(diff):.3f}")

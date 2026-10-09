"""
STEP 1b - Velocity dispersion of each component (paper Sect. 3.2, Fig. S3).

From the unconstrained double-Gaussian fits we enforce sigma_outflow > sigma_disc
in every spaxel (narrow = disc, broad = outflow) and take the median of each
distribution over all good spaxels. These medians are the fixed dispersions in
the constrained decomposition (Step 3).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .linefit import LineMaps


def dispersion_samples(lm: LineMaps, sigma_range: Tuple[float, float] = (15.0, 600.0),
                       min_ratio: Optional[float] = None):
    """Arrays of (disc, outflow) dispersions [km/s] over the good spaxels.

    Values outside ``sigma_range`` (fits stuck at a bound, noise) are dropped.
    min_ratio : if given, only spaxels where sigma_broad >= min_ratio * sigma_narrow are used.
        Where a spaxel really holds a single component, a double-Gaussian fit is degenerate: both
        Gaussians describe the same profile and get (nearly) the same sigma. Such spaxels put
        "broad" values at the narrow dispersion; ``min_ratio`` (e.g. 1.3) excludes them.
    """
    d = lm.double
    sn, sb = d["sigma_n"][lm.good], d["sigma_b"][lm.good]
    ok_n = np.isfinite(sn) & (sn >= sigma_range[0]) & (sn <= sigma_range[1])
    ok_b = np.isfinite(sb) & (sb >= sigma_range[0]) & (sb <= sigma_range[1])
    if min_ratio is not None:
        two = np.isfinite(sn) & np.isfinite(sb) & (sb >= min_ratio * sn)
        ok_n &= two
        ok_b &= two
    return sn[ok_n], sb[ok_b]


def dispersion_medians(line_maps: Dict[str, LineMaps], sigma_range: Tuple[float, float] = (15.0, 600.0),
                       overrides: Optional[Dict[str, Tuple[float, float]]] = None,
                       min_ratio: Optional[float] = None,
                       verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """Median (and 16/84th percentile) disc and outflow dispersion for every line.

    Parameters
    ----------
    min_ratio : exclude single-component (degenerate) spaxels, see :func:`dispersion_samples`.
    overrides : ``{line: (sigma_disc, sigma_outflow)}`` to replace the measured values,
        e.g. for lines with a single component (``(np.nan, 255)`` for [Ne V]).

    Returns
    -------
    ``{line: {"disc", "outflow", "disc_lo", "disc_hi", "outflow_lo", "outflow_hi", "n"}}``
    """
    overrides = overrides or {}
    out = {}
    if verbose:
        print(f"{'line':>8s} {'IP':>6s} {'sigma_disc':>18s} {'sigma_outflow':>18s} {'N':>5s}")
    for name, lm in line_maps.items():
        sd, so = dispersion_samples(lm, sigma_range, min_ratio)
        res = {"n": int(min(sd.size, so.size))}
        for key, arr in (("disc", sd), ("outflow", so)):
            if arr.size:
                p16, p50, p84 = np.percentile(arr, [16, 50, 84])
                res.update({key: p50, f"{key}_lo": p50 - p16, f"{key}_hi": p84 - p50})
            else:
                res.update({key: np.nan, f"{key}_lo": np.nan, f"{key}_hi": np.nan})
        flag = ""
        if name in overrides:
            res["disc"], res["outflow"] = overrides[name]
            flag = "  (override)"
        out[name] = res
        if verbose:
            fmt = lambda k: f"{res[k]:6.0f} +{res[k+'_hi']:4.0f}/-{res[k+'_lo']:4.0f}"
            print(f"{lm.label:>8s} {lm.ip:6.1f} {fmt('disc'):>18s} {fmt('outflow'):>18s} {res['n']:5d}{flag}")
    return out

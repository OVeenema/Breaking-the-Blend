"""
STEP 1 - Measure every emission line in every spaxel.

For one line, :func:`fit_line`

1. cuts the fit window out of the cube,
2. fits and subtracts a polynomial continuum (line region masked),
3. integrates the line flux [erg s^-1 cm^-2 per spaxel],
4. masks faint spaxels (below ``line.mask_percentile`` of the flux),
5. fits a single and a double Gaussian (in velocity space) to each good spaxel,
6. computes flux-weighted moments (velocity, dispersion) as a model-free check.

The double-Gaussian components are always ordered so that the *narrow*
component has the smaller dispersion. Under the paper's assumption
sigma_outflow > sigma_disc the narrow component is the disc candidate and the
broad component the outflow candidate (see :mod:`dispersion`).

Results are stored in a :class:`LineMaps` object. Run all lines with
:func:`fit_lines`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.optimize import least_squares

from .lines import C_KMS, LineSpec
from .cubes import Cube
from .utils import progress, trapezoid, gaussian, jy_to_flambda


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------
@dataclass
class LineMaps:
    """All maps measured for one line.

    Map attributes are 2D arrays (ny, nx) on the pixel grid of the cube the
    line was measured in. Velocities/dispersions are in km/s relative to the
    rest wavelength (after de-redshifting with the galaxy redshift).

    single : dict with ``amp``, ``v``, ``sigma`` of the single-Gaussian fit.
    double : dict with ``amp_n, v_n, sigma_n`` (narrow) and ``amp_b, v_b, sigma_b`` (broad).
    """

    line: LineSpec
    cube_name: str
    pixel_scale_pc: float
    agn_xy: Tuple[float, float]
    wave: np.ndarray            # rest wavelength of the cut-out [micron]
    vel_axis: np.ndarray        # velocity of each channel [km/s]
    in_line: np.ndarray         # bool, channels inside line_region
    contsub: np.ndarray         # (n_chan, ny, nx) continuum-subtracted flux density [Jy]
    flux: np.ndarray            # integrated line flux [erg/s/cm^2]
    good: np.ndarray            # bool, spaxels kept for the analysis
    single: Dict[str, np.ndarray] = field(default_factory=dict)
    double: Dict[str, np.ndarray] = field(default_factory=dict)
    mom1: Optional[np.ndarray] = None
    mom2: Optional[np.ndarray] = None
    fitted_spaxels: Optional[list] = None      # set when only some spaxels were fitted (check_line)

    # convenience -------------------------------------------------------
    @property
    def name(self) -> str:
        return self.line.name

    @property
    def label(self) -> str:
        return self.line.label

    @property
    def ip(self) -> float:
        return self.line.ip

    @property
    def shape(self):
        return self.flux.shape

    def offsets_pc(self):
        """(x_pc, y_pc) offsets of every spaxel from the AGN."""
        ny, nx = self.shape
        yy, xx = np.indices((ny, nx), dtype=float)
        s = self.pixel_scale_pc
        return (xx - self.agn_xy[0]) * s, (yy - self.agn_xy[1]) * s

    def extent_pc(self):
        """``imshow`` extent [pc] with the AGN at (0, 0)."""
        ny, nx = self.shape
        s = self.pixel_scale_pc
        x0, y0 = self.agn_xy
        return [(-0.5 - x0) * s, (nx - 0.5 - x0) * s, (-0.5 - y0) * s, (ny - 0.5 - y0) * s]

    def velocity_map(self, source: str = "single", masked: bool = True) -> np.ndarray:
        """Observed velocity field used for kinematic modelling.

        source : ``"single"`` (single-Gaussian centroid, as in the paper) or ``"moment"``.
        """
        v = {"single": self.single.get("v"), "moment": self.mom1}[source]
        if v is None:
            raise ValueError(f"No '{source}' velocities for {self.name}")
        v = np.array(v, dtype=float)
        if masked:
            v[~self.good] = np.nan
        return v

    def masked(self, arr: np.ndarray) -> np.ndarray:
        out = np.array(arr, dtype=float)
        out[~self.good] = np.nan
        return out

    def __repr__(self) -> str:
        return (f"LineMaps({self.label}, IP={self.ip} eV, cube={self.cube_name}, "
                f"{self.good.sum()} good spaxels)")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def subtract_continuum(wave: np.ndarray, cut: np.ndarray, line_region: Tuple[float, float],
                       order: int = 3) -> np.ndarray:
    """Fit a polynomial to the line-free channels of every spaxel at once and subtract it.

    wave : (n,) ; cut : (n, ny, nx). Returns the continuum-subtracted cut-out.
    """
    n, ny, nx = cut.shape
    lfree = (wave < line_region[0]) | (wave > line_region[1])
    if lfree.sum() <= order + 1:
        raise ValueError("Too few line-free channels for the continuum fit: widen fit_window "
                         "or reduce cont_order.")
    flat = cut.reshape(n, -1).astype(float)
    xw = wave - wave.mean()                       # centre for numerical stability
    coeffs = np.polyfit(xw[lfree], flat[lfree], order)   # (order+1, nspax)
    cont = np.vander(xw, order + 1) @ coeffs
    return (flat - cont).reshape(n, ny, nx)


def make_mask(flux: np.ndarray, fov: np.ndarray, percentile: float, dilate: int = 2,
              percentile_over: str = "fov") -> np.ndarray:
    """Good-spaxel mask: flux above a percentile, with the rejected region grown by
    ``dilate`` pixels (removes ragged edges, as in the original notebooks)."""
    vals = flux[fov] if percentile_over == "fov" else flux
    thresh = np.nanpercentile(vals, percentile)
    bad = ~(flux >= thresh) | ~fov
    if dilate > 0:
        bad = binary_dilation(bad, structure=generate_binary_structure(2, 1), iterations=dilate)
    return ~bad


def _moments(v, y):
    yp = np.clip(y, 0, None)
    s = yp.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = (yp * v[:, None]).sum(axis=0) / s
        m2 = np.sqrt((yp * (v[:, None] - m1) ** 2).sum(axis=0) / s)
    return m1, m2


def _single_model(p, v):
    return gaussian(v, *p)


def _double_model(p, v):
    return gaussian(v, *p[:3]) + gaussian(v, *p[3:])


_CAPFIT = None


def _get_capfit():
    global _CAPFIT
    if _CAPFIT is None:
        try:
            from capfit.capfit import capfit
            _CAPFIT = capfit
        except ImportError:
            _CAPFIT = False
    return _CAPFIT


def _fit(model, p0, lo, hi, v, y, w, fitter="trf"):
    """Least-squares fit of ``model(p, v)`` to ``y`` with weights ``w``.

    fitter : ``"trf"`` bounded trust-region (default), ``"lm"`` unbounded
    Levenberg-Marquardt (scipy), ``"capfit"`` unbounded CapFit (Cappellari 2023),
    as in the original NGC 5728 analysis.
    """
    xs = np.where(np.arange(len(p0)) % 3 == 0, 1.0, 100.0)
    resid = lambda p: (model(p, v) - y) * w
    if fitter == "trf":
        p0 = np.clip(p0, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
        res = least_squares(resid, p0, bounds=(lo, hi), method="trf", x_scale=xs)
        return res.x if res.success else None
    if fitter == "lm":
        try:
            res = least_squares(resid, p0, method="lm", x_scale=xs)
        except Exception:
            return None
        p = res.x.copy()
    elif fitter == "capfit":
        cf = _get_capfit()
        try:
            p = np.asarray(cf(resid, p0, verbose=0).x, dtype=float)
        except Exception:
            return None
    else:
        raise ValueError("fitter must be 'trf', 'lm' or 'capfit'")
    p[2::3] = np.abs(p[2::3])        # sigma > 0 (unbounded fits)
    return p


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def fit_line(cube: Cube, line: LineSpec, fit_single: bool = True, fit_double: bool = True,
             dilate: int = 2, percentile_over: str = "fov", weighting: str = "none",
             fitter: str = "trf", p0_center: str = "moment",
             fit_all_spaxels: bool = False, spaxels=None, show_progress: bool = True) -> LineMaps:
    """Measure one line in one cube (see module docstring).

    Parameters
    ----------
    cube : the :class:`Cube` containing the line (``cube.agn_xy`` must be set).
    line : the :class:`LineSpec`.
    fit_single, fit_double : which Gaussian fits to run.
    dilate : grow the masked (faint) region by this many pixels.
    percentile_over : ``"fov"`` -> flux percentile over spaxels inside the field of view;
        ``"all"`` -> over the full array incl. empty padding (what the original notebooks did).
    weighting : ``"none"`` (unweighted), ``"err"`` (cube ERR extension), or
        ``"sqrt"`` (uncertainty = sqrt(|flux in Jy|), as in the original NGC 5728 analysis;
        this up-weights the faint line wings).
    fitter : ``"trf"`` (bounded, default), ``"lm"`` (unbounded Levenberg-Marquardt) or
        ``"capfit"`` (unbounded CapFit, ``pip install capfit``; falls back to ``"lm"``).
        Unbounded fits correspond to the "fully unconstrained" fits of the paper.
    p0_center : initial centroid of the Gaussians: ``"moment"`` (moment-1 velocity of the
        spaxel) or ``"systemic"`` (0 km/s, as in the original analysis), plus ``p0_dv_kms``.
    fit_all_spaxels : fit every spaxel in the FOV, not just the good ones (slower).
    spaxels : list of (row, col) to fit (all others get NaN fits) - used by :func:`check_line`.
    """
    if cube.agn_xy is None:
        raise ValueError(f"Set the AGN position of cube '{cube.name}' first (set_agn_positions).")
    lo, hi = line.get_fit_window()
    lr = line.get_line_region()
    sel = (cube.wave >= lo) & (cube.wave <= hi)
    if sel.sum() < 10:
        raise ValueError(f"{line.name}: fit window {lo:.3f}-{hi:.3f} um has only {sel.sum()} "
                         f"channels in cube '{cube.name}' ({cube.wave_range})")
    wave = cube.wave[sel]
    cut = np.asarray(cube.data[sel], dtype=float)
    fov = np.any(cut != 0, axis=0)

    contsub = subtract_continuum(wave, cut, lr, line.cont_order)
    in_line = (wave >= lr[0]) & (wave <= lr[1])
    vel = C_KMS * (wave - line.wave) / line.wave

    # integrated flux over the line region, in erg/s/cm^2 (observed-frame wavelengths)
    wl_obs_A = wave[in_line] * (1 + cube.z) * 1e4
    flam = jy_to_flambda(contsub[in_line], wl_obs_A[:, None, None])
    flux = trapezoid(flam, wl_obs_A, axis=0)
    flux[~fov] = 0.0

    good = make_mask(flux, fov, line.mask_percentile, dilate, percentile_over)

    # moments (model-free)
    ny, nx = flux.shape
    m1, m2 = _moments(vel[in_line], contsub[in_line].reshape(in_line.sum(), -1))
    mom1, mom2 = m1.reshape(ny, nx), m2.reshape(ny, nx)

    lm = LineMaps(line=line, cube_name=cube.name, pixel_scale_pc=cube.pixel_scale_pc,
                  agn_xy=cube.agn_xy, wave=wave, vel_axis=vel, in_line=in_line,
                  contsub=contsub, flux=flux, good=good, mom1=mom1, mom2=mom2)

    keys1 = ["amp", "v", "sigma"]
    keys2 = ["amp_n", "v_n", "sigma_n", "amp_b", "v_b", "sigma_b"]
    single = {k: np.full((ny, nx), np.nan) for k in keys1}
    double = {k: np.full((ny, nx), np.nan) for k in keys2}

    if fitter == "capfit" and not _get_capfit():
        print("  capfit not installed (pip install capfit) - using fitter='lm' instead")
        fitter = "lm"

    if fit_single or fit_double:
        vb, sb = line.vel_bounds_kms, line.sigma_bounds_kms
        lo1 = np.array([0.0, vb[0], sb[0]]); hi1 = np.array([3.0, vb[1], sb[1]])
        lo2 = np.tile(lo1, 2); hi2 = np.tile(hi1, 2)
        err = None
        if weighting == "err" and cube.err is not None:
            err = np.asarray(cube.err[sel], dtype=float)

        if spaxels is not None:
            todo = np.array([tuple(p) for p in spaxels], dtype=int).reshape(-1, 2)
            lm.fitted_spaxels = [tuple(map(int, p)) for p in todo]
        else:
            todo = np.argwhere(fov if fit_all_spaxels else good)
        for (i, j) in progress(todo, desc=f"{line.name:>6s}", disable=not show_progress):
            y = contsub[:, i, j]
            ymax = np.max(y[in_line])
            if not np.isfinite(ymax) or ymax <= 0:
                continue
            yn = y / ymax
            if weighting == "err" and err is not None:
                e = err[:, i, j] / ymax
                w = np.where(e > 0, 1.0 / np.where(e > 0, e, 1), 0.0)
            elif weighting == "sqrt":
                yerr = np.sqrt(np.abs(y))          # in Jy, exactly as in the original notebooks
                yerr[yerr == 0] = 1e-6
                w = 1.0 / yerr
            else:
                w = np.ones_like(yn)
            if p0_center == "systemic" or not np.isfinite(mom1[i, j]):
                v0 = 0.0
            else:
                v0 = float(np.clip(mom1[i, j], *vb))
            s0 = np.clip(mom2[i, j], sb[0] * 1.5, sb[1] / 1.5) if np.isfinite(mom2[i, j]) else 100.0

            if fit_single:
                p = _fit(_single_model, np.array([1.0, v0, s0]), lo1, hi1, vel, yn, w, fitter)
                if p is not None:
                    single["amp"][i, j], single["v"][i, j], single["sigma"][i, j] = p[0] * ymax, p[1], p[2]

            if fit_double:
                (fa, fb), (dva, dvb), (sa, sbb) = line.p0_amp_frac, line.p0_dv_kms, line.p0_sigma_kms
                p0 = np.array([fa, v0 + dva, sa, fb, v0 + dvb, sbb])
                p = _fit(_double_model, p0, lo2, hi2, vel, yn, w, fitter)
                if p is not None:
                    a, b = (p[:3], p[3:]) if p[2] <= p[5] else (p[3:], p[:3])   # narrow first
                    for k, val in zip(keys2, [a[0] * ymax, a[1], a[2], b[0] * ymax, b[1], b[2]]):
                        double[k][i, j] = val

    lm.single, lm.double = single, double
    return lm


def fit_lines(cubes: Dict[str, Cube], lines: Dict[str, LineSpec], **kwargs) -> Dict[str, LineMaps]:
    """Run :func:`fit_line` for every line (each line must have ``line.cube`` set,
    see :func:`breaking_the_blend.cubes.assign_cubes`). Extra keywords go to :func:`fit_line`."""
    out = {}
    for name, line in lines.items():
        if line.cube is None:
            raise ValueError(f"{name}: line.cube is None - run assign_cubes() first")
        out[name] = fit_line(cubes[line.cube], line, **kwargs)
        print(f"  -> {out[name]}")
    return out


def spaxel_fit_curves(lm: LineMaps, i: int, j: int) -> Dict[str, np.ndarray]:
    """Model curves of the single/double fits in spaxel (row i, column j), for plotting."""
    v = lm.vel_axis
    out = {"v": v, "data": lm.contsub[:, i, j]}
    s = lm.single
    if s and np.isfinite(s["amp"][i, j]):
        out["single"] = gaussian(v, s["amp"][i, j], s["v"][i, j], s["sigma"][i, j])
    d = lm.double
    if d and np.isfinite(d["amp_n"][i, j]):
        out["narrow"] = gaussian(v, d["amp_n"][i, j], d["v_n"][i, j], d["sigma_n"][i, j])
        out["broad"] = gaussian(v, d["amp_b"][i, j], d["v_b"][i, j], d["sigma_b"][i, j])
        out["double"] = out["narrow"] + out["broad"]
    return out


def check_line(cube: Cube, line: LineSpec, n_spaxels: int = 6, seed: int = 0, spaxels=None,
               **fit_kwargs) -> LineMaps:
    """Quick test of a line's settings: maps and mask for all spaxels, but Gaussian fits only
    for the brightest good spaxel plus ``n_spaxels - 1`` random good ones (or ``spaxels``).
    Use with :func:`breaking_the_blend.plotting.plot_spaxel_fits` and :func:`check_summary`.
    """
    lm = fit_line(cube, line, fit_single=False, fit_double=False, show_progress=False,
                  **{k: v for k, v in fit_kwargs.items() if k in ("dilate", "percentile_over")})
    if spaxels is None:
        good = np.argwhere(lm.good)
        rng = np.random.default_rng(seed)
        peak = np.unravel_index(np.nanargmax(np.where(lm.good, lm.flux, -np.inf)), lm.shape)
        others = [tuple(p) for p in good if tuple(p) != tuple(peak)]
        pick = rng.choice(len(others), size=min(n_spaxels - 1, len(others)), replace=False) if others else []
        spaxels = [tuple(peak)] + [others[k] for k in pick]
    return fit_line(cube, line, spaxels=spaxels, show_progress=False, **fit_kwargs)


def check_summary(checks: Dict[str, LineMaps], ratio_warn: float = 1.3, verbose: bool = True) -> Dict[str, dict]:
    """Table of the checked spaxels per line: failed fits, fits at a bound, and how often the
    two Gaussians are not clearly different (broad/narrow sigma ratio < ``ratio_warn``)."""
    out = {}
    if verbose:
        print(f"{'line':>9s} {'N':>3s} {'failed':>7s} {'at bound':>9s} {'similar sigmas':>15s} "
              f"{'median sigma_n':>15s} {'median sigma_b':>15s}")
    for name, lm in checks.items():
        sp = lm.fitted_spaxels or [tuple(p) for p in np.argwhere(lm.good)]
        d = lm.double
        sn = np.array([d["sigma_n"][p] for p in sp]); sb = np.array([d["sigma_b"][p] for p in sp])
        vn = np.array([d["v_n"][p] for p in sp]); vb = np.array([d["v_b"][p] for p in sp])
        ok = np.isfinite(sn) & np.isfinite(sb)
        lo_s, hi_s = lm.line.sigma_bounds_kms
        lo_v, hi_v = lm.line.vel_bounds_kms
        tol = 1e-3
        at_bound = ok & ((np.minimum(sn, sb) <= lo_s * (1 + tol)) | (np.maximum(sn, sb) >= hi_s * (1 - tol)) |
                         (np.minimum(vn, vb) <= lo_v + 1) | (np.maximum(vn, vb) >= hi_v - 1))
        similar = ok & (sb < ratio_warn * sn)
        res = dict(n=len(sp), failed=int((~ok).sum()), at_bound=int(at_bound.sum()), similar=int(similar.sum()),
                   sigma_n=float(np.nanmedian(sn)) if ok.any() else np.nan,
                   sigma_b=float(np.nanmedian(sb)) if ok.any() else np.nan)
        out[name] = res
        if verbose:
            flag = "  <-- check" if (res["failed"] or res["at_bound"] or res["similar"] > res["n"] / 2) else ""
            print(f"{lm.label:>9s} {res['n']:3d} {res['failed']:7d} {res['at_bound']:9d} {res['similar']:15d} "
                  f"{res['sigma_n']:15.0f} {res['sigma_b']:15.0f}{flag}")
    return out

"""
STEP 5 - Use the stacked map to build star-forming and AGN spectra (paper Sect. 4.2.2)
and to split continuum maps between the two components (Sect. 4.2.2, Fig. S7).

The stacked map lives on a parsec grid centred on the AGN, so it can be
projected onto *any* cube (every NIRSpec / MIRI band) using that cube's AGN
position and pixel scale.

Two extraction methods:

* ``"threshold"`` (paper): sum spaxels with f_out > ``agn_min`` (AGN-dominated)
  and with f_out < ``sf_max`` (star-formation dominated);
* ``"weighted"``: every spaxel contributes ``f_out * spectrum`` to the AGN
  spectrum and ``f_disc * spectrum`` to the SF spectrum (assumes the
  ionised-gas fractions apply to all emission - an approximation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .cubes import Cube
from .stacking import StackedMap


@dataclass
class ComponentSpectra:
    cube_name: str
    wave: np.ndarray
    agn: np.ndarray        # summed spectrum [Jy]
    sf: np.ndarray
    agn_mask: Optional[np.ndarray]
    sf_mask: Optional[np.ndarray]
    f_disc_map: np.ndarray = field(repr=False)
    method: str = "threshold"
    pixel_scale: float = 1.0      # arcsec, to compare sky areas between bands
    footprint: Optional[np.ndarray] = field(default=None, repr=False)   # spaxels with data in this cube
    agn_err: Optional[np.ndarray] = field(default=None, repr=False)     # 1-sigma uncertainty [Jy]
    sf_err: Optional[np.ndarray] = field(default=None, repr=False)
    err_method: Optional[str] = None

    def __repr__(self):
        n = "" if self.agn_mask is None else f", N_agn={self.agn_mask.sum()}, N_sf={self.sf_mask.sum()}"
        return f"ComponentSpectra({self.cube_name}, {self.method}{n})"


def fraction_on_cube(stack: StackedMap, cube: Cube, which: str = "f_disc") -> np.ndarray:
    """The stacked map projected onto the pixel grid of ``cube`` (NaN where undefined)."""
    x, y = cube.offsets_pc()
    return stack.sample(x, y, which)


def cube_footprint(cube: Cube, min_valid: float = 0.5) -> np.ndarray:
    """Spaxels that hold data in this cube (non-zero in more than ``min_valid`` of the channels).
    JWST cubes are padded with empty (NaN -> 0) spaxels around the observed field."""
    return (np.asarray(cube.data) != 0).mean(axis=0) > min_valid


def _sum_with_gaps(data, w, min_valid=0.5):
    """Weighted sum over spaxels per channel; channels where most of the contributing spaxels
    have no data (detector gaps / edges are stored as exact zeros) become NaN."""
    valid = (data != 0)
    wsum = np.einsum("lij,ij->l", valid.astype(float), w)
    tot = np.einsum("lij,ij->l", data, w)
    out = tot.astype(float)
    out[wsum < min_valid * w.sum()] = np.nan
    return out


def _err_quadrature(err, w):
    """Uncertainty of a weighted sum of spaxels: sqrt(sum (w_i sigma_i)^2), per channel."""
    return np.sqrt(np.einsum("lij,ij->l", np.asarray(err, float) ** 2, w ** 2))


def empirical_errors(flux, window: int = 31, detrend: int = 7) -> np.ndarray:
    """Noise estimated from the spectrum itself: the robust (MAD) scatter of the flux about a
    running median (``detrend`` channels), measured in a sliding ``window`` of channels.
    It includes noise correlated between spaxels (which the pipeline ERR ignores) but is
    inflated close to strong, narrow lines."""
    from scipy.ndimage import median_filter
    f = np.asarray(flux, float)
    ok = np.isfinite(f)
    out = np.full_like(f, np.nan)
    if ok.sum() < window:
        return out
    fi = np.interp(np.arange(f.size), np.flatnonzero(ok), f[ok])
    resid = fi - median_filter(fi, size=detrend, mode="nearest")
    mad = median_filter(np.abs(resid - median_filter(resid, size=window, mode="nearest")), size=window, mode="nearest")
    # residual about a 'detrend'-channel median underestimates the per-channel scatter slightly
    out[ok] = 1.4826 * mad[ok] * np.sqrt(detrend / (detrend - 1.0))
    return out


def extract_spectra(cube: Cube, stack: StackedMap, method: str = "threshold",
                    agn_min: float = 0.8, sf_max: float = 0.2, min_spaxels: int = 3,
                    err_method: Optional[str] = "cube") -> ComponentSpectra:
    """AGN and star-forming spectra from one cube (see module docstring).

    agn_min : minimum f_out for an AGN-dominated spaxel (paper: 0.8).
    sf_max : maximum f_out for a star-formation-dominated spaxel (paper: 0.2).
    min_spaxels : a component spectrum built from fewer spaxels than this is set to NaN
        (e.g. a small NIRSpec field that barely reaches the star-forming ring).
    err_method : uncertainty of each extracted spectrum (1 sigma, Jy):

        * ``"cube"`` - the cube's ERR extension summed in quadrature over the selected spaxels
          (needs the cubes loaded with ``load_err=True``). Treats spaxels as independent, so it
          is a lower limit where neighbouring spaxels share noise (resampled cubes);
        * ``"empirical"`` - the measured channel-to-channel scatter of the extracted spectrum
          (:func:`empirical_errors`); includes correlated noise, but is inflated near lines;
        * ``"max"`` - the larger of the two in each channel (conservative);
        * ``None`` - no errors.

    Channels without data (detector gaps, e.g. the NIRSpec G395H NRS1/NRS2 gap) are NaN, and only
    spaxels inside the cube's own observed field are used (the stacked map, built from several bands,
    usually extends beyond the smallest fields).
    """
    fd = fraction_on_cube(stack, cube)
    fo = 1.0 - fd
    data = np.asarray(cube.data, dtype=float)
    footprint = cube_footprint(cube)
    ok = np.isfinite(fd) & footprint          # the stacked map can extend beyond this cube's field
    if method == "threshold":
        agn_mask = ok & (fo >= agn_min)
        sf_mask = ok & (fo <= sf_max)
        w_agn, w_sf = agn_mask.astype(float), sf_mask.astype(float)
    elif method == "weighted":
        agn_mask = sf_mask = None
        w_agn, w_sf = np.where(ok, fo, 0.0), np.where(ok, fd, 0.0)
    else:
        raise ValueError("method must be 'threshold' or 'weighted'")
    agn, sf = _sum_with_gaps(data, w_agn), _sum_with_gaps(data, w_sf)

    agn_err = sf_err = None
    if err_method is not None:
        if err_method not in ("cube", "empirical", "max"):
            raise ValueError("err_method must be 'cube', 'empirical', 'max' or None")
        e_cube_a = e_cube_s = None
        if err_method in ("cube", "max"):
            if cube.err is None:
                raise ValueError(f"cube '{cube.name}' has no ERR data - load the cubes with load_err=True, "
                                 "or use err_method='empirical'")
            e_cube_a, e_cube_s = _err_quadrature(cube.err, w_agn), _err_quadrature(cube.err, w_sf)
        if err_method == "cube":
            agn_err, sf_err = e_cube_a, e_cube_s
        else:
            e_emp_a, e_emp_s = empirical_errors(agn), empirical_errors(sf)
            if err_method == "empirical":
                agn_err, sf_err = e_emp_a, e_emp_s
            else:
                agn_err, sf_err = np.fmax(e_cube_a, e_emp_a), np.fmax(e_cube_s, e_emp_s)
        agn_err = np.where(np.isfinite(agn), agn_err, np.nan)
        sf_err = np.where(np.isfinite(sf), sf_err, np.nan)

    if method == "threshold":
        if agn_mask.sum() < min_spaxels:
            agn[:] = np.nan
            if agn_err is not None:
                agn_err[:] = np.nan
        if sf_mask.sum() < min_spaxels:
            sf[:] = np.nan
            if sf_err is not None:
                sf_err[:] = np.nan
    return ComponentSpectra(cube.name, cube.wave.copy(), agn, sf, agn_mask, sf_mask, fd, method,
                            cube.pixel_scale, footprint, agn_err=agn_err, sf_err=sf_err, err_method=err_method)


def extract_all(cubes: Dict[str, Cube], stack: StackedMap, **kwargs) -> Dict[str, ComponentSpectra]:
    """:func:`extract_spectra` for every cube, ordered by wavelength."""
    out = {}
    for name, cube in sorted(cubes.items(), key=lambda kv: kv[1].wave_range[0]):
        out[name] = extract_spectra(cube, stack, **kwargs)
        print(out[name])
    return out


def _weight(s: ComponentSpectra, component: str) -> float:
    """How much a band's spectrum can be trusted for normalisation: the sky area it is built
    from (number of selected spaxels, or summed fractions, times the pixel area)."""
    m = s.agn_mask if component == "agn" else s.sf_mask
    if m is not None:
        n = float(m.sum())
    else:
        n = float(np.nansum(1 - s.f_disc_map if component == "agn" else s.f_disc_map))
    return n * s.pixel_scale ** 2


def _merge(W, F, E, w, f, e, min_overlap_points):
    """Scale (w, f, e) to (W, F) in their overlap and merge with a linear blend (errors are
    scaled and combined consistently). Returns (W, F, E, scale) or None if no usable overlap."""
    lo, hi = max(W.min(), w.min()), min(W.max(), w.max())
    cur = (w >= lo) & (w <= hi)
    if lo >= hi or cur.sum() < min_overlap_points:
        return None
    ratio = np.interp(w[cur], W, F) / f[cur]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size < min_overlap_points:
        return None
    med = np.median(ratio)
    mad = 1.4826 * np.median(np.abs(ratio - med))
    if mad > 0:
        ratio = ratio[np.abs(ratio - med) < 3 * mad]
    k = float(np.median(ratio))
    f, e = f * k, e * k
    grid = np.unique(np.concatenate([W[(W >= lo) & (W <= hi)], w[cur]]))
    t = (grid - lo) / (hi - lo)
    new_is_red = w.max() > W.max()
    wt_new = t if new_is_red else 1 - t
    blend = (1 - wt_new) * np.interp(grid, W, F) + wt_new * np.interp(grid, w, f)
    blend_e = np.sqrt(((1 - wt_new) * np.interp(grid, W, E)) ** 2 + (wt_new * np.interp(grid, w, e)) ** 2)
    if new_is_red:
        lo_part = (W[W < lo], F[W < lo], E[W < lo]); hi_part = (w[w > hi], f[w > hi], e[w > hi])
    else:
        lo_part = (w[w < lo], f[w < lo], e[w < lo]); hi_part = (W[W > hi], F[W > hi], E[W > hi])
    return (np.concatenate([lo_part[0], grid, hi_part[0]]),
            np.concatenate([lo_part[1], blend, hi_part[1]]),
            np.concatenate([lo_part[2], blend_e, hi_part[2]]), k)


def stitch_spectra(spectra: Dict[str, ComponentSpectra], component: str = "agn",
                   reference: Optional[str] = None, min_overlap_points: int = 5, verbose: bool = True,
                   with_errors: bool = False):
    """Join the per-band spectra into one.

    One band is the **reference** (kept unscaled); the others are scaled band by band outwards
    from it, using the (sigma-clipped) median flux ratio in each overlap, and blended linearly.
    By default the reference is the band whose spectrum covers the largest sky area (selected
    spaxels x pixel area), so the overall normalisation is not set by a small field of view.
    Bands without usable data (all NaN) are skipped.

    Returns (wave, flux, scale_factors), or (wave, flux, error, scale_factors) with
    ``with_errors=True`` - the errors are scaled with each band and combined in the blended
    overlaps (they do not include the uncertainty of the scale factors themselves).
    """
    segs = {}
    for s in spectra.values():
        w = np.asarray(s.wave, float)
        f = np.asarray(getattr(s, component), float)
        err = getattr(s, f"{component}_err", None)
        e = np.asarray(err, float) if err is not None else np.full_like(f, np.nan)
        ok = np.isfinite(w) & np.isfinite(f) & (f > 0)
        if ok.sum() >= min_overlap_points:
            o = np.argsort(w[ok])
            segs[s.cube_name] = (w[ok][o], f[ok][o], e[ok][o], _weight(s, component))
        elif verbose:
            print(f"{s.cube_name}: no usable {component} spectrum - skipped")
    if not segs:
        raise ValueError(f"No band has a usable {component} spectrum")
    names = sorted(segs, key=lambda n: segs[n][0][0])
    ref = reference if reference is not None else max(names, key=lambda n: segs[n][3])
    if ref not in segs:
        raise KeyError(f"reference band '{ref}' has no usable spectrum ({list(segs)})")
    i0 = names.index(ref)
    W, F, E = segs[ref][0], segs[ref][1], segs[ref][2]
    scales = {ref: 1.0}
    order = names[i0 + 1:] + names[:i0][::-1]          # redwards first, then bluewards
    for n in order:
        w, f, e, _ = segs[n]
        res = _merge(W, F, E, w, f, e, min_overlap_points)
        if res is None:
            scales[n] = 1.0
            keep = (w > W.max()) | (w < W.min())
            W, F, E = (np.concatenate([W, w[keep]]), np.concatenate([F, f[keep]]),
                       np.concatenate([E, e[keep]]))
            o = np.argsort(W)
            W, F, E = W[o], F[o], E[o]
            if verbose:
                print(f"{n}: no usable overlap - appended unscaled")
            continue
        W, F, E, scales[n] = res
        if verbose:
            print(f"{n}: scale = {scales[n]:.3f}")
    if verbose:
        print(f"reference band: {ref}")
    return (W, F, E, scales) if with_errors else (W, F, scales)


def decompose_continuum(cube: Cube, stack: StackedMap, wave_range: Tuple[float, float]) -> Dict[str, object]:
    """Split a continuum map (median over ``wave_range``, e.g. 21 um warm dust) into
    disc and outflow parts with the stacked fractions (paper Sect. 4.2.2, Fig. S7).

    Returns ``{"total", "disc", "outflow"}`` maps and ``frac_outflow`` = the outflow
    share of the total continuum over spaxels where the fraction is defined.
    """
    total = cube.continuum_image(wave_range)
    fd = fraction_on_cube(stack, cube)
    ok = np.isfinite(fd)
    disc = np.where(ok, total * fd, np.nan)
    out = np.where(ok, total * (1 - fd), np.nan)
    frac = np.nansum(out) / np.nansum(total[ok])
    print(f"{cube.name} {wave_range[0]}-{wave_range[1]} um: outflow share = {100 * frac:.1f}% "
          f"(over {ok.sum()} spaxels with a defined fraction)")
    return {"total": np.where(ok, total, np.nan), "disc": disc, "outflow": out,
            "frac_outflow": float(frac), "f_disc_map": fd}

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


def extract_spectra(cube: Cube, stack: StackedMap, method: str = "threshold",
                    agn_min: float = 0.8, sf_max: float = 0.2, min_spaxels: int = 3) -> ComponentSpectra:
    """AGN and star-forming spectra from one cube (see module docstring).

    agn_min : minimum f_out for an AGN-dominated spaxel (paper: 0.8).
    sf_max : maximum f_out for a star-formation-dominated spaxel (paper: 0.2).
    min_spaxels : a component spectrum built from fewer spaxels than this is set to NaN
        (e.g. a small NIRSpec field that barely reaches the star-forming ring).
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
        agn = _sum_with_gaps(data, agn_mask.astype(float))
        sf = _sum_with_gaps(data, sf_mask.astype(float))
        if agn_mask.sum() < min_spaxels:
            agn[:] = np.nan
        if sf_mask.sum() < min_spaxels:
            sf[:] = np.nan
    elif method == "weighted":
        agn_mask = sf_mask = None
        agn = _sum_with_gaps(data, np.where(ok, fo, 0.0))
        sf = _sum_with_gaps(data, np.where(ok, fd, 0.0))
    else:
        raise ValueError("method must be 'threshold' or 'weighted'")
    return ComponentSpectra(cube.name, cube.wave.copy(), agn, sf, agn_mask, sf_mask, fd, method, cube.pixel_scale,
                            footprint)


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


def _merge(W, F, w, f, min_overlap_points):
    """Scale (w, f) to (W, F) in their overlap and merge with a linear blend. Returns
    (W, F, scale) or None if there is no usable overlap."""
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
    f = f * k
    grid = np.unique(np.concatenate([W[(W >= lo) & (W <= hi)], w[cur]]))
    t = (grid - lo) / (hi - lo)
    new_is_red = w.max() > W.max()
    wt_new = t if new_is_red else 1 - t
    blend = (1 - wt_new) * np.interp(grid, W, F) + wt_new * np.interp(grid, w, f)
    lo_part = (W[W < lo], F[W < lo]) if new_is_red else (w[w < lo], f[w < lo])
    hi_part = (w[w > hi], f[w > hi]) if new_is_red else (W[W > hi], F[W > hi])
    return (np.concatenate([lo_part[0], grid, hi_part[0]]),
            np.concatenate([lo_part[1], blend, hi_part[1]]), k)


def stitch_spectra(spectra: Dict[str, ComponentSpectra], component: str = "agn",
                   reference: Optional[str] = None, min_overlap_points: int = 5, verbose: bool = True):
    """Join the per-band spectra into one.

    One band is the **reference** (kept unscaled); the others are scaled band by band outwards
    from it, using the (sigma-clipped) median flux ratio in each overlap, and blended linearly.
    By default the reference is the band whose spectrum covers the largest sky area (selected
    spaxels x pixel area), so the overall normalisation is not set by a small field of view. Bands without
    usable data (all NaN) are skipped.

    Returns (wave, flux, scale_factors).
    """
    segs = {}
    for s in spectra.values():
        w = np.asarray(s.wave, float)
        f = np.asarray(getattr(s, component), float)
        ok = np.isfinite(w) & np.isfinite(f) & (f > 0)
        if ok.sum() >= min_overlap_points:
            o = np.argsort(w[ok])
            segs[s.cube_name] = (w[ok][o], f[ok][o], _weight(s, component))
        elif verbose:
            print(f"{s.cube_name}: no usable {component} spectrum - skipped")
    if not segs:
        raise ValueError(f"No band has a usable {component} spectrum")
    names = sorted(segs, key=lambda n: segs[n][0][0])
    ref = reference if reference is not None else max(names, key=lambda n: segs[n][2])
    if ref not in segs:
        raise KeyError(f"reference band '{ref}' has no usable spectrum ({list(segs)})")
    i0 = names.index(ref)
    W, F = segs[ref][0], segs[ref][1]
    scales = {ref: 1.0}
    order = names[i0 + 1:] + names[:i0][::-1]          # redwards first, then bluewards
    for n in order:
        w, f, _ = segs[n]
        res = _merge(W, F, w, f, min_overlap_points)
        if res is None:
            scales[n] = 1.0
            keep = (w > W.max()) | (w < W.min())
            W, F = np.concatenate([W, w[keep]]), np.concatenate([F, f[keep]])
            o = np.argsort(W)
            W, F = W[o], F[o]
            if verbose:
                print(f"{n}: no usable overlap - appended unscaled")
            continue
        W, F, scales[n] = res
        if verbose:
            print(f"{n}: scale = {scales[n]:.3f}")
    if verbose:
        print(f"reference band: {ref}")
    return W, F, scales


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

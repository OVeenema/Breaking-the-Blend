"""
Spaxel-by-spaxel extinction from the mid-IR H2 rotational lines (S(1)-S(7)) and the extinction
correction of the AGN- and SF-dominated spectra.

Method (Donnan et al. 2024, MNRAS 529, 1386, Sect. 4.3.2 and Fig. 7)
---------------------------------------------------------------------
The H2 0-0 S(3) line (9.665 um) sits in the 9.8 um silicate absorption, so with dust in front of the
gas it falls below the other transitions in the rotation (excitation) diagram, log(N_u/g_u) against
E_u/k. A single foreground screen,

    F_obs = F_int exp(-tau_9.8 tau(lambda)),         tau(lambda) = extinction law normalised to 1 at 9.8 um,

is applied with increasing tau_9.8 until the dip disappears:

* ``method="smoothness"`` (the paper, their eq. 18): minimise the summed squared discrete Laplacian
  P_H2 = sum_i [log(N_{i-1}/g_{i-1}) - 2 log(N_i/g_i) + log(N_{i+1}/g_{i+1})]^2 of the corrected diagram;
* ``method="powerlaw"``: fit a power law N_u/g_u = a E_u^b (a straight line in log-log; expected for a
  power-law distribution of gas temperatures) to the corrected transitions other than S(3), and take the
  tau_9.8 for which the corrected S(3) lies on it.

Both are linear in tau_9.8 (log N_corr = log N_obs + tau_9.8 tau(lambda)), so the best value is found
analytically in every spaxel; its uncertainty comes from Monte-Carlo perturbations of the line fluxes.

Steps (functions)
-----------------
1. :func:`h2_line_maps` - continuum-subtracted H2 line flux maps (native grid of each cube, or
   PSF-matched and resampled onto one target grid, e.g. the MRS channel-3 cube holding S(1));
2. :func:`fit_tau` - tau_9.8 in every spaxel (-> :class:`TauMap`);
3. :func:`tau_on_cube` - the tau_9.8 map on the grid of any cube (gaps filled, optionally smoothed);
4. :func:`extinction_corrected_spectra` - Step-5 extraction (same spaxels) of cubes corrected spaxel by
   spaxel with exp(+tau_9.8 tau(lambda)).

Extinction law
--------------
Default (``law="donnan2023"``): the law of Donnan et al. (2023a, appendix A), used by Donnan et al. (2024):
tau(lambda) = 0.1 (9.8/lambda)^1.7 + 0.9 tau_Sil(lambda), with tau_Sil the empirical silicate profile (9.8 and 18 um,
including crystalline features) of IRAS 08572+3915, normalised to 1 at 9.8 um. The table is bundled
(``data/D23ExtCurve.txt``, from Donnan's SPIRIT code, BSD-3 licence). ``law="kvt"`` is the PAHFIT power law + silicate
law (Smith et al. 2007; Kemper, Vriend & Tielens 2004 profile, (1 - 0.1) S_KVT + 0.1 (9.7/lambda)^1.7). Any other law
can be passed as a ``(wave, tau)`` table or a function of wavelength.

Caveat: the correction assumes the whole spectrum of a spaxel (continuum, PAHs, ionised lines) is behind
the same screen as the warm H2. Donnan et al. find different components see different extinction, so the
corrected spectra show the effect of the H2 extinction, not necessarily the intrinsic spectra.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt

from .cubes import Cube
from .extraction import cube_footprint, extract_spectra
from .lines import C_KMS
from .psf import psf_fwhm_arcsec

H_PLANCK = 6.62607015e-34      # J s
C_MS = 299792458.0             # m / s
ARCSEC_PER_RAD = 206264.806


# ======================================================================
# H2 rotational lines
# ======================================================================
@dataclass(frozen=True)
class H2Line:
    """One H2 0-0 S(J) transition.

    wave : rest wavelength [um]; A : Einstein coefficient [s^-1]; Eu : upper-level energy E_u/k [K];
    g : upper-level degeneracy (2J_u+1, x3 for ortho levels, i.e. ortho-to-para = 3);
    exclude : ``((wave_um, halfwidth_kms), ...)`` nearby lines kept out of the continuum windows (the windows are
        extended beyond them);
    deblend : wavelengths [um] of lines blended with this one - fitted together with it as Gaussians (the H2
        velocity and width are tied to another H2 line measured on the same grid, see :func:`h2_line_maps`).
    """

    name: str
    label: str
    wave: float
    A: float
    Eu: float
    g: int
    exclude: Tuple[Tuple[float, float], ...] = ()
    deblend: Tuple[float, ...] = ()
    cont_order: int = 1


# Roueff et al. (2019) / Wolniewicz et al. (1998) values
H2_CATALOGUE: Dict[str, H2Line] = {l.name: l for l in [
    H2Line("S1", "S(1)", 17.0348, 4.76e-10, 1015.0, 21),
    H2Line("S2", "S(2)", 12.2786, 2.75e-9, 1682.0, 9, exclude=((12.3719, 600.0),)),        # Hu alpha
    H2Line("S3", "S(3)", 9.6649, 9.84e-9, 2504.0, 33),
    H2Line("S4", "S(4)", 8.0250, 2.64e-8, 3474.0, 13, cont_order=2),                      # 7.7 um PAH wing
    H2Line("S5", "S(5)", 6.9095, 5.88e-8, 4586.0, 45, exclude=((6.9853, 1500.0),)),       # [Ar II]
    H2Line("S6", "S(6)", 6.1086, 1.14e-7, 5829.0, 17, cont_order=2),                      # 6.2 um PAH wing
    H2Line("S7", "S(7)", 5.5112, 2.00e-7, 7197.0, 57, exclude=((5.4466, 1500.0),), deblend=(5.5032,)),  # [Fe VIII]; [Mg VII]
    H2Line("S8", "S(8)", 5.0531, 3.24e-7, 8677.0, 21, deblend=(5.0623,)),                 # [Fe II]
]}
DEFAULT_H2_LINES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")


def get_h2_lines(names: Iterable[str] = DEFAULT_H2_LINES) -> Dict[str, H2Line]:
    """``{name: H2Line}`` ordered by upper-level energy (``"S1"`` ... ``"S8"``, or ``"S(3)"`` style)."""
    out = []
    for n in names:
        key = str(n).replace("(", "").replace(")", "").upper()
        if key not in H2_CATALOGUE:
            raise KeyError(f"Unknown H2 line '{n}'. Available: {list(H2_CATALOGUE)}")
        out.append(H2_CATALOGUE[key])
    return {l.name: l for l in sorted(out, key=lambda l: l.Eu)}


# ======================================================================
# Extinction law
# ======================================================================
_KVT_WAVE = np.array([8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.7, 9.75, 9.8, 10.0, 10.2, 10.4, 10.6,
                      10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.4, 12.6, 12.7])
_KVT_TAU = np.array([.06, .09, .16, .275, .415, .575, .755, .895, .98, .99, 1.0, .99, .94, .83, .745, .655,
                     .58, .525, .43, .35, .27, .20, .13, .09, .06, .045, .04314])


def _kvt_profile(lam):
    """Kemper, Vriend & Tielens (2004) silicate profile with PAHFIT's extensions (exponential fall-off
    below 8 um, Drude profile of the 18 um feature above 12.7 um)."""
    lam = np.asarray(lam, float)
    out = np.empty_like(lam)
    mid = (lam >= _KVT_WAVE[0]) & (lam <= _KVT_WAVE[-1])
    out[mid] = PchipInterpolator(_KVT_WAVE, _KVT_TAU)(lam[mid])
    short = lam < _KVT_WAVE[0]
    out[short] = _KVT_TAU[0] * np.exp(2.03 * (lam[short] - _KVT_WAVE[0]))
    long_ = lam > _KVT_WAVE[-1]
    drude = lambda x, x0=18.0, frac=0.247: 0.4 * frac ** 2 / ((x / x0 - x0 / x) ** 2 + frac ** 2)
    out[long_] = drude(lam[long_]) * (_KVT_TAU[-1] / drude(_KVT_WAVE[-1]))
    return out


LawType = Union[str, Callable, Tuple[np.ndarray, np.ndarray]]


@lru_cache(maxsize=1)
def _d23_table():
    """Donnan et al. (2023a) extinction curve (bundled, 1-40 um; negative values beyond 28 um clipped to 0)."""
    import os
    path = os.path.join(os.path.dirname(__file__), "data", "D23ExtCurve.txt")
    w, t = np.loadtxt(path, unpack=True)
    return w, np.clip(t, 0.0, None)


def extinction_law(lam, law: LawType = "donnan2023", beta: float = 0.1) -> np.ndarray:
    """tau(lambda) / tau_9.8: the extinction law normalised to 1 at 9.8 um (rest-frame um).

    law : ``"donnan2023"`` (default; Donnan et al. 2023a, as used by Donnan et al. 2024: 0.1 (9.8/lambda)^1.7 +
        0.9 tau_Sil, with the empirical silicate profile of IRAS 08572+3915 incl. crystalline features; bundled table,
        1-40 um, flat beyond), ``"kvt"`` (PAHFIT / Smith et al. 2007: (1 - beta) KVT silicate profile + beta
        (9.7/lambda)^1.7, with silicate absorption at 9.8 and 18 um), ``"powerlaw"`` ((9.8/lambda)^1.7 only - no silicate feature,
        for tests), a ``(wave, tau)`` table (interpolated; e.g. the Donnan et al. 2023a law if you have it) or a
        function ``tau(lambda)``. Every law is renormalised to 1 at 9.8 um.
    """
    lam = np.asarray(lam, float)

    def raw(x):
        x = np.asarray(x, float)
        if callable(law):
            return np.asarray(law(x), float)
        if isinstance(law, (tuple, list)) and len(law) == 2:
            w, t = np.asarray(law[0], float), np.asarray(law[1], float)
            o = np.argsort(w)
            return np.interp(x, w[o], t[o])
        if law in ("donnan2023", "D23"):
            w, t = _d23_table()
            return np.interp(x, w, t)
        if law == "kvt":
            return (1 - beta) * _kvt_profile(x) + beta * (9.7 / x) ** 1.7
        if law == "powerlaw":
            return (9.8 / x) ** 1.7
        raise ValueError("law must be 'donnan2023', 'kvt', 'powerlaw', a (wave, tau) table or a function")

    return raw(lam) / float(raw(np.array([9.8]))[0])


def extinction_factor(lam, tau98, law: LawType = "donnan2023") -> np.ndarray:
    """Correction factor F_int / F_obs = exp(+tau_9.8 tau(lambda)) of a foreground screen."""
    return np.exp(np.asarray(tau98, float) * extinction_law(lam, law))


# ======================================================================
# Regridding between cubes (sky-aligned cubes: same orientation, positions relative to the AGN)
# ======================================================================
@dataclass
class Grid:
    """Pixel grid of a map: shape (ny, nx), pixel scale [arcsec], AGN position (x, y) [pixel]."""

    shape: Tuple[int, int]
    pixel_scale: float
    agn_xy: Tuple[float, float]
    pc_per_arcsec: float = 1.0
    name: str = ""

    @classmethod
    def of(cls, cube: Cube) -> "Grid":
        if cube.agn_xy is None:
            raise ValueError(f"cube '{cube.name}' has no AGN position (set_agn_positions)")
        return cls(tuple(cube.shape2d), float(cube.pixel_scale), tuple(map(float, cube.agn_xy)),
                   float(cube.pc_per_arcsec), cube.name)

    def offsets_pc(self):
        ny, nx = self.shape
        yy, xx = np.indices((ny, nx), dtype=float)
        s = self.pixel_scale * self.pc_per_arcsec
        return (xx - self.agn_xy[0]) * s, (yy - self.agn_xy[1]) * s

    @property
    def extent_pc(self):
        ny, nx = self.shape
        s = self.pixel_scale * self.pc_per_arcsec
        x0, y0 = self.agn_xy
        return [(-0.5 - x0) * s, (nx - 0.5 - x0) * s, (-0.5 - y0) * s, (ny - 0.5 - y0) * s]


def _sample_coords(src: Grid, tgt: Grid, oversample: int):
    """Source-pixel coordinates (y, x) of an oversample x oversample sub-grid of every target pixel."""
    ny, nx = tgt.shape
    sub = (np.arange(oversample) + 0.5) / oversample - 0.5
    yt = (np.arange(ny)[:, None] + sub[None, :]).ravel()
    xt = (np.arange(nx)[:, None] + sub[None, :]).ravel()
    r = tgt.pixel_scale / src.pixel_scale
    ys = src.agn_xy[1] + (yt - tgt.agn_xy[1]) * r
    xs = src.agn_xy[0] + (xt - tgt.agn_xy[0]) * r
    return np.meshgrid(ys, xs, indexing="ij")


def regrid(img: np.ndarray, src: Grid, tgt: Grid, kind: str = "flux", oversample: int = 4,
           min_valid: float = 0.9, outside: str = "nan") -> np.ndarray:
    """Resample a map (ny, nx) or a stack of maps (n, ny, nx) from grid ``src`` onto grid ``tgt``.

    kind : ``"flux"`` - values per pixel (Jy/pixel, W m^-2 per pixel): converted to surface brightness,
        averaged over each target pixel and multiplied by its area (flux conserving);
        ``"mean"`` - intensive quantities (e.g. tau_9.8): averaged over each target pixel.
    NaN marks invalid pixels; target pixels with less than ``min_valid`` of their area on valid source
    pixels are NaN. ``outside="nearest"`` extends the edge values beyond the source map instead (only for
    maps without NaN).
    """
    img = np.asarray(img, float)
    single = img.ndim == 2
    stack = img[None] if single else img
    if kind == "flux":
        stack = stack / src.pixel_scale ** 2
    elif kind != "mean":
        raise ValueError("kind must be 'flux' or 'mean'")
    Y, X = _sample_coords(src, tgt, oversample)
    ny, nx = tgt.shape
    out = np.empty((stack.shape[0], ny, nx))
    if outside == "nearest":
        for k, m in enumerate(stack):
            v = map_coordinates(m, [Y, X], order=1, mode="nearest")
            out[k] = v.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))
    else:
        valid2d = np.isfinite(stack).all(axis=0)
        vw = map_coordinates(valid2d.astype(float), [Y, X], order=1, mode="constant", cval=0.0)
        vw = vw.reshape(ny, oversample, nx, oversample).sum(axis=(1, 3))
        frac = vw / oversample ** 2
        for k, m in enumerate(stack):
            v = map_coordinates(np.where(valid2d, m, 0.0), [Y, X], order=1, mode="constant", cval=0.0)
            num = v.reshape(ny, oversample, nx, oversample).sum(axis=(1, 3))
            with np.errstate(invalid="ignore", divide="ignore"):
                out[k] = np.where(frac >= min_valid, num / vw, np.nan)
    if kind == "flux":
        out *= tgt.pixel_scale ** 2
    return out[0] if single else out


def _smooth_nan(stack: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Gaussian smoothing of (n, ny, nx) maps ignoring NaN pixels (normalised convolution); NaN stays NaN."""
    if sigma_pix <= 0:
        return stack
    valid = np.isfinite(stack)
    num = gaussian_filter(np.where(valid, stack, 0.0), sigma=(0, sigma_pix, sigma_pix), mode="constant")
    den = gaussian_filter(valid.astype(float), sigma=(0, sigma_pix, sigma_pix), mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid & (den > 1e-3), num / den, np.nan)


# ======================================================================
# Step 1-2: H2 line flux maps
# ======================================================================
@dataclass
class LineFluxMap:
    """Continuum-subtracted flux map of one line [W m^-2 per pixel of ``grid``] with its 1-sigma error."""

    line: H2Line
    cube_name: str
    flux: np.ndarray
    err: np.ndarray
    grid: Grid = field(repr=False)
    fwhm_arcsec: Optional[float] = None      # PSF FWHM the map was matched to (None = native)
    method: str = "integrate"

    @property
    def snr(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.flux / self.err

    @property
    def name(self) -> str:
        return self.line.name

    def __repr__(self):
        ok = np.isfinite(self.flux)
        nd = int((np.nan_to_num(self.snr) >= 3).sum())
        return (f"LineFluxMap(H2 {self.line.label}, {self.cube_name} -> {self.grid.name}, {self.method}, "
                f"{ok.sum()} spaxels, {nd} with S/N >= 3)")


def _cube_for_line(cubes: Dict[str, Cube], lo: float, hi: float, wave: float, prefer_mrs: bool = True) -> Optional[str]:
    cands = [c for c in cubes.values() if c.covers(lo, hi)]
    if prefer_mrs and any(c.name.lower().startswith("ch") for c in cands):
        cands = [c for c in cands if c.name.lower().startswith("ch")]
    if not cands:
        return None
    return min(cands, key=lambda c: abs(wave - np.mean(c.wave_range)) / np.ptp(c.wave_range)).name


def _windows(line: H2Line, halfwidth_kms: float, cont_kms: float, exclude_kms: float):
    """Integrated line region, modelled region (incl. blended lines) and the fit window. The window extends
    ``cont_kms`` beyond the modelled region on each side, and further where a nearby excluded line eats into it."""
    lo, hi = line.wave * (1 - halfwidth_kms / C_KMS), line.wave * (1 + halfwidth_kms / C_KMS)
    blo, bhi = lo, hi
    for wb in line.deblend:                                  # blended lines: widen the modelled region
        blo, bhi = min(blo, wb * (1 - exclude_kms / C_KMS)), max(bhi, wb * (1 + exclude_kms / C_KMS))
    wlo, whi = blo * (1 - cont_kms / C_KMS), bhi * (1 + cont_kms / C_KMS)
    for wx, hw in line.exclude:
        elo, ehi = wx * (1 - hw / C_KMS), wx * (1 + hw / C_KMS)
        if elo < whi and ehi > bhi:                          # red side
            whi = max(whi, ehi + (whi - max(elo, bhi)))
        if ehi > wlo and elo < blo:                          # blue side
            wlo = min(wlo, elo - (min(ehi, blo) - wlo))
    return (lo, hi), (blo, bhi), (wlo, whi)


def _gauss_v(v, amp, v0, sig):
    return amp * np.exp(-0.5 * ((v - v0) / sig) ** 2)


def _continuum(wave, block, line: H2Line, halfwidth_kms, cont_kms, exclude_kms):
    """Polynomial continuum of every spaxel fitted to the line-free channels and subtracted.
    Returns resid (n, ngood) [Jy], good spaxel mask (ny*nx,), robust channel noise (ngood,), continuum mask,
    and the projection matrix (continuum channels -> polynomial coefficients) for the error propagation."""
    (lo, hi), (blo, bhi), _ = _windows(line, halfwidth_kms, cont_kms, exclude_kms)
    n = block.shape[0]
    flat = block.reshape(n, -1)
    good = np.isfinite(flat).all(axis=0)
    cmask = (wave < blo) | (wave > bhi)
    for wx, hw in line.exclude:
        cmask &= np.abs(wave - wx) > wx * hw / C_KMS
    order = line.cont_order
    if cmask.sum() <= order + 2 or not good.any():
        return None
    xw = wave - line.wave
    Vc = np.vander(xw[cmask], order + 1)
    M = np.linalg.pinv(Vc)                                   # (order+1, n_cont)
    coeffs = M @ flat[cmask][:, good]
    resid = flat[:, good] - np.vander(xw, order + 1) @ coeffs
    rc = resid[cmask]
    sig = 1.4826 * np.median(np.abs(rc - np.median(rc, axis=0)), axis=0)
    sig = np.where(sig > 0, sig, np.std(rc, axis=0))
    sig *= np.sqrt(cmask.sum() / max(cmask.sum() - order - 1, 1))      # degrees of freedom of the fit
    return resid, good, sig, cmask, M, np.vander(xw, order + 1)


def _dnu(wave):
    return C_MS / (wave * 1e-6) ** 2 * np.abs(np.gradient(wave)) * 1e-6          # Hz per channel


def _fit_kinematics(wave, block, line: H2Line, halfwidth_kms, cont_kms, exclude_kms, snr_min=5.0):
    """Velocity and dispersion [km/s] of a single Gaussian fitted to the line in every spaxel (NaN where the fit
    fails or the line is weaker than ``snr_min``)."""
    from scipy.optimize import least_squares
    ny, nx = block.shape[1:]
    vmap, smap = np.full(ny * nx, np.nan), np.full(ny * nx, np.nan)
    c = _continuum(wave, block, line, halfwidth_kms, cont_kms, exclude_kms)
    if c is None:
        return vmap.reshape(ny, nx), smap.reshape(ny, nx)
    resid, good, sig, _, _, _ = c
    (lo, hi), _, _ = _windows(line, halfwidth_kms, cont_kms, exclude_kms)
    sel = (wave >= lo) & (wave <= hi)
    v = C_KMS * (wave[sel] - line.wave) / line.wave
    idx = np.flatnonzero(good)
    for k in range(resid.shape[1]):
        y = resid[sel, k]
        s = sig[k] if sig[k] > 0 else 1.0
        if y.max() < snr_min * s / 2:
            continue
        try:
            r = least_squares(lambda p: (_gauss_v(v, *p) - y) / s, [y.max(), v[np.argmax(y)], 80.0],
                              bounds=([0, -halfwidth_kms, 15.0], [np.inf, halfwidth_kms, 0.6 * halfwidth_kms]))
        except Exception:
            continue
        flux_snr = r.x[0] * np.sqrt(2 * np.pi) * r.x[2] / (s * np.sqrt(abs(v[1] - v[0]) * r.x[2] * 2 * np.sqrt(np.pi)))
        if r.success and flux_snr >= snr_min:
            vmap[idx[k]], smap[idx[k]] = r.x[1], r.x[2]
    return vmap.reshape(ny, nx), smap.reshape(ny, nx)


def _measure_block(wave, block, line: H2Line, halfwidth_kms, cont_kms, exclude_kms, kin=None):
    """Line flux [W m^-2] and error of every spaxel of a (n, ny, nx) cut-out (Jy per pixel, NaN = invalid).
    ``kin`` = (velocity, dispersion) maps of another H2 line on the same grid, used to tie the H2 component when
    deblending."""
    from scipy.optimize import least_squares
    (lo, hi), (blo, bhi), _ = _windows(line, halfwidth_kms, cont_kms, exclude_kms)
    n, ny, nx = block.shape
    flux = np.full(ny * nx, np.nan)
    err = np.full(ny * nx, np.nan)
    c = _continuum(wave, block, line, halfwidth_kms, cont_kms, exclude_kms)
    if c is None:
        return flux.reshape(ny, nx), err.reshape(ny, nx)
    resid, good, sig, cmask, M, V = c
    dnu = _dnu(wave)
    sel = (wave >= lo) & (wave <= hi)
    # integration error incl. the continuum-fit uncertainty (exact for white noise)
    gvec = V[sel].T @ dnu[sel]
    noise_fac = np.sqrt((dnu[sel] ** 2).sum() + ((M.T @ gvec) ** 2).sum())
    if not line.deblend:
        f = (resid[sel] * dnu[sel, None]).sum(axis=0) * 1e-26                  # W m^-2
        flux[good], err[good] = f, sig * noise_fac * 1e-26
        return flux.reshape(ny, nx), err.reshape(ny, nx)

    # Gaussian deblending: H2 + each blended line, per spaxel
    bsel = (wave >= blo) & (wave <= bhi)
    v = C_KMS * (wave[bsel] - line.wave) / line.wave
    vb = [C_KMS * (wb - line.wave) / line.wave for wb in line.deblend]
    sep = min(abs(x) for x in vb)
    nu0 = C_MS / (line.wave * 1e-6)
    c1 = np.sqrt(2 * np.pi) / C_KMS * nu0 * 1e-26
    idx = np.flatnonzero(good)
    kv = ks = None
    if kin is not None:
        kv, ks = (np.ravel(kin[0]), np.ravel(kin[1]))
        okk = np.isfinite(kv) & np.isfinite(ks)
        v_med = np.median(kv[okk]) if okk.any() else 0.0
        s_med = np.median(ks[okk]) if okk.any() else 80.0
    for k in range(resid.shape[1]):
        y = resid[bsel, k]
        s = sig[k] if sig[k] > 0 else 1.0
        pk = max(np.nanmax(y), s)
        if kin is not None:                       # H2 kinematics tied to the reference line (or its median)
            v0 = kv[idx[k]] if np.isfinite(kv[idx[k]]) else v_med
            s0 = ks[idx[k]] if np.isfinite(ks[idx[k]]) else s_med
            lb, ub = [0.0, v0 - 60.0, 0.7 * s0], [np.inf, v0 + 60.0, 1.5 * s0]
            p0 = [0.5 * pk, v0, s0]
        else:                                     # free, but the H2 centroid stays closer to its own wavelength
            vlim = min(halfwidth_kms, 0.45 * sep)
            lb, ub = [0.0, -vlim, 20.0], [np.inf, vlim, 0.6 * halfwidth_kms]
            p0 = [0.5 * pk, 0.0, 80.0]
        for vv in vb:                             # blended line: its centroid may not cross half-way to H2
            p0 += [0.5 * pk, 0.0, 250.0]
            lb += [0.0, -exclude_kms if vv < 0 else -0.5 * vv, 30.0]
            ub += [np.inf, 0.5 * abs(vv) if vv < 0 else exclude_kms, exclude_kms]
        p0 = np.clip(p0, np.array(lb) + 1e-6, np.where(np.isfinite(ub), np.array(ub) - 1e-6, np.inf))

        def model(p):
            m = _gauss_v(v, p[0], p[1], p[2])
            for j, vv in enumerate(vb):
                m = m + _gauss_v(v, p[3 + 3 * j], vv + p[4 + 3 * j], p[5 + 3 * j])
            return m

        try:
            r = least_squares(lambda p: (model(p) - y) / s, p0, bounds=(lb, ub), x_scale="jac")
        except Exception:
            continue
        amp, sg = r.x[0], r.x[2]
        f = c1 * amp * sg
        try:
            cov = np.linalg.pinv(r.jac.T @ r.jac) * max(2 * r.cost / max(y.size - len(p0), 1), 1.0)
            gv = np.zeros(len(p0)); gv[0], gv[2] = c1 * sg, c1 * amp
            e = float(np.sqrt(max(gv @ cov @ gv, 0.0)))
        except Exception:
            e = np.nan
        # never below the noise integrated over the line (zero-amplitude fits carry no covariance information)
        e_int = s * np.sqrt((dnu[sel] ** 2).sum()) * 1e-26
        flux[idx[k]], err[idx[k]] = f, max(e if np.isfinite(e) else 0.0, e_int)
    return flux.reshape(ny, nx), err.reshape(ny, nx)


def h2_line_maps(cubes: Dict[str, Cube], lines: Iterable[str] = DEFAULT_H2_LINES, target: Optional[str] = None,
                 match_psf: bool = True, target_fwhm_arcsec: Optional[float] = None, halfwidth_kms: float = 700.0,
                 cont_kms: float = 1200.0, exclude_kms: float = 1500.0, oversample: int = 4,
                 prefer_mrs: bool = True, verbose: bool = True) -> Dict[str, LineFluxMap]:
    """Continuum-subtracted H2 line flux maps [W m^-2 per pixel].

    lines : H2 transitions (``"S1"`` ... ``"S8"``); each is measured in the cube that covers it best (MRS preferred).
    target : ``None`` - every map on its own cube's grid; a cube name - all maps resampled (flux conserving) onto
        that cube's grid; ``"S1"`` (or any line name) - onto the grid of the cube holding that line (S(1): MRS ch3).
    match_psf : convolve every cut-out with a Gaussian so its PSF FWHM (MRS: 0.033 lambda + 0.106", Argyriou et al.
        2023) matches ``target_fwhm_arcsec`` (default: the FWHM at the longest-wavelength line) before resampling.
    halfwidth_kms : half-width of the integrated line region; cont_kms : continuum on each side;
        exclude_kms : half-width of the region modelled for blended lines (``H2Line.deblend``, e.g. [Mg VII] next
        to S(7)) and the range of their velocity offsets.

    The continuum is a low-order polynomial fitted to every spaxel at once; the flux is the integral of the
    continuum-subtracted spectrum, the error the measured channel noise propagated through the integral and the
    continuum fit. Blended lines (S(7) + [Mg VII]) are fitted with Gaussians; on a common grid (``target``) the H2
    velocity and width are tied to the nearest unblended H2 line, which breaks the degeneracy between the two.
    """
    lines = get_h2_lines(lines)
    use = {}
    for name, ln in lines.items():
        _, _, (wlo, whi) = _windows(ln, halfwidth_kms, cont_kms, exclude_kms)
        cn = _cube_for_line(cubes, wlo, whi, ln.wave, prefer_mrs)
        if cn is None:
            if verbose:
                print(f"H2 {ln.label}: no loaded cube covers {wlo:.3f}-{whi:.3f} um - skipped")
            continue
        use[name] = cn
    if not use:
        raise ValueError("no H2 line is covered by the loaded cubes")

    tgt_grid = None
    if target is not None:
        key = str(target).replace("(", "").replace(")", "").upper()
        tname = use.get(key, target)
        if tname not in cubes:
            raise KeyError(f"target '{target}' is neither a loaded cube nor a measured H2 line ({list(use)})")
        tgt_grid = Grid.of(cubes[tname])
    lam_obs = {n: lines[n].wave * (1 + cubes[c].z) for n, c in use.items()}
    if match_psf and target_fwhm_arcsec is None:
        target_fwhm_arcsec = float(psf_fwhm_arcsec(max(lam_obs.values())))

    blocks = {}
    for name, cn in use.items():
        ln, cube = lines[name], cubes[cn]
        _, _, (wlo, whi) = _windows(ln, halfwidth_kms, cont_kms, exclude_kms)
        sel = (cube.wave >= wlo) & (cube.wave <= whi)
        block = np.asarray(cube.data[sel], float)
        block = np.where(cube_footprint(cube)[None], block, np.nan)
        block[block == 0] = np.nan                                   # detector gaps / empty spaxels
        src = Grid.of(cube)
        fwhm = None
        if match_psf:
            fw_src = float(psf_fwhm_arcsec(lam_obs[name]))
            if target_fwhm_arcsec > fw_src:
                ker = np.sqrt(target_fwhm_arcsec ** 2 - fw_src ** 2) / 2.3548 / cube.pixel_scale
                block = _smooth_nan(block, ker)
            fwhm = target_fwhm_arcsec
        grid = src
        if tgt_grid is not None and (tgt_grid.name != src.name):
            block = regrid(block, src, tgt_grid, kind="flux", oversample=oversample)
            grid = tgt_grid
        blocks[name] = (cube.wave[sel], block, grid, fwhm)

    out = {}
    order = sorted(use, key=lambda n: (bool(lines[n].deblend), lines[n].Eu))    # unblended lines first
    for name in order:
        ln = lines[name]
        w, block, grid, fwhm = blocks[name]
        kin = None
        if ln.deblend and tgt_grid is not None:
            refs = [n for n in out if not lines[n].deblend]
            if refs:
                ref = min(refs, key=lambda n: abs(lines[n].wave - ln.wave))
                kin = _fit_kinematics(*blocks[ref][:2], lines[ref], halfwidth_kms, cont_kms, exclude_kms)
                if verbose:
                    print(f"  H2 {ln.label}: deblended with the velocity and width of H2 {lines[ref].label} "
                          f"({np.isfinite(kin[0]).sum()} spaxels)")
        f, e = _measure_block(w, block, ln, halfwidth_kms, cont_kms, exclude_kms, kin=kin)
        out[name] = LineFluxMap(ln, use[name], f, e, grid, fwhm, "deblend" if ln.deblend else "integrate")
    out = {n: out[n] for n in sorted(out, key=lambda n: lines[n].Eu)}
    if verbose:
        for m in out.values():
            print(m)
    return out


# ======================================================================
# Step 3-4: tau_9.8 per spaxel
# ======================================================================
@dataclass
class TauMap:
    """tau_9.8 in every spaxel of ``grid`` from the H2 rotation diagram.

    tau : best value (NaN where it cannot be measured); tau_err : half the 16-84 per cent range of the Monte-Carlo
    values; ok : spaxels with a measurement; n_lines : transitions used per spaxel.
    lnNg, sig_ln : ln(N_u/g_u) [cm^-2] of the observed lines (n_lines, ny, nx) and their 1-sigma errors (NaN where
    a line is not used); Eu, k : upper-level energies [K] and extinction-law values tau(lambda)/tau_9.8 of the lines.
    """

    tau: np.ndarray
    tau_err: np.ndarray
    tau_lo: np.ndarray
    tau_hi: np.ndarray
    ok: np.ndarray
    n_lines: np.ndarray
    method: str
    law: object = field(repr=False)
    names: Tuple[str, ...] = ()
    labels: Tuple[str, ...] = ()
    Eu: np.ndarray = field(default=None, repr=False)
    k: np.ndarray = field(default=None, repr=False)
    lnNg: np.ndarray = field(default=None, repr=False)
    sig_ln: np.ndarray = field(default=None, repr=False)
    snr_s3: np.ndarray = field(default=None, repr=False)
    grid: Grid = field(default=None, repr=False)
    settings: dict = field(default_factory=dict, repr=False)

    @property
    def extent_pc(self):
        return self.grid.extent_pc

    def __repr__(self):
        t = self.tau[self.ok]
        s = (f"median {np.median(t):.2f}, 16-84%: {np.percentile(t, 16):.2f}-{np.percentile(t, 84):.2f}"
             if t.size else "no valid spaxels")
        return f"TauMap({self.method}, grid {self.grid.name}, {self.ok.sum()} spaxels; tau_9.8 {s})"


def column_densities(maps: Dict[str, LineFluxMap], snr_min: float = 3.0, sys_frac: float = 0.05):
    """ln(N_u/g_u) [N_u in cm^-2] and 1-sigma errors (relative flux error + ``sys_frac`` in quadrature) of every
    line on a common grid; lines with S/N < ``snr_min`` (or flux <= 0) are NaN. Optically thin emission:
    N_u = 4 pi F lambda / (h c A Omega_pixel)."""
    names = [n for n in sorted(maps, key=lambda n: maps[n].line.Eu)]
    grids = {maps[n].grid.name for n in names}
    if len(grids) > 1:
        raise ValueError(f"maps are on different grids {grids}: build them with h2_line_maps(..., target=...)")
    g0 = maps[names[0]].grid
    omega = (g0.pixel_scale / ARCSEC_PER_RAD) ** 2
    ln_ng, sig = [], []
    for n in names:
        m, ln = maps[n], maps[n].line
        with np.errstate(invalid="ignore", divide="ignore"):
            N = 4 * np.pi * m.flux * ln.wave * 1e-6 / (H_PLANCK * C_MS * ln.A * omega) * 1e-4      # cm^-2
            good = np.isfinite(m.flux) & (m.flux > 0) & (m.snr >= snr_min)
            ln_ng.append(np.where(good, np.log(N / ln.g), np.nan))
            sig.append(np.where(good, np.sqrt((m.err / m.flux) ** 2 + sys_frac ** 2), np.nan))
    Eu = np.array([maps[n].line.Eu for n in names])
    return names, np.array(ln_ng), np.array(sig), Eu, g0


def _tau_powerlaw(Y, V, W, x, k, i3):
    """Analytic tau for the power-law method. Y (L, S) ln(N/g); V valid; W weights; x = ln Eu; k law; i3 = S(3)."""
    Wf = np.where(V, W, 0.0)
    Wf[i3] = 0.0
    X = x[:, None]

    def pred(Z):
        Sw, Sx, Sxx = Wf.sum(0), (Wf * X).sum(0), (Wf * X ** 2).sum(0)
        Sy, Sxy = (Wf * Z).sum(0), (Wf * X * Z).sum(0)
        det = Sw * Sxx - Sx ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            b = (Sw * Sxy - Sx * Sy) / det
            a = (Sy - b * Sx) / Sw
        return a + b * x[i3]

    with np.errstate(invalid="ignore", divide="ignore"):
        p0 = pred(np.where(V, Y, 0.0))
        p1 = pred(np.broadcast_to(k[:, None], Y.shape))
        return -(Y[i3] - p0) / (k[i3] - p1)


def _tau_smooth(Y, V, k, i3):
    """Analytic tau minimising the summed squared discrete Laplacian (Donnan et al. 2024 eq. 18), using the
    valid transitions of each spaxel in order of energy."""
    L, S = Y.shape
    out = np.full(S, np.nan)
    pats, inv = np.unique(V.T, axis=0, return_inverse=True)
    inv = np.ravel(inv)
    for p_idx, pat in enumerate(pats):
        idx = np.flatnonzero(pat)
        if i3 not in idx or idx.size < 3 or idx[0] == i3 or idx[-1] == i3:
            continue
        cols = np.flatnonzero(inv == p_idx)
        D = np.zeros((idx.size - 2, idx.size))
        for r in range(idx.size - 2):
            D[r, r:r + 3] = (1.0, -2.0, 1.0)
        L0 = D @ Y[np.ix_(idx, cols)]
        L1 = D @ k[idx]
        out[cols] = -(L1[:, None] * L0).sum(0) / (L1 ** 2).sum()
    return out


def fit_tau(maps: Dict[str, LineFluxMap], method: str = "powerlaw", law: LawType = "donnan2023", snr_min: float = 3.0,
            sys_frac: float = 0.05, n_mc: int = 200, tau_range: Tuple[float, float] = (-1.0, 15.0),
            fit_lines: Optional[Sequence[str]] = None, seed: int = 0, verbose: bool = True) -> TauMap:
    """tau_9.8 in every spaxel from the H2 rotation diagram (maps from :func:`h2_line_maps` with a common ``target``).

    method : ``"powerlaw"`` (S(3) on the power law through the other corrected transitions) or ``"smoothness"``
        (minimum Laplacian penalty of the corrected diagram, Donnan et al. 2024 eq. 18 / Fig. 7).
    snr_min : transitions below this S/N are not used in a spaxel. A spaxel needs S(3) and at least one usable
        transition on each side of it in energy.
    sys_frac : fractional flux uncertainty added in quadrature (calibration, PSF matching); also the minimum weight
        scale of each point in the power-law fit.
    n_mc : Monte-Carlo realisations of the fluxes for the uncertainty.
    tau_range : allowed range; values outside are flagged (not ok). Negative values (S(3) above the trend) are
        kept so that noise does not bias the map; :func:`tau_on_cube` clips them to 0 by default.
    fit_lines : transitions to use (default: all maps), e.g. without S(7) where it is blended with [Mg VII].
    """
    if method not in ("powerlaw", "smoothness"):
        raise ValueError("method must be 'powerlaw' or 'smoothness'")
    if fit_lines is not None:
        keep = set(get_h2_lines(fit_lines))
        maps = {n: m for n, m in maps.items() if n in keep}
    if "S3" not in maps:
        raise ValueError("S(3) is needed to measure the extinction")
    names, lnNg, sig, Eu, grid = column_densities(maps, snr_min, sys_frac)
    k = np.array([extinction_law(maps[n].line.wave, law) for n in names]).ravel()
    i3 = names.index("S3")
    x = np.log(Eu)
    L, ny, nx = lnNg.shape
    Y = lnNg.reshape(L, -1)
    Sg = sig.reshape(L, -1)
    V = np.isfinite(Y)
    below = V[:i3].any(0) if i3 > 0 else np.zeros(Y.shape[1], bool)
    above = V[i3 + 1:].any(0) if i3 < L - 1 else np.zeros(Y.shape[1], bool)
    usable = V[i3] & below & above

    def solve(Ymat):
        if method == "powerlaw":
            return _tau_powerlaw(Ymat, V, 1.0 / np.where(V, Sg, 1.0) ** 2, x, k, i3)
        return _tau_smooth(np.where(V, Ymat, 0.0), V, k, i3)

    tau = np.where(usable, solve(Y), np.nan)
    rng = np.random.default_rng(seed)
    mc = np.full((n_mc, Y.shape[1]), np.nan)
    Yf = np.where(V, Y, 0.0)
    Sf = np.where(V, Sg, 0.0)
    for j in range(n_mc):
        mc[j] = solve(Yf + rng.normal(size=Y.shape) * Sf)
    mc[:, ~usable] = np.nan
    lo = hi = np.full(Y.shape[1], np.nan)
    if usable.any():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(mc, [16, 84], axis=0)
    ok = usable & np.isfinite(tau) & (tau >= tau_range[0]) & (tau <= tau_range[1])
    tau = np.where(ok, tau, np.nan)
    shape = (ny, nx)
    snr3 = maps["S3"].snr
    tm = TauMap(tau.reshape(shape), np.where(ok, 0.5 * (hi - lo), np.nan).reshape(shape),
                lo.reshape(shape), hi.reshape(shape), ok.reshape(shape), V.sum(0).reshape(shape), method, law,
                tuple(names), tuple(maps[n].line.label for n in names), Eu, k, lnNg, sig, snr3, grid,
                dict(snr_min=snr_min, sys_frac=sys_frac, n_mc=n_mc, tau_range=tau_range))
    if verbose:
        n_out = int((usable & ~ok & np.isfinite(np.where(usable, solve(Y), np.nan))).sum())
        print(tm)
        print(f"  {usable.sum()} spaxels with S(3) at S/N >= {snr_min:g} and transitions on both sides; "
              f"{n_out} outside tau_range {tau_range}; law tau(lambda)/tau_9.8 at the lines: "
              + ", ".join(f"{maps[n].line.label} {kk:.2f}" for n, kk in zip(names, k)))
    return tm


def penalty_curve(tm: TauMap, iy: int, ix: int, taus: np.ndarray) -> np.ndarray:
    """The quantity minimised in spaxel (iy, ix) as a function of tau_9.8: the Laplacian penalty P_H2
    (``"smoothness"``) or chi^2 of S(3) about the power law through the other corrected transitions (``"powerlaw"``)."""
    y = tm.lnNg[:, iy, ix]
    s = tm.sig_ln[:, iy, ix]
    v = np.isfinite(y)
    i3 = list(tm.names).index("S3")
    taus = np.asarray(taus, float)
    Yc = y[:, None] + taus[None] * tm.k[:, None]
    if tm.method == "smoothness":
        idx = np.flatnonzero(v)
        Z = Yc[idx]
        return ((Z[:-2] - 2 * Z[1:-1] + Z[2:]) ** 2).sum(0)
    x = np.log(tm.Eu)
    f = v.copy(); f[i3] = False
    w = 1.0 / s[f] ** 2
    out = np.empty(taus.size)
    for j in range(taus.size):
        b, a = np.polyfit(x[f], Yc[f, j], 1, w=np.sqrt(w))
        out[j] = ((Yc[i3, j] - (a + b * x[i3])) / s[i3]) ** 2
    return out


def powerlaw_fit(tm: TauMap, iy: int, ix: int, tau: float, include_s3: bool = False):
    """(slope, intercept) of ln(N/g) = a + b ln(E_u) through the corrected transitions of spaxel (iy, ix)."""
    y = tm.lnNg[:, iy, ix] + tau * tm.k
    s = tm.sig_ln[:, iy, ix]
    f = np.isfinite(y)
    if not include_s3:
        f[list(tm.names).index("S3")] = False
    b, a = np.polyfit(np.log(tm.Eu[f]), y[f], 1, w=1.0 / s[f])
    return b, a


def example_spaxels(tm: TauMap, n: int = 5, quantiles: Optional[Sequence[float]] = None,
                    min_snr_s3: Optional[float] = None) -> list:
    """``n`` spaxels (iy, ix) spanning the tau_9.8 distribution (quantiles), among those with a measurement and the
    brightest S(3) (S/N above the median of the measured spaxels, or ``min_snr_s3``)."""
    q = np.linspace(0.05, 0.95, n) if quantiles is None else np.asarray(quantiles)
    snr = np.nan_to_num(tm.snr_s3)
    ok = tm.ok & np.isfinite(tm.tau)
    if not ok.any():
        return []
    thr = np.median(snr[ok]) if min_snr_s3 is None else min_snr_s3
    cand = ok & (snr >= thr)
    if cand.sum() < n:
        cand = ok
    iy, ix = np.nonzero(cand)
    t = tm.tau[iy, ix]
    chosen = []
    for qq in q:
        target = np.quantile(t, qq)
        order = np.argsort(np.abs(t - target))
        for o in order:
            if (iy[o], ix[o]) not in chosen:
                chosen.append((int(iy[o]), int(ix[o])))
                break
    return chosen


# ======================================================================
# Step 5: the tau map on any cube, corrected cubes and spectra
# ======================================================================
def tau_on_cube(tm: TauMap, cube: Cube, fill: Union[str, float] = "nearest", clip_negative: bool = True,
                smooth_fwhm_arcsec: float = 0.0, max_err: Optional[float] = None, oversample: int = 4) -> np.ndarray:
    """The tau_9.8 map resampled (area-averaged) onto the pixel grid of ``cube``.

    fill : value for spaxels without a measurement - ``"nearest"`` (nearest measured spaxel), ``"median"`` (median of
        the map), ``"zero"`` (no correction) or a number. Positions outside the tau map get the nearest edge value.
    clip_negative : set negative tau_9.8 (S(3) above the trend: noise) to 0.
    smooth_fwhm_arcsec : optional Gaussian smoothing of the (filled) map before resampling, to reduce noise.
    max_err : treat spaxels with tau_err above this as unmeasured (then filled).
    """
    t = np.where(tm.ok, tm.tau, np.nan)
    if max_err is not None:
        t = np.where(tm.tau_err <= max_err, t, np.nan)
    if clip_negative:
        t = np.where(np.isfinite(t), np.clip(t, 0.0, None), np.nan)
    bad = ~np.isfinite(t)
    if bad.all():
        raise ValueError("the tau map has no valid spaxels")
    if bad.any():
        if fill == "nearest":
            _, (iy, ix) = distance_transform_edt(bad, return_indices=True)
            t = t[iy, ix]
        elif fill == "median":
            t = np.where(bad, np.nanmedian(t), t)
        elif fill == "zero":
            t = np.where(bad, 0.0, t)
        else:
            t = np.where(bad, float(fill), t)
    if smooth_fwhm_arcsec > 0:
        t = gaussian_filter(t, smooth_fwhm_arcsec / 2.3548 / tm.grid.pixel_scale, mode="nearest")
    return regrid(t, tm.grid, Grid.of(cube), kind="mean", oversample=oversample, outside="nearest")


def correct_cube(cube: Cube, tau_map: np.ndarray, law: LawType = "donnan2023") -> Cube:
    """A copy of ``cube`` with every spaxel multiplied by exp(+tau_9.8(x, y) tau(lambda)) (errors too)."""
    k = extinction_law(cube.wave, law)
    fac = np.exp(np.asarray(tau_map, float)[None] * k[:, None, None]).astype(np.float32)
    data = np.asarray(cube.data) * fac
    err = None if cube.err is None else np.asarray(cube.err) * fac
    return replace(cube, data=data, err=err)


def extinction_corrected_spectra(cubes: Dict[str, Cube], stack, tm: TauMap, law: Optional[LawType] = None,
                                 fill: Union[str, float] = "nearest", clip_negative: bool = True,
                                 smooth_fwhm_arcsec: float = 0.0, max_err: Optional[float] = None,
                                 verbose: bool = True, **extract_kwargs):
    """Step 5 again on extinction-corrected cubes: for every cube, the tau map on its grid (:func:`tau_on_cube`),
    every spaxel corrected with exp(+tau_9.8 tau(lambda)), then :func:`extract_spectra` with ``extract_kwargs``
    (use the Step-5 settings, so the same spaxels are summed). Cubes are corrected one at a time (no copies kept).

    Returns ``(spectra, tau_maps)``: ``{band: ComponentSpectra}`` and ``{band: tau_9.8 map on that band's grid}``.
    """
    law = tm.law if law is None else law
    spectra, taus = {}, {}
    for name, cube in sorted(cubes.items(), key=lambda kv: kv[1].wave_range[0]):
        t = tau_on_cube(tm, cube, fill=fill, clip_negative=clip_negative, smooth_fwhm_arcsec=smooth_fwhm_arcsec,
                        max_err=max_err)
        taus[name] = t
        spectra[name] = extract_spectra(correct_cube(cube, t, law), stack, **extract_kwargs)
        if verbose:
            sp = spectra[name]
            ta = np.median(t[sp.agn_mask]) if sp.agn_mask is not None and sp.agn_mask.any() else np.nan
            ts = np.median(t[sp.sf_mask]) if sp.sf_mask is not None and sp.sf_mask.any() else np.nan
            print(f"{sp}  median tau_9.8: AGN-dominated {ta:.2f}, SF-dominated {ts:.2f}")
    return spectra, taus

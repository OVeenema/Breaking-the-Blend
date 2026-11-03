"""
PSF correction of the extracted spectra (AGN point source).

The AGN is treated as an unresolved point source at the AGN position of each cube. For every band, the model PSF
gives, as a function of wavelength,

* ``EE(lambda)``     - the fraction of the point source's flux inside the nuclear aperture (aperture weights);
* ``f_R(lambda)``    - the fraction that leaks into each region R (AGN-dominated, f_out >= agn_min; SF-dominated,
  f_out <= sf_max), with the same spaxel weights as the extraction.

Then, with F_ap the flux measured in the aperture,

* **AGN**          = F_ap / EE                      (total point-source flux, aperture-corrected);
* **AGN-outflow**  = F(AGN region) - (f_AGN / EE) F_ap   (AGN-dominated region with the point source removed);
* **SF**           = F(SF region)  - (f_SF  / EE) F_ap   (SF-dominated region with the AGN's PSF wings removed).

All three are linear in the cube, so the errors are propagated exactly from the ERR extension, including the
covariance of the pixels shared by the aperture and a region:
var = sum (w_R - k w_ap)^2 sigma^2 with k = f_R / EE. A PSF-model uncertainty of ``psf_sys`` x (size of each
correction) is added in quadrature.

PSF models
----------
* ``"webbpsf"`` - STPSF (formerly WebbPSF; either package is used) monochromatic PSFs of the MIRI/MRS or NIRSpec IFU
  at each sample wavelength. Needs the package and its data files (STPSF_PATH / WEBBPSF_PATH).
* ``"gaussian"`` - a circular Gaussian with FWHM = 0.033 lambda + 0.106 arcsec (MRS, Argyriou et al. 2023;
  lambda in observed micron) or ~0.033 lambda (NIRSpec), scaled by ``fwhm_scale``. Useful without the PSF data,
  and as a cross-check.

Notes: models of the instrument PSF do not include the broadening by cube building, which makes real MRS cube
PSFs somewhat broader (cf. Argyriou et al. 2023); ``symmetric=True`` (default) azimuthally averages the PSF, which
avoids having to match the PSF orientation to the cube (sky-aligned cubes are rotated relative to the instrument).
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import erf

from .cubes import Cube
from .extraction import ApertureSpectrum, ComponentSpectra, _sum_with_gaps, empirical_errors

MRS_SUBBAND = {"short": "A", "medium": "B", "long": "C"}


def psf_fwhm_arcsec(lam_obs_um):
    """Approximate PSF FWHM [arcsec] of the cubes: MIRI/MRS 0.033 lambda + 0.106 (Argyriou et al. 2023);
    NIRSpec (lambda < 4.9 um) ~1.03 lambda/D, at least 0.1 arcsec."""
    lam = np.asarray(lam_obs_um, float)
    return np.where(lam < 4.9, np.maximum(0.0327 * lam, 0.10), 0.033 * lam + 0.106)


def _psf_package():
    for name in ("stpsf", "webbpsf"):
        try:
            return __import__(name)
        except ImportError:
            continue
    return None


def _gaussian_image(fwhm_arcsec, pixscale_o, n):
    """Pixel-integrated circular Gaussian (sums to ~1) on an n x n grid of pixel size pixscale_o [arcsec]."""
    s = fwhm_arcsec / 2.3548 / pixscale_o
    c = (n - 1) / 2
    edges = np.arange(n + 1) - 0.5 - c
    g = 0.5 * (erf(edges[1:] / (np.sqrt(2) * s)) - erf(edges[:-1] / (np.sqrt(2) * s)))
    return np.outer(g, g)


def _webbpsf_image(cube_name, lam_obs_um, pixscale_o, n, oversample):
    return _webbpsf_image_cached(cube_name, round(float(lam_obs_um), 4), round(float(pixscale_o), 6), int(n),
                                 int(oversample))


@lru_cache(maxsize=256)
def _webbpsf_image_cached(cube_name, lam_obs_um, pixscale_o, n, oversample):
    """STPSF monochromatic IFU PSF (cached: each band/wavelength is computed once per session). STPSF downloads
    its data files (~90 MB) to ~/data/stpsf-data on first use unless STPSF_PATH points elsewhere."""
    pkg = _psf_package()
    if pkg is None:
        raise ImportError("PSF model 'webbpsf' needs the stpsf (formerly webbpsf) package and its data files - "
                          "see https://stpsf.readthedocs.io - or use model='gaussian'")
    name = cube_name.lower()
    if name.startswith("ch"):
        inst = pkg.MIRI()
        inst.mode = "IFU"
        inst.band = f"{name[2]}{MRS_SUBBAND[name.split('-')[1]]}"
    else:
        inst = pkg.NIRSpec()
        inst.mode = "IFU"
        disp, filt = name.upper().split("-")[:2]
        inst.disperser, inst.filter = disp, filt
    inst.pixelscale = pixscale_o * oversample
    with warnings.catch_warnings():                    # field-point clipping notices of the OTE model are harmless
        warnings.filterwarnings("ignore", message=".*field dependence model.*")
        hdul = inst.calc_psf(monochromatic=lam_obs_um * 1e-6, oversample=oversample, fov_arcsec=n * pixscale_o)
    img = np.asarray(hdul[0].data, float)
    # crop / pad to n x n around the centre
    out = np.zeros((n, n))
    m = min(n, img.shape[0], img.shape[1])
    oy, ox = (img.shape[0] - m) // 2, (img.shape[1] - m) // 2
    o2 = (n - m) // 2
    out[o2:o2 + m, o2:o2 + m] = img[oy:oy + m, ox:ox + m]
    return out


def _radial_average(img):
    n = img.shape[0]
    c = (n - 1) / 2
    yy, xx = np.indices(img.shape)
    r = np.hypot(xx - c, yy - c)
    rb = np.round(r * 2).astype(int)                         # half-pixel radial bins
    sums = np.bincount(rb.ravel(), img.ravel())
    cnt = np.bincount(rb.ravel())
    prof = sums / np.maximum(cnt, 1)
    return prof[rb]


def psf_on_cube(cube: Cube, lam_obs_um: float, model: str = "webbpsf", oversample: int = 4,
                fov_arcsec: Optional[float] = None, symmetric: bool = True, fwhm_scale: float = 1.0) -> np.ndarray:
    """Fraction of a point source at the cube's AGN position that falls in each spaxel, at one wavelength.

    The model PSF is computed on a grid ``oversample`` x finer than the cube, shifted to the (sub-pixel) AGN position
    and integrated over each spaxel.
    """
    if cube.agn_xy is None:
        raise ValueError(f"cube '{cube.name}' has no AGN position")
    ny, nx = cube.shape2d
    pix = cube.pixel_scale
    x0, y0 = cube.agn_xy
    if fov_arcsec is None:
        fov_arcsec = 2 * pix * max(nx, ny) + 2.0
    pso = pix / oversample
    n = int(np.ceil(fov_arcsec / pso)) | 1                     # odd
    if model == "gaussian":
        img = _gaussian_image(float(psf_fwhm_arcsec(lam_obs_um)) * fwhm_scale, pso, n)
    elif model == "webbpsf":
        img = _webbpsf_image(cube.name, lam_obs_um, pso, n, oversample)
    else:
        raise ValueError("model must be 'webbpsf' or 'gaussian'")
    if symmetric:
        img = _radial_average(img)
    c = (n - 1) / 2
    sub = (np.arange(oversample) + 0.5) / oversample - 0.5
    yy, xx = np.mgrid[0:ny, 0:nx]
    X = (xx[..., None, None] + sub[None, None, None, :] - x0) * oversample + c
    Y = (yy[..., None, None] + sub[None, None, :, None] - y0) * oversample + c
    X, Y = np.broadcast_arrays(X, Y)
    vals = map_coordinates(img, [Y.ravel(), X.ravel()], order=1, mode="constant", cval=0.0)
    return vals.reshape(ny, nx, oversample, oversample).sum(axis=(2, 3))


def _fit_point_source(cube, lam_s, Ps, r_fit):
    """Per channel, fit the spaxels within r_fit of the AGN with  S x PSF + B0 + B1 r  (point source + a smooth,
    centrally sloped extended component; linear least squares weighted by the ERR cube). PSFs between the sample
    wavelengths are interpolated linearly. Returns (S, err_S) in Jy (total point-source flux)."""
    from .extraction import cube_footprint
    ny, nx = cube.shape2d
    x0, y0 = cube.agn_xy
    yy, xx = np.indices((ny, nx))
    r = np.hypot(xx - x0, yy - y0) * cube.pixel_scale
    sel = (r <= r_fit) & cube_footprint(cube)
    rs = r[sel]
    Psel = np.array([P[sel] for P in Ps])
    data = np.asarray(cube.data, float)
    err = np.asarray(cube.err, float) if cube.err is not None else None
    n = len(cube.wave)
    S, eS = np.full(n, np.nan), np.full(n, np.nan)
    fidx = np.interp(cube.wave, lam_s, np.arange(len(lam_s)))
    for l in range(n):
        y = data[l][sel]
        ok = np.isfinite(y) & (y != 0)
        if ok.sum() < 8:
            continue
        i0 = min(int(np.floor(fidx[l])), len(lam_s) - 1); i1 = min(i0 + 1, len(lam_s) - 1); t = fidx[l] - i0
        p = (1 - t) * Psel[i0] + t * Psel[i1]
        A = np.vstack([p, np.ones_like(rs), rs]).T[ok]
        w = 1.0 / err[l][sel][ok] if err is not None else np.ones(ok.sum())
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        Aw, yw = A * w[:, None], y[ok] * w
        coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        cov = np.linalg.pinv(Aw.T @ Aw)
        chi2 = np.sum((Aw @ coef - yw) ** 2) / max(ok.sum() - 3, 1)
        S[l], eS[l] = coef[0], np.sqrt(max(cov[0, 0], 0) * max(chi2, 1.0))
    return S, eS


def _region_weights(sp: ComponentSpectra, component: str) -> np.ndarray:
    mask = sp.agn_mask if component == "agn" else sp.sf_mask
    if mask is not None:
        return mask.astype(float)
    ok = np.isfinite(sp.f_disc_map) & (sp.footprint if sp.footprint is not None else True)
    fd = np.where(ok, sp.f_disc_map, 0.0)
    return np.where(ok, 1.0 - fd if component == "agn" else fd, 0.0)


def psf_correct(cubes: Dict[str, Cube], spectra: Dict[str, ComponentSpectra],
                aperture_spectra: Dict[str, ApertureSpectrum], model: str = "webbpsf", n_wave: int = 6,
                oversample: int = 4, symmetric: bool = True, fwhm_scale: float = 1.0, psf_sys: float = 0.1,
                annulus_arcsec=None, point_source: str = "aperture", fit_radius_arcsec: Optional[float] = None,
                verbose: bool = True):
    """PSF-corrected AGN (point source), AGN-outflow and SF spectra, per band (see module docstring).

    spectra : the region spectra of Step 5 (``extract_all``); aperture_spectra : ``{band: ApertureSpectrum}`` of the
    nuclear aperture (``extract_apertures(...)["nuclear"]``).
    n_wave : PSF sample wavelengths per band (the fractions are interpolated linearly in between).
    annulus_arcsec : ``None`` - everything in the aperture is the point source (simplest; extended emission inside
        the aperture is then over-corrected by 1/EE). ``(r_in, r_out)`` - the extended emission is assumed to have the
        same mean surface brightness in the aperture as in this annulus around the AGN, and the point source S and the
        extended emission are solved for together: S = (F_ap - a F_ann) / (EE - a f_ann), a = area_ap / area_ann.

    point_source : how the AGN point-source flux S is measured.
        ``"aperture"`` - from the aperture flux, S = F_ap / EE (optionally with ``annulus_arcsec``): assumes all
        light in the aperture is the point source.
        ``"fit"`` - per channel, the spaxels within ``fit_radius_arcsec`` (default max(0.6", 2 FWHM)) are fitted with
        S x PSF + a smooth extended component (B0 + B1 r): separates the sharp PSF core from extended emission
        (recommended when the curve of growth shows extended emission around the nucleus). The region errors then
        neglect the (small) covariance between the fitted S and the region pixels.

    Returns ``(result, fractions)``: ``result = {"AGN" | "AGN-outflow" | "SF": {band: ApertureSpectrum}}`` (stitch with
    :func:`breaking_the_blend.stitch_spectra`, component ``"flux"``), and ``fractions = {band: dict(wave, ee, f_agn,
    f_sf, f_ann)}`` at the sample wavelengths.
    """
    from .extraction import aperture_weights, cube_footprint
    result = {"AGN": {}, "AGN-outflow": {}, "SF": {}}
    fractions = {}
    for band, sp in sorted(spectra.items(), key=lambda kv: cubes[kv[0]].wave_range[0] if kv[0] in cubes else 0):
        ap = aperture_spectra.get(band)
        cube = cubes.get(band)
        if ap is None or cube is None or not np.isfinite(ap.flux).any():
            continue
        w_ap = ap.weights
        if annulus_arcsec is not None:
            r_in, r_out = annulus_arcsec
            x0, y0 = ap.center_xy
            w_ann = (aperture_weights(cube.shape2d, x0, y0, r_out / cube.pixel_scale)
                     - aperture_weights(cube.shape2d, x0, y0, r_in / cube.pixel_scale)) * cube_footprint(cube)
            w_ann = np.clip(w_ann, 0, None)
            a_ratio = w_ap.sum() / w_ann.sum() if w_ann.sum() > 0 else 0.0
        else:
            w_ann, a_ratio = np.zeros_like(w_ap), 0.0
        w_r = {"AGN-outflow": _region_weights(sp, "agn"), "SF": _region_weights(sp, "sf")}
        lam_s = np.linspace(cube.wave.min(), cube.wave.max(), n_wave)
        z = getattr(cube, "z", 0.0) or 0.0
        ee, fa, fs, fn, Ps = [], [], [], [], []
        for lam in lam_s:
            P = psf_on_cube(cube, lam * (1 + z), model, oversample, symmetric=symmetric, fwhm_scale=fwhm_scale)
            Ps.append(P)
            ee.append(float((P * w_ap).sum())); fa.append(float((P * w_r["AGN-outflow"]).sum()))
            fs.append(float((P * w_r["SF"]).sum())); fn.append(float((P * w_ann).sum()))
        ee, fa, fs, fn = map(np.array, (ee, fa, fs, fn))
        fractions[band] = dict(wave=lam_s, ee=ee, f_agn=fa, f_sf=fs, f_ann=fn)
        EE, FN = np.interp(cube.wave, lam_s, ee), np.interp(cube.wave, lam_s, fn)
        FR = {"AGN-outflow": np.interp(cube.wave, lam_s, fa), "SF": np.interp(cube.wave, lam_s, fs)}

        # point source S = alpha * F_ap - beta * F_ann  (beta = 0 without annulus)
        alpha = 1.0 / (EE - a_ratio * FN)
        beta = a_ratio * alpha
        data = np.asarray(cube.data, float)
        have_err = cube.err is not None
        e2 = np.asarray(cube.err, float) ** 2 if have_err else None
        F_ap = _sum_with_gaps(data, w_ap)
        F_ann = _sum_with_gaps(data, w_ann) if annulus_arcsec is not None else np.zeros_like(F_ap)
        q = lambda wa, wb: np.einsum("lij,ij->l", e2, wa * wb)            # sum sigma^2 wa wb
        if point_source == "fit":
            r_fit = fit_radius_arcsec or max(0.6, 2 * float(psf_fwhm_arcsec(cube.wave.max() * (1 + z))) * fwhm_scale)
            S, eS = _fit_point_source(cube, lam_s, Ps, r_fit)
        elif point_source == "aperture":
            S = alpha * F_ap - beta * F_ann
            if have_err:
                vS = alpha ** 2 * q(w_ap, w_ap) + beta ** 2 * q(w_ann, w_ann) - 2 * alpha * beta * q(w_ap, w_ann)
                eS = np.sqrt(np.clip(vS, 0, None))
            else:
                eS = empirical_errors(S)
        else:
            raise ValueError("point_source must be 'aperture' or 'fit'")
        e_agn = np.hypot(eS, psf_sys * np.abs(S - F_ap))
        result["AGN"][band] = ApertureSpectrum("AGN", band, cube.wave.copy(), S, e_agn, w_ap, ap.center_xy,
                                               ap.radius_arcsec, cube.pixel_scale, ap.covered_frac, "psf")
        # regions with the point source removed: F_R - f_R S  ->  weights w_R - f_R alpha w_ap + f_R beta w_ann
        for name, w in w_r.items():
            if w.sum() == 0:
                continue
            c1, c2 = FR[name] * alpha, FR[name] * beta
            F = _sum_with_gaps(data, w) - FR[name] * S
            if point_source == "fit":
                eR = np.sqrt(q(w, w)) if have_err else empirical_errors(F)
                e = np.hypot(np.hypot(eR, FR[name] * eS), psf_sys * np.abs(FR[name] * S))
                result[name][band] = ApertureSpectrum(name, band, cube.wave.copy(), F, e, w, ap.center_xy,
                                                      ap.radius_arcsec, cube.pixel_scale, 1.0, "psf")
                continue
            if have_err:
                var = (q(w, w) + c1 ** 2 * q(w_ap, w_ap) + c2 ** 2 * q(w_ann, w_ann) - 2 * c1 * q(w, w_ap)
                       + 2 * c2 * q(w, w_ann) - 2 * c1 * c2 * q(w_ap, w_ann))
                e_stat = np.sqrt(np.clip(var, 0, None))
            else:
                e_stat = empirical_errors(F)
            e = np.hypot(e_stat, psf_sys * np.abs(FR[name] * S))
            result[name][band] = ApertureSpectrum(name, band, cube.wave.copy(), F, e, w, ap.center_xy,
                                                  ap.radius_arcsec, cube.pixel_scale, 1.0, "psf")
        if verbose:
            print(f"{band:>14s}: EE(aperture) {ee.min():.2f}-{ee.max():.2f}, leakage into AGN region "
                  f"{100 * fa.min():.1f}-{100 * fa.max():.1f}%, into SF region {100 * fs.min():.2f}-{100 * fs.max():.2f}% "
                  f"of the point-source flux")
    if verbose and model == "gaussian":
        print("(Gaussian PSF model - use model='webbpsf' with STPSF installed for the instrument PSF)")
    return result, fractions


# ----------------------------------------------------------------------
# Checks: contamination maps and curves of growth
# ----------------------------------------------------------------------
def _pick_cube(cubes, lam_rest):
    """Cube covering lam_rest [rest um] with the wavelength furthest from its band edges."""
    best, margin = None, -1.0
    for name, c in cubes.items():
        lo, hi = c.wave.min(), c.wave.max()
        if lo < lam_rest < hi and min(lam_rest - lo, hi - lam_rest) > margin:
            best, margin = name, min(lam_rest - lo, hi - lam_rest)
    return best


def psf_contamination(cubes: Dict[str, Cube], psf_result, lam_rest: float, model: str = "webbpsf",
                      oversample: int = 4, symmetric: bool = True, fwhm_scale: float = 1.0, half_width: float = 0.05):
    """Spaxel maps of the AGN point source at one rest wavelength, on the native grid of the cube covering it.

    Returns dict(band, lam, psf_pct = % of the AGN point-source flux in each spaxel, contam_pct = % of each spaxel's
    observed flux that is AGN PSF light (point-source flux x PSF / data), agn_flux = point-source flux [Jy],
    data = observed flux map [Jy / spaxel]); the fluxes are medians over +/- half_width um (lines included, so pick
    continuum wavelengths)."""
    band = _pick_cube(cubes, lam_rest)
    if band is None or band not in psf_result["AGN"]:
        return None
    cube = cubes[band]
    z = getattr(cube, "z", 0.0) or 0.0
    P = psf_on_cube(cube, lam_rest * (1 + z), model, oversample, symmetric=symmetric, fwhm_scale=fwhm_scale)
    sel = np.abs(cube.wave - lam_rest) <= half_width
    agn = psf_result["AGN"][band]
    S = float(np.nanmedian(agn.flux[sel]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = np.nanmedian(np.where(np.asarray(cube.data[sel]) != 0, np.asarray(cube.data[sel], float), np.nan), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        contam = np.where(data > 0, 100.0 * S * P / data, np.nan)
    return dict(band=band, lam=lam_rest, psf_pct=100.0 * P, contam_pct=contam, agn_flux=S, data=data,
                agn_xy=cube.agn_xy, pixel_scale=cube.pixel_scale)


def growth_curve(cubes: Dict[str, Cube], lam_rest: float, radii_arcsec=None, model: str = "webbpsf",
                 oversample: int = 4, symmetric: bool = True, fwhm_scale: float = 1.0, half_width: float = 0.05):
    """Curve of growth of the observed continuum around the AGN vs that of the model PSF, at one wavelength.

    Both are normalised at the first radius. If the data grow faster than the PSF, there is extended emission
    around the nucleus: the point-source correction then over-estimates the AGN (and over-subtracts the regions).
    Returns dict(band, radii, data, psf) or None."""
    from .extraction import aperture_weights, cube_footprint
    band = _pick_cube(cubes, lam_rest)
    if band is None:
        return None
    cube = cubes[band]
    radii = np.asarray(radii_arcsec if radii_arcsec is not None else [0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5], float)
    z = getattr(cube, "z", 0.0) or 0.0
    P = psf_on_cube(cube, lam_rest * (1 + z), model, oversample, symmetric=symmetric, fwhm_scale=fwhm_scale)
    sel = np.abs(cube.wave - lam_rest) <= half_width
    img = np.nanmedian(np.asarray(cube.data[sel], float), axis=0)
    foot = cube_footprint(cube)
    x0, y0 = cube.agn_xy
    d, m = [], []
    for r in radii:
        w = aperture_weights(cube.shape2d, x0, y0, r / cube.pixel_scale) * foot
        d.append(np.nansum(w * img)); m.append(np.sum(w * P))
    d, m = np.array(d), np.array(m)
    return dict(band=band, lam=lam_rest, radii=radii, data=d / d[0], psf=m / m[0])

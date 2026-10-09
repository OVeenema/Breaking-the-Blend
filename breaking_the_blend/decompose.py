"""
STEP 3 - "Breaking the blend": constrained two-component decomposition per spaxel
(paper Sect. 3.3).

In every good spaxel the continuum-subtracted line profile is fitted with two
Gaussians whose

* centroids are FIXED to the disc and outflow model velocities in that spaxel,
* dispersions are FIXED to the disc / outflow dispersions of the line,

so only the two (non-negative) amplitudes are free - a linear problem solved
exactly with non-negative least squares. The disc fraction is

    f_disc = A_disc*sigma_disc / (A_disc*sigma_disc + A_out*sigma_out)     (paper eq. 12)

Options
-------
* ``external`` : use velocity / dispersion maps from other codes (3DBarolo, MOKA3D...)
  instead of the built-in models (see :mod:`breaking_the_blend.external`).
* ``recut`` : re-extract the line from the cube with a different window / continuum
  for the decomposition than for Step 1 (the original NGC 5728 analysis used a
  wider window with a linear continuum here).
* ``sigma_mode="free"`` : let the dispersions vary per spaxel (robustness check).
* ``sigma_fraction`` : use different dispersions in eq. 12 than in the fit
  (only to reproduce the original notebooks; normally leave as ``None``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import nnls, least_squares

from .lines import C_KMS
from .linefit import LineMaps, subtract_continuum
from .models import PA_OFFSET_DEG, component_velocities
from .utils import gaussian, progress

EXTERNAL_KEYS = ("v_disc", "v_out", "sigma_disc", "sigma_out")


@dataclass
class Decomposition:
    """Result of :func:`decompose_line` (all maps on the line's own pixel grid)."""

    line_name: str
    label: str
    ip: float
    cube_name: str
    pixel_scale_pc: float
    agn_xy: Tuple[float, float]
    good: np.ndarray
    params: Dict[str, float]
    sigma_disc: float
    sigma_out: float
    v_disc: np.ndarray          # velocity maps used as fixed centroids
    v_out: np.ndarray
    A_disc: np.ndarray          # fitted amplitudes [Jy]
    A_out: np.ndarray
    S_disc: np.ndarray          # dispersion used per spaxel
    S_out: np.ndarray
    f_disc: np.ndarray          # disc flux fraction
    residual_frac: np.ndarray   # sum(residual)/sum(data) over the line region
    residual_v: np.ndarray      # flux-weighted velocity of the residual
    flux_disc: np.ndarray = field(default=None, repr=False)   # line flux split [erg/s/cm2]
    flux_out: np.ndarray = field(default=None, repr=False)
    vel_axis: np.ndarray = field(default=None, repr=False)    # spectra used in the fit
    contsub: np.ndarray = field(default=None, repr=False)
    external: Tuple[str, ...] = ()

    @property
    def f_out(self) -> np.ndarray:
        return 1.0 - self.f_disc

    @property
    def mean_f_disc(self) -> float:
        return float(np.nanmean(self.f_disc))

    def offsets_pc(self):
        ny, nx = self.f_disc.shape
        yy, xx = np.indices((ny, nx), dtype=float)
        s = self.pixel_scale_pc
        return (xx - self.agn_xy[0]) * s, (yy - self.agn_xy[1]) * s

    def extent_pc(self):
        ny, nx = self.f_disc.shape
        s = self.pixel_scale_pc
        x0, y0 = self.agn_xy
        return [(-0.5 - x0) * s, (nx - 0.5 - x0) * s, (-0.5 - y0) * s, (ny - 0.5 - y0) * s]

    def __repr__(self):
        ext = f", external={list(self.external)}" if self.external else ""
        return (f"Decomposition({self.label}, IP={self.ip}, sigma_disc={self.sigma_disc:.0f}, "
                f"sigma_out={self.sigma_out:.0f}, <f_disc>={self.mean_f_disc:.2f}{ext})")


def line_params(line_maps: Dict[str, LineMaps], base: Dict[str, float],
                fits: Optional[Dict[str, object]] = None,
                ip_relations: Optional[Dict[str, Callable[[float], float]]] = None,
                per_line: Optional[Dict[str, Dict[str, float]]] = None,
                verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """Build the full model-parameter dict to use for each line in the decomposition.

    Priority (later overrides earlier):

    1. ``base`` - values common to all lines (e.g. the geometry);
    2. ``ip_relations`` - ``{param: f(IP)}``, e.g. ``{"R_out": lambda ip: 3.6*ip + 535}``;
    3. ``fits`` - per-line :class:`KinematicFit` results (their ``.params``);
    4. ``per_line`` - ``{line: {param: value}}`` set by hand.
    """
    out = {}
    for name, lm in line_maps.items():
        p = dict(base)
        for k, f in (ip_relations or {}).items():
            p[k] = float(f(lm.ip))
        if fits and name in fits:
            p.update(fits[name].params)
        if per_line and name in per_line:
            p.update(per_line[name])
        priors = [k for k, v in p.items() if isinstance(v, (tuple, list))]
        if priors:
            raise ValueError(f"{name}: {priors} are (lo, hi) prior ranges, not values. The decomposition needs "
                             "numbers - take them from the fits, e.g. base={**disc_fit.params, **out_fit.params}")
        out[name] = p
        if verbose:
            show = ", ".join(f"{k}={p[k]:.1f}" for k in ("v_disc", "R_disc", "v_out", "R_out",
                                                          "theta_in", "theta_out") if k in p)
            print(f"{lm.label:>9s}: {show}")
    return out


def _external_on(lm, ext, key):
    if ext is None or key not in ext or ext[key] is None:
        return None
    m = ext[key]
    if hasattr(m, "on_grid"):
        return m.on_grid(lm)
    m = np.asarray(m, float)
    if m.shape != lm.shape:
        raise ValueError(f"external '{key}' array has shape {m.shape}, line grid is {lm.shape}; "
                         "use load_external_map() so it can be re-projected")
    return m


def decompose_line(lm: LineMaps, params: Dict[str, float], sigma_disc: float, sigma_out: float,
                   sigma_mode: str = "fixed", sigma_free_range: Tuple[float, float] = (0.5, 2.0),
                   external: Optional[dict] = None, sigma_fraction: Optional[Tuple[float, float]] = None,
                   cube=None, recut: Optional[dict] = None,
                   pa_offset: float = PA_OFFSET_DEG, outside_cone: float = 0.0,
                   show_progress: bool = True) -> Decomposition:
    """Constrained decomposition of one line into disc and outflow (see module docstring).

    Parameters
    ----------
    lm : :class:`LineMaps` from Step 1 (gives the spaxel mask and, by default, the spectra).
    params : full model parameters (disc + outflow) for this line, e.g. from :func:`line_params`.
    sigma_disc, sigma_out : dispersions [km/s] of the two components.
    sigma_mode : ``"fixed"`` (paper) or ``"free"`` (vary within ``sigma_free_range`` x the values).
    external : ``{"v_disc"|"v_out"|"sigma_disc"|"sigma_out": ExternalMap or array}`` overriding
        the model velocities / fixed dispersions (NaN spaxels fall back to the defaults).
    sigma_fraction : (sigma_disc, sigma_out) used only in eq. 12 (default: the fit values).
    cube, recut : re-extract the spectra from ``cube`` with
        ``recut = dict(window=(lo, hi), line_region=(lo, hi), cont_order=1)`` [rest micron].
    outside_cone : outflow model velocity outside the bicone (0 = systemic, as in the paper).
    """
    x, y = lm.offsets_pc()
    vd_map, vo_map = component_velocities(x, y, params, pa_offset, outside=outside_cone)
    used_ext = []
    m = _external_on(lm, external, "v_disc")
    if m is not None:
        used_ext.append("v_disc")
        vd_map = np.where(np.isfinite(m), m, vd_map)
    m = _external_on(lm, external, "v_out")
    if m is not None:
        used_ext.append("v_out")
        vo_map = np.where(np.isfinite(m), m, vo_map)
    sd_map = np.full(lm.shape, float(sigma_disc))
    so_map = np.full(lm.shape, float(sigma_out))
    for key, arr in (("sigma_disc", sd_map), ("sigma_out", so_map)):
        m = _external_on(lm, external, key)
        if m is not None:
            used_ext.append(key)
            ok = np.isfinite(m) & (m > 0)
            arr[ok] = m[ok]

    # spectra to decompose
    if recut is not None:
        if cube is None:
            raise ValueError("recut needs the cube")
        lo, hi = recut["window"]
        sel = (cube.wave >= lo) & (cube.wave <= hi)
        wave = cube.wave[sel]
        lr = recut["line_region"]
        spec = subtract_continuum(wave, np.asarray(cube.data[sel], float), lr, recut.get("cont_order", 1))
        in_line = (wave >= lr[0]) & (wave <= lr[1])
        v = C_KMS * (wave - lm.line.wave) / lm.line.wave
    else:
        spec, v, in_line = lm.contsub, lm.vel_axis, lm.in_line

    ny, nx = lm.shape
    nanmap = lambda: np.full((ny, nx), np.nan)
    A_d, A_o, S_d, S_o, res_f, res_v = (nanmap() for _ in range(6))

    for (i, j) in progress(np.argwhere(lm.good), desc=f"{lm.name:>6s}", disable=not show_progress):
        yspec = spec[:, i, j]
        vd, vo, s_d0, s_o0 = vd_map[i, j], vo_map[i, j], sd_map[i, j], so_map[i, j]
        if not (np.isfinite(vd) and np.isfinite(vo)):
            continue
        if sigma_mode == "fixed":
            basis = np.vstack([gaussian(v, 1.0, vd, s_d0), gaussian(v, 1.0, vo, s_o0)]).T
            (ad, ao), _ = nnls(basis, yspec)
            sd, so = s_d0, s_o0
        elif sigma_mode == "free":
            scale = max(np.max(yspec[in_line]), 1e-30)
            f = lambda p: gaussian(v, p[0], vd, p[2]) + gaussian(v, p[1], vo, p[3]) - yspec / scale
            lo_ = [0, 0, s_d0 * sigma_free_range[0], s_o0 * sigma_free_range[0]]
            hi_ = [5, 5, s_d0 * sigma_free_range[1], s_o0 * sigma_free_range[1]]
            r = least_squares(f, [0.5, 0.5, s_d0, s_o0], bounds=(lo_, hi_))
            ad, ao, sd, so = r.x[0] * scale, r.x[1] * scale, r.x[2], r.x[3]
        else:
            raise ValueError("sigma_mode must be 'fixed' or 'free'")

        A_d[i, j], A_o[i, j], S_d[i, j], S_o[i, j] = ad, ao, sd, so
        resid = (yspec - gaussian(v, ad, vd, sd) - gaussian(v, ao, vo, so))[in_line]
        tot = np.sum(yspec[in_line])
        if tot != 0:
            res_f[i, j] = np.sum(resid) / tot
        if np.sum(resid) != 0:
            res_v[i, j] = np.sum(resid * v[in_line]) / np.sum(resid)

    if sigma_fraction is not None:
        fd, fo = A_d * sigma_fraction[0], A_o * sigma_fraction[1]
    else:
        fd, fo = A_d * S_d, A_o * S_o
    with np.errstate(invalid="ignore", divide="ignore"):
        f_disc = np.where(fd + fo > 0, fd / (fd + fo), np.nan)

    return Decomposition(line_name=lm.name, label=lm.label, ip=lm.ip, cube_name=lm.cube_name,
                         pixel_scale_pc=lm.pixel_scale_pc, agn_xy=lm.agn_xy, good=lm.good,
                         params=dict(params), sigma_disc=float(sigma_disc), sigma_out=float(sigma_out),
                         v_disc=np.where(lm.good, vd_map, np.nan), v_out=np.where(lm.good, vo_map, np.nan),
                         A_disc=A_d, A_out=A_o, S_disc=S_d, S_out=S_o, f_disc=f_disc,
                         residual_frac=res_f, residual_v=res_v,
                         flux_disc=lm.flux * f_disc, flux_out=lm.flux * (1 - f_disc),
                         vel_axis=v, contsub=spec, external=tuple(used_ext))


def _external_for(external, name):
    """``external`` given once for all lines ({key: map}) or per line ({line: {key: map}})."""
    if not external:
        return None
    if any(k in EXTERNAL_KEYS for k in external):
        return external
    return external.get(name)


def decompose_lines(line_maps: Dict[str, LineMaps], params: Dict[str, Dict[str, float]],
                    sigmas: Dict[str, Dict[str, float]], names=None, cubes=None,
                    recut: Optional[dict] = None, external: Optional[dict] = None,
                    sigma_fraction: Optional[dict] = None, **kwargs) -> Dict[str, Decomposition]:
    """Run :func:`decompose_line` for several lines.

    params : ``{line: param_dict}`` (from :func:`line_params`).
    sigmas : ``{line: {"disc": s_d, "outflow": s_o}}`` (from :func:`dispersion_medians`).
    names : subset of lines to decompose (default: all in ``params``).
    cubes, recut : ``recut = {line: dict(window=..., line_region=..., cont_order=...)}``.
    external : maps for all lines (``{"v_disc": map, ...}``) or per line (``{line: {...}}``).
    sigma_fraction : ``{line: (s_d, s_o)}`` used only in the fraction (see :func:`decompose_line`).
    """
    names = names or list(params)
    out = {}
    for n in names:
        rc = (recut or {}).get(n)
        if rc and cubes is None:
            raise ValueError("recut needs cubes=...")
        out[n] = decompose_line(line_maps[n], params[n], sigmas[n]["disc"], sigmas[n]["outflow"],
                                external=_external_for(external, n),
                                sigma_fraction=(sigma_fraction or {}).get(n),
                                cube=cubes[line_maps[n].cube_name] if rc else None,
                                recut=rc, **kwargs)
        print(f"  -> {out[n]}")
    return out

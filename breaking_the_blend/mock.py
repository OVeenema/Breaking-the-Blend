"""
Synthetic JWST IFS cubes of a galaxy with a rotating star-forming ring and an
AGN-driven bicone - used for the tests and the hands-on quick-start notebook,
so the whole method can be run (and checked against the known truth) without data.

How the mock galaxy is built (``make_mock_cubes``)
---------------------------------------------------
1. **Bands.** Each cube covers a real instrument band (default: the 12 MIRI/MRS
   sub-bands, :data:`MRS_BANDS`, with their observed wavelength ranges, pixel
   sizes and resolving powers). Wavelengths are generated in the observed frame
   and divided by (1 + z), exactly as for real data.
2. **Geometry.** Every cube covers the same field of view (``fov_arcsec``) on its own
   pixel grid, with the AGN near the centre.
3. **Two gas components** with surface brightness maps:

   * a star-forming **ring** in an inclined thin disc (Gaussian ring + small nuclear
     part), rotating with the thin-disc model (tanh rotation curve);
   * a hollow **bicone** (exponential fall-off with radius + a compact nucleus),
     moving with the Das et al. bicone model.

   Their velocity fields use :data:`TRUE_PARAMS` and the package's own models.
4. **Lines.** Every catalogue line inside a band is added as the sum of a disc
   Gaussian (sigma_disc) and an outflow Gaussian (sigma_out). The disc share of a
   line falls with its ionisation potential (:func:`disc_fraction_of_ip`), like in real AGN.
5. **Continuum + noise.** A rising AGN point-source continuum (PSF growing with
   wavelength) plus a faint disc continuum, and Gaussian noise.

The returned ``truth`` dict holds the true parameters and, for every line, the
true disc-fraction map on its cube's grid.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .cubes import Cube, pc_per_arcsec
from .lines import C_KMS, LineSpec, get_lines
from .models import component_velocities, bicone_mask

# observed-frame wavelength range [micron], pixel size [arcsec], resolving power
MRS_BANDS: Dict[str, Tuple[float, float, float, float]] = {
    "ch1-short":  (4.90, 5.74, 0.13, 3500), "ch1-medium": (5.66, 6.63, 0.13, 3500),
    "ch1-long":   (6.53, 7.65, 0.13, 3300), "ch2-short":  (7.51, 8.77, 0.17, 3000),
    "ch2-medium": (8.67, 10.13, 0.17, 2900), "ch2-long":  (10.02, 11.70, 0.17, 2700),
    "ch3-short":  (11.55, 13.47, 0.20, 2600), "ch3-medium": (13.34, 15.57, 0.20, 2400),
    "ch3-long":   (15.41, 17.98, 0.20, 2200), "ch4-short":  (17.70, 20.95, 0.35, 1900),
    "ch4-medium": (20.69, 24.48, 0.35, 1700), "ch4-long":   (24.19, 27.90, 0.35, 1500),
}
NIRSPEC_BANDS: Dict[str, Tuple[float, float, float, float]] = {
    "g395h-f290lp": (2.87, 5.27, 0.10, 2700),
}

TRUE_PARAMS = dict(v_sys=0.0, v_disc=300.0, R_disc=250.0, inc_disc=45.0, pa_disc=200.0,
                   v_out=600.0, R_out=500.0, inc_out=50.0, pa_out=110.0,
                   theta_in=20.0, theta_out=60.0)


def disc_fraction_of_ip(ip: float) -> float:
    """Line-integrated weight of the disc for a line of a given IP (mock recipe)."""
    return float(np.clip(0.85 - 0.0105 * ip, 0.0, 1.0))


def surface_brightness(x, y, p, ring_radius=500.0, ring_width=150.0, out_scale=300.0):
    """Disc (ring) and outflow (bicone) surface-brightness maps, peak ~1-2."""
    phi = np.deg2rad(p["pa_disc"] + 90)
    inc = np.deg2rad(p["inc_disc"])
    xp = x * np.cos(phi) + y * np.sin(phi)
    yp = (-x * np.sin(phi) + y * np.cos(phi)) / np.cos(inc)
    R = np.hypot(xp, yp)
    disc = 2.0 * np.exp(-0.5 * ((R - ring_radius) / ring_width) ** 2) + 0.3 * np.exp(-R / 150.0)
    inside, _ = bicone_mask(x, y, p["inc_out"], p["pa_out"], p["theta_in"], p["theta_out"])
    r = np.hypot(x, y)
    out = np.where(inside, np.exp(-r / out_scale), 0.0) + np.exp(-0.5 * (r / 120.0) ** 2)
    return disc, out


def make_mock_cubes(bands: Optional[Dict[str, tuple]] = None, lines: Optional[Dict[str, LineSpec]] = None,
                    z: float = 0.0093, fov_arcsec: float = 7.0, sigma_disc: float = 70.0,
                    sigma_out: float = 200.0, noise: float = 0.01, params: Optional[dict] = None,
                    agn_offset=(0.3, -0.2), seed: int = 0):
    """Build mock cubes (see module docstring).

    bands : ``{name: (lo_obs, hi_obs, pixel_scale_arcsec, resolving_power)}`` in the
        OBSERVED frame [micron]; default :data:`MRS_BANDS`.
    lines : lines to include (default: whole catalogue).
    noise : noise level relative to the line peak scale (1e-3 Jy).
    Returns ``(cubes, truth)``.
    """
    rng = np.random.default_rng(seed)
    p = {**TRUE_PARAMS, **(params or {})}
    bands = bands or MRS_BANDS
    lines = lines or get_lines()
    scale = pc_per_arcsec(z)
    cubes, truth = {}, {"params": p, "sigma_disc": sigma_disc, "sigma_out": sigma_out,
                        "agn_xy": {}, "f_disc": {}, "z": z}
    for name, b in bands.items():
        lo_obs, hi_obs, pix, R = b
        npix = int(round(fov_arcsec / pix)) | 1
        c0 = (npix - 1) / 2
        agn_xy = (c0 + agn_offset[0], c0 + agn_offset[1])
        yy, xx = np.indices((npix, npix), dtype=float)
        spx = pix * scale
        x, y = (xx - agn_xy[0]) * spx, (yy - agn_xy[1]) * spx
        sb_d, sb_o = surface_brightness(x, y, p)
        vd, vo = component_velocities(x, y, p)

        wave_obs = np.arange(lo_obs, hi_obs, np.mean([lo_obs, hi_obs]) / R / 2)
        wave = wave_obs / (1 + z)                                 # rest frame
        psf_sig = max(0.6, 0.014 * wave.mean() / pix)            # PSF grows with wavelength [pix]
        psf = np.exp(-0.5 * ((xx - agn_xy[0]) ** 2 + (yy - agn_xy[1]) ** 2) / psf_sig ** 2)
        data = ((wave / 10.0) ** 2.0)[:, None, None] * 3e-3 * psf[None] \
            + ((wave / 10.0) ** 1.0)[:, None, None] * 1.5e-4 * sb_d[None]
        for ln in lines.values():
            if not (wave[0] + 0.03 < ln.wave < wave[-1] - 0.03):
                continue
            fd = disc_fraction_of_ip(ln.ip)
            amp_d, amp_o = 1e-3 * fd * sb_d, 1e-3 * (1 - fd) * sb_o
            v = C_KMS * (wave - ln.wave) / ln.wave
            ch = np.abs(v) < 4000                                 # only channels near the line
            gd = np.exp(-0.5 * ((v[ch, None, None] - vd[None]) / sigma_disc) ** 2)
            go = np.exp(-0.5 * ((v[ch, None, None] - vo[None]) / sigma_out) ** 2)
            data[ch] += amp_d[None] * gd + amp_o[None] * go
            with np.errstate(invalid="ignore"):
                truth["f_disc"][ln.name] = amp_d * sigma_disc / (amp_d * sigma_disc + amp_o * sigma_out)
        data += rng.normal(0, noise * 1e-3, data.shape)
        cubes[name] = Cube(name=name, data=data.astype(np.float32), wave=wave, pixel_scale=pix,
                           pc_per_arcsec=scale, z=z,
                           err=np.full(data.shape, noise * 1e-3, dtype=np.float32))
        truth["agn_xy"][name] = agn_xy
    return cubes, truth

"""
Loading JWST NIRSpec / MIRI-MRS cubes and basic spatial bookkeeping.

A :class:`Cube` holds one IFS cube in Jy/pixel with a rest-frame wavelength
axis, its pixel scale and the AGN pixel position. All later steps work with
offsets from the AGN in parsec, so cubes with different pixel scales
(NIRSpec 0.1", MIRI ch1-4 0.13-0.35") can be combined.

Orientation convention
----------------------
Maps are stored exactly as in the cube (array ``[y, x]``) and plotted with
``origin="lower"``. For pipeline ``skyalign`` cubes this is north up, east
left, so +x points west. Position angles (PA, east of north) are converted
to the mathematical angle from +x with ``angle = PA + pa_offset`` and
``pa_offset = 90`` (see :mod:`breaking_the_blend.models`).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Iterable

import numpy as np
from scipy.integrate import quad

from .lines import C_KMS, LineSpec

ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0


# ----------------------------------------------------------------------
# Distances
# ----------------------------------------------------------------------
def angular_diameter_distance_mpc(z: float, H0: float = 67.66, Om0: float = 0.30966) -> float:
    """Flat LCDM angular-diameter distance [Mpc] (defaults: Planck18)."""
    integrand = lambda zz: 1.0 / np.sqrt(Om0 * (1 + zz) ** 3 + (1 - Om0))
    dc = C_KMS / H0 * quad(integrand, 0.0, z)[0]
    return dc / (1.0 + z)


def pc_per_arcsec(z: Optional[float] = None, distance_mpc: Optional[float] = None) -> float:
    """Physical scale [pc/arcsec]. Give a distance to override the redshift-based value."""
    if distance_mpc is None:
        if z is None:
            raise ValueError("Give z or distance_mpc")
        distance_mpc = angular_diameter_distance_mpc(z)
    return distance_mpc * 1e6 / ARCSEC_PER_RAD


# ----------------------------------------------------------------------
# Cube container
# ----------------------------------------------------------------------
@dataclass
class Cube:
    """One IFS cube.

    Attributes
    ----------
    name : short name, e.g. ``"ch3-short"`` or ``"g395h-f290lp"``.
    data : (n_wave, ny, nx) flux density [Jy/pixel]. NaNs replaced by 0.
    wave : (n_wave,) rest-frame wavelength [micron].
    pixel_scale : arcsec/pixel.
    pc_per_arcsec : physical scale [pc/arcsec].
    err : optional uncertainty cube [Jy/pixel].
    agn_xy : (x, y) AGN pixel position (set with :func:`set_agn_positions`).
    """

    name: str
    data: np.ndarray
    wave: np.ndarray
    pixel_scale: float
    pc_per_arcsec: float
    err: Optional[np.ndarray] = None
    agn_xy: Optional[Tuple[float, float]] = None
    z: float = 0.0
    path: Optional[str] = None
    header: Optional[dict] = field(default=None, repr=False)

    @property
    def shape2d(self) -> Tuple[int, int]:
        return self.data.shape[1:]

    @property
    def pixel_scale_pc(self) -> float:
        return self.pixel_scale * self.pc_per_arcsec

    @property
    def wave_range(self) -> Tuple[float, float]:
        return float(np.nanmin(self.wave)), float(np.nanmax(self.wave))

    def covers(self, lo: float, hi: float) -> bool:
        wmin, wmax = self.wave_range
        return (lo >= wmin) and (hi <= wmax)

    def continuum_image(self, wave_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Median image over the full cube (or over ``wave_range`` in micron)."""
        if wave_range is None:
            sel = slice(None)
        else:
            sel = (self.wave >= wave_range[0]) & (self.wave <= wave_range[1])
        return np.nanmedian(self.data[sel], axis=0)

    def offsets_pc(self, agn_xy: Optional[Tuple[float, float]] = None):
        """2D arrays (x_pc, y_pc) of spaxel offsets from the AGN."""
        x0, y0 = agn_xy if agn_xy is not None else self.agn_xy
        ny, nx = self.shape2d
        yy, xx = np.indices((ny, nx), dtype=float)
        s = self.pixel_scale_pc
        return (xx - x0) * s, (yy - y0) * s

    def __repr__(self) -> str:
        w0, w1 = self.wave_range
        return (f"Cube('{self.name}', {w0:.3f}-{w1:.3f} um, shape={self.data.shape}, "
                f"{self.pixel_scale:.3f}\"/pix = {self.pixel_scale_pc:.1f} pc/pix, agn_xy={self.agn_xy})")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def guess_cube_name(path: str) -> str:
    """Short name from a JWST file name: ``ch3-short``, ``g395h-f290lp``, or the file stem."""
    base = os.path.basename(path).lower()
    m = re.search(r"ch(\d)[-_](short|medium|long)", base)
    if m:
        return f"ch{m.group(1)}-{m.group(2)}"
    m = re.search(r"(g\d{3}[hm])[-_](f\d{3}lp|clear)", base)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return base.split(".")[0]


def _wavelength_axis(hdul, hdr) -> np.ndarray:
    n = hdr["NAXIS3"]
    names = [h.name for h in hdul]
    if "WCS-TABLE" in names and "TAB" in str(hdr.get("CTYPE3", "")):
        return np.asarray(hdul["WCS-TABLE"].data["wavelength"]).ravel()[:n]
    return hdr["CRVAL3"] + (np.arange(n) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]


def load_cube(path: str, z: float, name: Optional[str] = None,
              distance_mpc: Optional[float] = None, ext: str = "SCI",
              load_err: bool = False) -> Cube:
    """Read a JWST ``s3d`` cube, convert MJy/sr -> Jy/pixel and de-redshift the wavelengths.

    Parameters
    ----------
    path : FITS file (``.fits`` or ``.fits.gz``).
    z : redshift (used to de-redshift and, if no distance is given, for the pc scale).
    name : cube name; default guessed from the file name.
    distance_mpc : optional distance [Mpc] for the physical scale.
    load_err : also read the ERR extension.
    """
    from astropy.io import fits  # imported here so the rest of the package works without astropy

    with fits.open(path) as hdul:
        hdr = hdul[ext].header
        data = np.asarray(hdul[ext].data, dtype=np.float32)
        wave_obs = _wavelength_axis(hdul, hdr)
        err = np.asarray(hdul["ERR"].data, dtype=np.float32) if load_err and "ERR" in hdul else None
        hdr_dict = dict(hdr)

    pix_deg = abs(hdr["CDELT2"])
    pixar_sr = hdr.get("PIXAR_SR", abs(hdr["CDELT1"]) * abs(hdr["CDELT2"]) * (np.pi / 180) ** 2)
    to_jy = pixar_sr * 1e6  # MJy/sr -> Jy/pixel
    data = np.nan_to_num(data * to_jy, nan=0.0)
    if err is not None:
        err = np.nan_to_num(err * to_jy, nan=0.0)

    return Cube(name=name or guess_cube_name(path), data=data, wave=wave_obs / (1.0 + z),
                pixel_scale=pix_deg * 3600.0, pc_per_arcsec=pc_per_arcsec(z, distance_mpc),
                err=err, z=z, path=path, header=hdr_dict)


def load_cubes(paths: Iterable[str] | Dict[str, str], z: float, **kwargs) -> Dict[str, Cube]:
    """Load several cubes. ``paths`` may be a list (names guessed) or a ``{name: path}`` dict."""
    if isinstance(paths, dict):
        items = paths.items()
    else:
        items = [(guess_cube_name(p), p) for p in paths]
    cubes = {}
    for name, p in items:
        cubes[name] = load_cube(p, z, name=name, **kwargs)
        print(cubes[name])
    return cubes


# ----------------------------------------------------------------------
# AGN position and line -> cube assignment
# ----------------------------------------------------------------------
def find_agn_position(image: np.ndarray, centroid_box: int = 1) -> Tuple[float, float]:
    """(x, y) of the peak of ``image``, refined by a flux-weighted centroid in a
    (2*box+1)^2 box (``centroid_box=0`` returns the integer peak)."""
    img = np.nan_to_num(image, nan=-np.inf)
    y0, x0 = np.unravel_index(np.argmax(img), img.shape)
    if centroid_box <= 0:
        return float(x0), float(y0)
    ys = slice(max(y0 - centroid_box, 0), y0 + centroid_box + 1)
    xs = slice(max(x0 - centroid_box, 0), x0 + centroid_box + 1)
    sub = np.clip(np.nan_to_num(image[ys, xs]), 0, None)
    if sub.sum() <= 0:
        return float(x0), float(y0)
    yy, xx = np.indices(sub.shape)
    return float((xx * sub).sum() / sub.sum() + xs.start), float((yy * sub).sum() / sub.sum() + ys.start)


def set_agn_positions(cubes: Dict[str, Cube], overrides: Optional[Dict[str, Tuple[float, float]]] = None,
                      wave_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                      centroid_box: int = 1, verbose: bool = True) -> Dict[str, Tuple[float, float]]:
    """Set ``cube.agn_xy`` for every cube from the continuum peak.

    ``overrides`` : ``{cube_name: (x, y)}`` for cubes where the peak is wrong.
    ``wave_ranges`` : ``{cube_name: (lo, hi)}`` to build the continuum image
    from a restricted wavelength range.
    """
    overrides = overrides or {}
    wave_ranges = wave_ranges or {}
    out = {}
    for name, cube in cubes.items():
        if name in overrides:
            cube.agn_xy = tuple(map(float, overrides[name]))
            pk = find_agn_position(cube.continuum_image(wave_ranges.get(name)), centroid_box)
            how = f"manual; continuum peak at ({pk[0]:.2f}, {pk[1]:.2f})"
        else:
            cube.agn_xy = find_agn_position(cube.continuum_image(wave_ranges.get(name)), centroid_box)
            how = "continuum peak"
        out[name] = cube.agn_xy
        if verbose:
            print(f"{name:>14s}: AGN at (x, y) = ({cube.agn_xy[0]:.2f}, {cube.agn_xy[1]:.2f})  [{how}]")
    return out


def assign_cubes(lines: Dict[str, LineSpec], cubes: Dict[str, Cube], verbose: bool = True,
                 drop_missing: bool = False) -> Dict[str, LineSpec]:
    """Fill ``line.cube`` for lines that have none: pick the cube that covers the
    full fit window with the line closest to the cube's centre in wavelength.

    drop_missing : skip (with a message) lines that no loaded cube covers, instead of raising.
    """
    out = {}
    for name, line in lines.items():
        if line.cube is not None:
            if line.cube not in cubes:
                raise KeyError(f"{name}: cube '{line.cube}' is not loaded ({list(cubes)})")
            out[name] = line
        else:
            lo, hi = line.get_fit_window()
            cands = [(abs(line.wave - np.mean(c.wave_range)) / np.ptp(c.wave_range), cn)
                     for cn, c in cubes.items() if c.covers(lo, hi)]
            if not cands:
                msg = f"{name} ({line.wave} um): no loaded cube covers {lo:.3f}-{hi:.3f} um"
                if drop_missing:
                    print(f"  skipped - {msg}")
                    continue
                raise ValueError(msg)
            out[name] = line.update(cube=min(cands)[1])
        if verbose:
            print(out[name])
    return out

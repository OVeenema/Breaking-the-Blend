"""
OPTIONAL - use velocity / dispersion maps from other modelling codes instead of
(or as well as) the built-in disc and bicone models.

Examples: a disc model velocity (and dispersion) field from **3DBarolo**, or an
outflow velocity field from **MOKA3D**. Each map is placed on the sky by giving the
AGN pixel position *in that map* and its pixel scale, and is then re-projected
onto the grid of every emission line (offsets from the AGN in parsec), exactly
like the built-in models.

    baro_v = btb.load_external_map("NGC5728_3dbarolo_vfield.fits", agn_xy=(41.2, 39.8),
                                   pixel_scale_arcsec=0.2, pc_per_arcsec=proj_pc_per_arcsec,
                                   subtract=2790.0)          # remove the systemic velocity
    decomps = btb.decompose_lines(..., external={"v_disc": baro_v})

Keys understood by the decomposition: ``"v_disc"``, ``"v_out"``, ``"sigma_disc"``,
``"sigma_out"``. Spaxels where an external map is undefined (NaN / outside it)
fall back to the built-in model velocity or to the fixed dispersion.

Orientation: maps are assumed north up, east left like the cubes. Use
``rotation_deg`` (counter-clockwise, applied to the external map) and/or
``flip_x`` if your map is stored differently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class ExternalMap:
    """A 2D map (km/s) that can be sampled at sky offsets from the AGN [pc]."""

    data: np.ndarray
    agn_xy: Tuple[float, float]
    pixel_scale_pc: float
    rotation_deg: float = 0.0
    flip_x: bool = False
    name: str = "external"

    def sample(self, x_pc, y_pc, method: str = "linear") -> np.ndarray:
        """Values at offsets (x_pc, y_pc) from the AGN (NaN outside the map)."""
        x, y = np.asarray(x_pc, float), np.asarray(y_pc, float)
        if self.rotation_deg:
            a = -np.deg2rad(self.rotation_deg)   # undo the map's rotation
            x, y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
        if self.flip_x:
            x = -x
        ny, nx = self.data.shape
        s = self.pixel_scale_pc
        xs = (np.arange(nx) - self.agn_xy[0]) * s
        ys = (np.arange(ny) - self.agn_xy[1]) * s
        f = RegularGridInterpolator((ys, xs), self.data, method=method, bounds_error=False, fill_value=np.nan)
        return f(np.column_stack([y.ravel(), x.ravel()])).reshape(x.shape)

    def on_grid(self, obj) -> np.ndarray:
        """Re-project onto the grid of a LineMaps / Decomposition / Cube (anything with ``offsets_pc()``)."""
        x, y = obj.offsets_pc()
        return self.sample(x, y)

    def __repr__(self):
        return (f"ExternalMap('{self.name}', shape={self.data.shape}, agn_xy={self.agn_xy}, "
                f"{self.pixel_scale_pc:.1f} pc/pix)")


def load_external_map(source: Union[str, np.ndarray], agn_xy: Tuple[float, float],
                      pixel_scale_arcsec: Optional[float] = None, pc_per_arcsec: Optional[float] = None,
                      pixel_scale_pc: Optional[float] = None, hdu: int = 0, plane: Optional[int] = None,
                      subtract: float = 0.0, scale: float = 1.0, rotation_deg: float = 0.0,
                      flip_x: bool = False, name: Optional[str] = None) -> ExternalMap:
    """Read a velocity or dispersion map (FITS file or 2D array).

    Parameters
    ----------
    source : FITS file name or a 2D numpy array.
    agn_xy : (x, y) pixel position of the AGN / kinematic centre in this map.
    pixel_scale_arcsec : pixel size [arcsec]; read from CDELT2 of a FITS file if not given.
    pc_per_arcsec : physical scale (e.g. ``cube.pc_per_arcsec``); or give ``pixel_scale_pc`` directly.
    hdu, plane : FITS extension, and plane index if the data are 3D.
    subtract, scale : the map is converted as ``(map - subtract) * scale``, e.g. ``subtract`` =
        systemic velocity if the map holds absolute velocities, ``scale=1e-3`` for m/s.
    rotation_deg, flip_x : orientation fixes (see module docstring).
    """
    if isinstance(source, str):
        from astropy.io import fits
        with fits.open(source) as h:
            data = np.asarray(h[hdu].data, dtype=float)
            hdr = h[hdu].header
        if pixel_scale_arcsec is None and pixel_scale_pc is None:
            pixel_scale_arcsec = abs(hdr["CDELT2"]) * 3600.0
        name = name or source.split("/")[-1]
    else:
        data = np.asarray(source, dtype=float)
        name = name or "array"
    data = np.squeeze(data)
    if data.ndim == 3:
        if plane is None:
            raise ValueError("3D map: give plane=...")
        data = data[plane]
    if pixel_scale_pc is None:
        if pixel_scale_arcsec is None or pc_per_arcsec is None:
            raise ValueError("Give pixel_scale_pc, or pixel_scale_arcsec and pc_per_arcsec")
        pixel_scale_pc = pixel_scale_arcsec * pc_per_arcsec
    data = (data - subtract) * scale
    data[~np.isfinite(data)] = np.nan
    return ExternalMap(data=data, agn_xy=tuple(map(float, agn_xy)), pixel_scale_pc=float(pixel_scale_pc),
                       rotation_deg=rotation_deg, flip_x=flip_x, name=name)

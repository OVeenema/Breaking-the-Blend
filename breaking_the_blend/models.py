"""
Kinematic models of the two components (paper Sect. 3.2).

All functions take sky-plane offsets ``x, y`` from the AGN in **parsec** and
return line-of-sight velocities in km/s.

Parameters (one flat dict is used everywhere)
---------------------------------------------
v_sys     systemic velocity [km/s] (0 if the redshift is right)
disc:     v_disc [km/s], R_disc [pc], inc_disc [deg], pa_disc [deg E of N]
outflow:  v_out [km/s], R_out [pc], inc_out [deg], pa_out [deg E of N],
          theta_in, theta_out [deg] (inner/outer opening angles of the hollow bicone)
combined: alpha (0-1), the disc weight in v = alpha*v_disc + (1-alpha)*v_outflow

Position angles
---------------
PAs are east of north. With the array plotted ``origin="lower"`` and north up,
east left (+x = west), the angle from +x is ``PA + pa_offset`` with
``pa_offset = 90`` (default). If your cube is rotated / flipped, change
``pa_offset`` (or rotate the cube) rather than editing the models.
For the disc, the receding (redshifted) side lies along ``pa_disc``; flip by
adding 180 deg. For the outflow, the redshifted cone lies along ``pa_out``.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

PA_OFFSET_DEG = 90.0

DISC_PARAMS = ["v_sys", "v_disc", "R_disc", "inc_disc", "pa_disc"]
OUTFLOW_PARAMS = ["v_sys", "v_out", "R_out", "inc_out", "pa_out", "theta_in", "theta_out"]
COMBINED_PARAMS = ["v_sys", "v_disc", "R_disc", "inc_disc", "pa_disc",
                   "v_out", "R_out", "inc_out", "pa_out", "theta_in", "theta_out", "alpha"]
COMPONENT_PARAMS = {"disc": DISC_PARAMS, "outflow": OUTFLOW_PARAMS, "combined": COMBINED_PARAMS}


def disc_velocity(x, y, v_disc, R_disc, inc_disc, pa_disc, v_sys=0.0, pa_offset=PA_OFFSET_DEG):
    """Thin inclined rotating disc with a tanh rotation curve (paper eqs. 1-5)."""
    phi = np.deg2rad(pa_disc + pa_offset)
    inc = np.deg2rad(inc_disc)
    xp = x * np.cos(phi) + y * np.sin(phi)          # along the major axis
    yp = -x * np.sin(phi) + y * np.cos(phi)         # along the minor axis
    R = np.maximum(np.sqrt(xp ** 2 + (yp / np.cos(inc)) ** 2), 1e-6)
    return v_sys + v_disc * np.tanh(R / R_disc) * np.sin(inc) * (xp / R)


def das_radial_velocity(r, v_out, R_out):
    """Piecewise-linear radial velocity law (paper eq. 9): rise to R_out, decline to 0 at 2 R_out."""
    r = np.asarray(r, dtype=float)
    v = np.zeros_like(r)
    rise = r <= R_out
    fall = (r > R_out) & (r <= 2 * R_out)
    v[rise] = v_out * r[rise] / R_out
    v[fall] = v_out * (2.0 - r[fall] / R_out)
    return v


def bicone_mask(x, y, inc_out, pa_out, theta_in, theta_out, pa_offset=PA_OFFSET_DEG):
    """True where the sky position falls inside the hollow bicone (paper eq. 8);
    also returns cos(theta) between the position vector and the cone axis."""
    phi = np.deg2rad(pa_out + pa_offset)
    inc = np.deg2rad(inc_out)
    R = np.maximum(np.sqrt(x ** 2 + y ** 2), 1e-6)
    cos_t = np.clip((x / R) * np.sin(inc) * np.cos(phi) + (y / R) * np.sin(inc) * np.sin(phi), -1, 1)
    t = np.arccos(cos_t)
    tin, tout = np.deg2rad(theta_in), np.deg2rad(theta_out)
    inside = ((t >= tin) & (t <= tout)) | ((np.pi - t >= tin) & (np.pi - t <= tout))
    return inside, cos_t


def bicone_velocity(x, y, v_out, R_out, inc_out, pa_out, theta_in, theta_out, v_sys=0.0,
                    pa_offset=PA_OFFSET_DEG, outside=0.0):
    """Projected velocity of a hollow bicone (paper eqs. 6-10).

    ``outside`` : value (relative to v_sys) outside the cone; 0 = systemic (as in
    the paper), ``np.nan`` = undefined.
    """
    inside, cos_t = bicone_mask(x, y, inc_out, pa_out, theta_in, theta_out, pa_offset)
    R = np.sqrt(x ** 2 + y ** 2)
    v = das_radial_velocity(R, v_out, R_out) * cos_t
    v = np.where(inside, v, outside)
    return v_sys + v


def component_velocities(x, y, p: Dict[str, float], pa_offset=PA_OFFSET_DEG, outside=0.0):
    """Disc and outflow model velocities for a full parameter dict."""
    vd = disc_velocity(x, y, p["v_disc"], p["R_disc"], p["inc_disc"], p["pa_disc"],
                       p.get("v_sys", 0.0), pa_offset)
    vo = bicone_velocity(x, y, p["v_out"], p["R_out"], p["inc_out"], p["pa_out"],
                         p["theta_in"], p["theta_out"], p.get("v_sys", 0.0), pa_offset, outside)
    return vd, vo


def model_velocity(x, y, p: Dict[str, float], component: str = "combined", pa_offset=PA_OFFSET_DEG):
    """Velocity field of ``component`` = ``"disc"``, ``"outflow"`` or ``"combined"`` (paper eq. 11)."""
    if component == "disc":
        return disc_velocity(x, y, p["v_disc"], p["R_disc"], p["inc_disc"], p["pa_disc"],
                             p.get("v_sys", 0.0), pa_offset)
    if component == "outflow":
        return bicone_velocity(x, y, p["v_out"], p["R_out"], p["inc_out"], p["pa_out"],
                               p["theta_in"], p["theta_out"], p.get("v_sys", 0.0), pa_offset)
    if component == "combined":
        vd, vo = component_velocities(x, y, p, pa_offset)
        a = p["alpha"]
        return a * vd + (1 - a) * vo
    raise ValueError("component must be 'disc', 'outflow' or 'combined'")

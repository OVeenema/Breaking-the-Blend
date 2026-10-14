"""
Schematic of the disc + bicone geometry relative to the line of sight
(in the spirit of Fischer et al. 2013, ApJS 209, 1).

Two views are drawn:

* **view from the observer** - the sky plane (north up, east left), with exact occlusion:
  parts of the cones behind the disc are drawn as wire-frame edges only, parts in front of
  it are shaded, and the lines where the cones cross the disc plane are highlighted;
* **3-D view** - an oblique view with the line of sight marked.

Conventions (identical to :mod:`breaking_the_blend.models`)
-----------------------------------------------------------
* coordinates (E, N, Z): east, north, and Z towards the observer;
* ``pa_out`` / ``pa_disc``: position angles east of north of the redshifted cone and of the
  receding side of the disc major axis;
* ``inc_out``: angle between the bicone axis and the line of sight (0 = pole-on,
  90 = in the plane of the sky); the redshifted cone points away from the observer.
  Note: Fischer et al. (2013) measure the inclination from the plane of the sky,
  i.e. ``inc_out = 90 - i_Fischer``;
* ``inc_disc``: disc inclination (0 = face-on, 90 = edge-on);
* ``theta_in`` / ``theta_out``: inner / outer opening angles of the hollow bicone, measured from the
  axis. In the kinematic model they apply to sky-projected positions, so by default the outer angle
  is drawn as ``theta_out / 2`` (``halve_theta_out=False`` draws it unchanged, e.g. for Fischer et al.
  half-opening angles).

Which side of the disc is nearer to us cannot be told from its velocity field; set it with
``disc_near_side`` (+1: the side at PA_disc + 90 deg is the near side; -1: PA_disc - 90 deg),
e.g. from the dust lanes.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RED = "#e0594d"
BLUE = "#4a8fd9"
DISC = "#8c8c8c"
CROSS = "#f2b600"

GEOMETRY_KEYS = ("inc_disc", "pa_disc", "inc_out", "pa_out", "theta_in", "theta_out")


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def outflow_axis(inc_out: float, pa_out: float) -> np.ndarray:
    """Unit vector (E, N, Z) of the redshifted (receding) cone axis."""
    i, pa = np.deg2rad(inc_out), np.deg2rad(pa_out)
    return np.array([np.sin(i) * np.sin(pa), np.sin(i) * np.cos(pa), -np.cos(i)])


def disc_frame(inc_disc: float, pa_disc: float, near_side: int = 1):
    """(major axis, in-disc minor axis, normal towards the observer) of the disc."""
    i, pa = np.deg2rad(inc_disc), np.deg2rad(pa_disc)
    m = np.array([np.sin(pa), np.cos(pa), 0.0])
    p = np.array([np.cos(pa), -np.sin(pa), 0.0])          # sky direction at PA_disc + 90 deg
    u = np.cos(i) * p + near_side * np.sin(i) * np.array([0.0, 0.0, 1.0])
    n = np.cross(m, u)
    if n[2] < 0:
        n = -n
    return m, u, n / np.linalg.norm(n)


def _basis(a):
    a = a / np.linalg.norm(a)
    t = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, t)
    e1 /= np.linalg.norm(e1)
    return a, e1, np.cross(a, e1)


def _generator(a, e1, e2, theta, phi):
    th = np.deg2rad(theta)
    return (np.cos(th) * a[:, None] + np.sin(th) * (np.cos(phi) * e1[:, None] + np.sin(phi) * e2[:, None]))


def _cone_quads(axis, theta, length, n_phi, n_s):
    """Quadrilateral facets (n, 4, 3) of a cone surface of half-angle ``theta``, truncated at
    distance ``length`` from the apex (the model outflow ends at a fixed radius, 2 R_out)."""
    a, e1, e2 = _basis(axis)
    th = min(theta, 89.0)
    smax = length
    phi = np.linspace(0, 2 * np.pi, n_phi + 1)
    s = np.linspace(0, smax, n_s + 1)
    g = _generator(a, e1, e2, th, phi)                        # (3, n_phi+1)
    P = s[:, None, None] * g.T[None, :, :]                    # (n_s+1, n_phi+1, 3)
    quads = np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]], axis=2)
    return quads.reshape(-1, 4, 3)


def _hidden(points, n, disc_radius):
    """True where a point lies behind the disc (as seen by the observer)."""
    d = points @ n
    nz = max(n[2], 1e-3)
    proj = points + (-d / nz)[..., None] * np.array([0.0, 0.0, 1.0])
    return (d < 0) & (np.linalg.norm(proj, axis=-1) <= disc_radius)


def _disc_crossings(axis, theta, n, smax):
    """End points of the cone generators lying in the disc plane."""
    a, e1, e2 = _basis(axis)
    th = np.deg2rad(min(theta, 89.0))
    A, B, C = np.sin(th) * (e1 @ n), np.sin(th) * (e2 @ n), np.cos(th) * (a @ n)
    R0 = np.hypot(A, B)
    if R0 == 0 or abs(C) > R0:
        return []
    phi0 = np.arctan2(B, A)
    dphi = np.arccos(np.clip(-C / R0, -1, 1))
    out = []
    for ph in (phi0 + dphi, phi0 - dphi):
        g = _generator(a, e1, e2, np.rad2deg(th), np.array([ph]))[:, 0]
        out.append(smax * g)
    return out


def geometry_params(p: Dict[str, float]) -> Dict[str, float]:
    """The geometry entries of a full parameter dict (``theta_in`` defaults to 0)."""
    missing = [k for k in ("inc_disc", "pa_disc", "inc_out", "pa_out", "theta_out") if k not in p]
    if missing:
        raise KeyError(f"geometry needs {missing}")
    g = {k: float(p.get(k, 0.0)) for k in GEOMETRY_KEYS}
    for k in ("R_out", "R_disc"):
        if k in p and p[k] is not None and np.isscalar(p[k]):
            g[k] = float(p[k])
    return g


def mean_geometry(params) -> Dict[str, float]:
    """Average geometry over several lines: ``params`` is ``{line: param_dict}`` (e.g. the
    parameters used in the decomposition) or ``{line: KinematicFit}``. Position angles are
    averaged on the circle."""
    dicts = [v.params if hasattr(v, "params") else v for v in (params.values() if isinstance(params, dict) else params)]
    out = {}
    for k in GEOMETRY_KEYS + ("R_out", "R_disc"):
        vals = np.array([d[k] for d in dicts if k in d and d[k] is not None and np.isscalar(d[k])], float)
        if vals.size == 0:
            continue
        if k.startswith("pa_"):
            a = np.deg2rad(vals)
            out[k] = float(np.rad2deg(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) % 360)
        else:
            out[k] = float(np.mean(vals))
    return out


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def drawn_angles(g: Dict[str, float], halve_theta_out: bool = True) -> Dict[str, float]:
    """Opening angles used for the drawing. With ``halve_theta_out`` the outer angle is drawn as
    theta_out / 2 (the model's angles are measured on sky-projected positions, so a full-size 3-D
    cone looks too wide); the inner angle is kept, but limited to just inside the drawn outer one."""
    g = dict(g)
    if halve_theta_out:
        g["theta_out"] = g["theta_out"] / 2.0
    if g["theta_in"] >= g["theta_out"]:
        g["theta_in"] = 0.0          # no room for a hollow core: draw a filled cone
    return g


def _setup(p, length_pc, disc_radius_pc, halve_theta_out=True):
    g = drawn_angles(geometry_params(p), halve_theta_out)
    L = length_pc or (2.0 * g["R_out"] if "R_out" in g else 1000.0)
    R = disc_radius_pc or 1.25 * L
    return g, L, R


def plot_sky_view(params: Dict[str, float], ax=None, length_pc: Optional[float] = None,
                  disc_radius_pc: Optional[float] = None, disc_near_side: int = 1,
                  n_phi: int = 72, n_s: int = 14, show_disc: bool = True, legend: bool = True,
                  title: Optional[str] = "View from the observer", halve_theta_out: bool = True):
    """The geometry as seen by the observer (north up, east left), with occlusion by the disc.
    ``halve_theta_out``: draw the outer opening angle as theta_out / 2 (see :func:`drawn_angles`)."""
    from .plotting import style_axes
    g, L, R = _setup(params, length_pc, disc_radius_pc, halve_theta_out)
    m, u, n = disc_frame(g["inc_disc"], g["pa_disc"], disc_near_side)
    a = outflow_axis(g["inc_out"], g["pa_out"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    style_axes(ax)

    surfaces = [(g["theta_out"], 0.42)] + ([(g["theta_in"], 0.14)] if g["theta_in"] > 0.5 else [])
    thetas = [t for t, _ in surfaces]
    front_q, front_c, front_a, front_z = [], [], [], []
    for axis, col in ((a, RED), (-a, BLUE)):
        for th, alpha in surfaces:
            q = _cone_quads(axis, th, L, n_phi, n_s)
            cen = q.mean(axis=1)
            hid = _hidden(cen, n, R) if show_disc else np.zeros(len(q), bool)
            front_q.extend(q[~hid])
            front_c.extend([col] * int((~hid).sum()))
            front_a.extend([alpha] * int((~hid).sum()))
            front_z.extend(cen[~hid, 2])
            # 1) parts behind the disc: a coarse wire-frame (edges only)
            qc = _cone_quads(axis, th, L, 24, 5)
            hc = _hidden(qc.mean(axis=1), n, R) if show_disc else np.zeros(len(qc), bool)
            if hc.any():
                ax.add_collection(PolyCollection(qc[hc][:, :, :2], facecolors="none", edgecolors=col,
                                                 linewidths=0.5, alpha=0.7 if alpha > 0.2 else 0.35, zorder=1))
    # 2) the disc
    if show_disc:
        psi = np.linspace(0, 2 * np.pi, 361)
        rim = R * (np.cos(psi)[:, None] * m + np.sin(psi)[:, None] * u)
        ax.fill(rim[:, 0], rim[:, 1], color=DISC, alpha=0.22, lw=0, zorder=2)
        segs = []
        for r in np.linspace(R / 6, R, 6):
            ring = r * (np.cos(psi)[:, None] * m + np.sin(psi)[:, None] * u)
            segs.append(ring[:, :2])
        for ps in np.deg2rad(np.arange(0, 360, 15)):
            segs.append(np.array([[0, 0], (R * (np.cos(ps) * m + np.sin(ps) * u))[:2]]))
        ax.add_collection(LineCollection(segs, colors="k", linewidths=0.35, alpha=0.45, zorder=2))
        maj = np.array([-R * m, R * m])
        ax.plot(maj[:, 0], maj[:, 1], ls="--", color="k", lw=1.0, alpha=0.7, zorder=2)
    # 3) parts in front of the disc: shaded, far to near
    if front_q:
        from matplotlib.colors import to_rgba
        order = np.argsort(front_z)
        fq = np.array(front_q)[order]
        rgba = np.array([to_rgba(c, al) for c, al in zip(np.array(front_c)[order], np.array(front_a)[order])])
        ax.add_collection(PolyCollection(fq[:, :, :2], facecolors=rgba, edgecolors=rgba, linewidths=0.3,
                                         zorder=3))
        # rims of the outer cones
        for axis, col in ((a, RED), (-a, BLUE)):
            ae, e1, e2 = _basis(axis)
            rim = (L * _generator(ae, e1, e2, min(g["theta_out"], 89.0), np.linspace(0, 2 * np.pi, 181))).T
            ax.plot(rim[:, 0], rim[:, 1], color=col, lw=1.0, alpha=0.8, zorder=3)
    # 4) where the cones cross the disc plane
    if show_disc:
        for axis in (a, -a):
            for th in thetas:
                for end in _disc_crossings(axis, th, n, min(L, R)):
                    ax.plot([0, end[0]], [0, end[1]], color=CROSS, lw=2.4, zorder=4)
    # outflow axis projection and AGN
    ax.plot([-L * a[0], L * a[0]], [-L * a[1], L * a[1]], ls=":", color="k", lw=1.0, zorder=4)
    ax.plot(0, 0, "o", color="k", ms=5, zorder=5)

    lim = 1.08 * max(R, L)
    ax.set_xlim(lim, -lim)                    # east to the left
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta$RA (east $\leftarrow$) [pc]")
    ax.set_ylabel(r"$\Delta$Dec [pc]")
    if title:
        ax.set_title(title)
    # compass
    c0 = np.array([0.88, 0.08])
    for d, lab in ((np.array([0.0, 0.09]), "N"), (np.array([-0.09, 0.0]), "E")):
        ax.annotate("", xy=c0 + d, xytext=c0, xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.3))
        ax.text(*(c0 + 1.35 * d), lab, transform=ax.transAxes, ha="center", va="center")
    if legend:
        handles = [Patch(color=RED, alpha=0.6, label="redshifted cone"),
                   Patch(color=BLUE, alpha=0.6, label="blueshifted cone"),
                   Line2D([], [], color=CROSS, lw=2.4, label="cone / disc intersection")]
        if show_disc:
            handles += [Patch(color=DISC, alpha=0.35, label="disc"),
                        Line2D([], [], color="k", ls="--", lw=1, label="disc major axis")]
        ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True, framealpha=0.85)
    return ax.figure


def plot_3d_view(params: Dict[str, float], ax=None, length_pc: Optional[float] = None,
                 disc_radius_pc: Optional[float] = None, disc_near_side: int = 1, elev: float = 35.0,
                 azim: float = -75.0, n_phi: int = 48, n_s: int = 10, show_disc: bool = True,
                 title: Optional[str] = "3-D view", halve_theta_out: bool = True):
    """Oblique 3-D view of the same geometry (Z points to the observer)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    g, L, R = _setup(params, length_pc, disc_radius_pc, halve_theta_out)
    m, u, n = disc_frame(g["inc_disc"], g["pa_disc"], disc_near_side)
    a = outflow_axis(g["inc_out"], g["pa_out"])
    if ax is None:
        fig = plt.figure(figsize=(6.4, 6.4), constrained_layout=True)
        ax = fig.add_subplot(projection="3d")
    thetas = [t for t in (g["theta_out"], g["theta_in"]) if t > 0.5]
    if show_disc:
        psi = np.linspace(0, 2 * np.pi, 121)
        rim = R * (np.cos(psi)[:, None] * m + np.sin(psi)[:, None] * u)
        ax.add_collection3d(Poly3DCollection([rim], facecolors=DISC, alpha=0.18, edgecolors="none"), autolim=False)
        segs = [r * (np.cos(psi)[:, None] * m + np.sin(psi)[:, None] * u) for r in np.linspace(R / 5, R, 5)]
        segs += [np.array([[0, 0, 0], R * (np.cos(p) * m + np.sin(p) * u)]) for p in np.deg2rad(np.arange(0, 360, 20))]
        ax.add_collection3d(Line3DCollection(segs, colors="k", linewidths=0.3, alpha=0.4), autolim=False)
    surfaces = [(g["theta_out"], 0.38)] + ([(g["theta_in"], 0.12)] if g["theta_in"] > 0.5 else [])
    for axis, col in ((a, RED), (-a, BLUE)):
        for th, alpha in surfaces:
            q = _cone_quads(axis, th, L, n_phi, n_s)
            hid = _hidden(q.mean(axis=1), n, R) if show_disc else np.zeros(len(q), bool)
            if (~hid).any():
                ax.add_collection3d(Poly3DCollection(q[~hid], facecolors=col, edgecolors="none", alpha=alpha),
                                    autolim=False)
            qc = _cone_quads(axis, th, L, 24, 5)
            hc = _hidden(qc.mean(axis=1), n, R) if show_disc else np.zeros(len(qc), bool)
            if hc.any():
                edges = np.concatenate([qc[hc][:, [0, 1]], qc[hc][:, [1, 2]]])
                ax.add_collection3d(Line3DCollection(edges, colors=col, linewidths=0.4, alpha=0.6), autolim=False)
        if show_disc:
            for th in thetas:
                for end in _disc_crossings(axis, th, n, min(L, R)):
                    ax.plot([0, end[0]], [0, end[1]], [0, end[2]], color=CROSS, lw=2.2)
    lim = 1.05 * max(R, L)
    ax.plot([0, 0], [0, 0], [0, 1.25 * lim], color="seagreen", lw=1.6)
    ax.text(0, 0, 1.3 * lim, "to observer", color="seagreen", ha="center")
    ax.plot([0, 0.45 * lim], [0, 0], [0, 0], color="k", lw=1.0)
    ax.text(0.5 * lim, 0, 0, "E", fontsize=10)
    ax.plot([0, 0], [0, 0.45 * lim], [0, 0], color="k", lw=1.0)
    ax.text(0, 0.5 * lim, 0, "N", fontsize=10)
    ax.scatter([0], [0], [0], color="k", s=12)
    ax.set_xlim(lim, -lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("E [pc]", labelpad=6)
    ax.set_ylabel("N [pc]", labelpad=6)
    ax.set_zlabel("towards observer [pc]", labelpad=6)
    ax.tick_params(labelsize=8)
    for axis_ in (ax.xaxis, ax.yaxis, ax.zaxis):          # clean background
        axis_.pane.fill = False
        axis_.pane.set_edgecolor("lightgrey")
    ax.grid(False)
    if title:
        ax.set_title(title)
    return ax.figure


def plot_geometry(params: Dict[str, float], length_pc: Optional[float] = None,
                  disc_radius_pc: Optional[float] = None, disc_near_side: int = 1,
                  views: Iterable[str] = ("sky", "3d"), elev: float = 35.0, azim: float = -75.0,
                  show_disc: bool = True, title: Optional[str] = None, print_params: bool = True,
                  halve_theta_out: bool = True):
    """Schematic of the disc + hollow-bicone geometry relative to the line of sight.

    Parameters
    ----------
    params : dict with ``inc_disc, pa_disc, inc_out, pa_out, theta_in, theta_out`` (and optionally
        ``R_out``, which sets the drawn cone length to 2 R_out) - e.g. a fit's ``.params``, one
        line's decomposition parameters, or :func:`mean_geometry` of several.
    length_pc : radial extent of each cone (default 2 R_out, where the model outflow velocity reaches 0).
    disc_radius_pc : radius of the drawn disc (default 1.25 x the cone length).
    disc_near_side : +1 / -1 - which side of the disc minor axis is nearer to us (see module docstring).
    views : any of ``"sky"`` (view from the observer) and ``"3d"``.
    elev, azim : viewing angles of the 3-D panel.
    halve_theta_out : draw the outer opening angle as theta_out / 2 (default). The model's opening
        angles are applied to sky-projected positions, so drawing them as full 3-D half-angles makes
        the cones look too wide. Set False for angles that already are 3-D half-opening angles
        (e.g. Fischer et al. 2013).
    """
    views = list(views)
    g = geometry_params(params)
    gd = drawn_angles(g, halve_theta_out)
    fig = plt.figure(figsize=(6.6 * len(views), 6.6), constrained_layout=True)
    for k, v in enumerate(views):
        if v == "sky":
            plot_sky_view(params, fig.add_subplot(1, len(views), k + 1), length_pc, disc_radius_pc,
                          disc_near_side, show_disc=show_disc, halve_theta_out=halve_theta_out)
        elif v == "3d":
            plot_3d_view(params, fig.add_subplot(1, len(views), k + 1, projection="3d"), length_pc,
                         disc_radius_pc, disc_near_side, elev, azim, show_disc=show_disc,
                         halve_theta_out=halve_theta_out)
        else:
            raise ValueError("views must contain 'sky' and/or '3d'")
    summary = (f"disc: i = {g['inc_disc']:.0f}$^\\circ$, PA = {g['pa_disc']:.0f}$^\\circ$   |   "
               f"bicone: i = {g['inc_out']:.0f}$^\\circ$ (from LOS), PA = {g['pa_out']:.0f}$^\\circ$, "
               f"$\\theta$ = {g['theta_in']:.0f}-{g['theta_out']:.0f}$^\\circ$"
               + (f" (drawn {gd['theta_in']:.0f}-{gd['theta_out']:.0f}$^\\circ$)" if halve_theta_out else ""))
    fig.suptitle((title + "\n" if title else "") + summary, fontsize=12)
    if print_params:
        print(f"disc:   inc = {g['inc_disc']:.1f} deg, PA = {g['pa_disc']:.1f} deg (receding side)")
        print(f"bicone: inc = {g['inc_out']:.1f} deg from the line of sight "
              f"({90 - g['inc_out']:.1f} deg from the sky plane), PA = {g['pa_out']:.1f} deg (redshifted cone), "
              f"theta_in/out = {g['theta_in']:.1f}/{g['theta_out']:.1f} deg")
        if halve_theta_out:
            print(f"  drawn with theta_out/2: theta_in/out = {gd['theta_in']:.1f}/{gd['theta_out']:.1f} deg"
                  + ("  (theta_in >= theta_out/2: drawn as a filled cone)" if gd["theta_in"] == 0 < g["theta_in"] else ""))
        if gd["theta_out"] > 89:
            print("  note: drawn theta_out > 89 deg is limited to 89 deg")
        beta = np.rad2deg(np.arccos(abs(outflow_axis(g['inc_out'], g['pa_out']) @
                                        disc_frame(g['inc_disc'], g['pa_disc'], disc_near_side)[2])))
        print(f"angle between the bicone axis and the disc normal: {beta:.1f} deg "
              f"(depends on disc_near_side = {disc_near_side:+d})")
    return fig

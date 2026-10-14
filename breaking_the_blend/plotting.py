"""
Plotting helpers for every step, in a publication style matching the
Breaking the Blend paper (serif STIX fonts, inward ticks on all sides, minor
ticks). The style is applied when this module is imported; call
:func:`set_style` to re-apply or change it (e.g. ``set_style(**{"font.size": 16})``).

All maps are drawn with ``origin="lower"`` and axes in parsec offsets from the
AGN (marked with a cross).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
from matplotlib.ticker import AutoMinorLocator

from .linefit import LineMaps
from .dispersion import dispersion_samples

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "axes.linewidth": 1.3,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 7, "ytick.major.size": 7,
    "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.minor.width": 0.9, "ytick.minor.width": 0.9,
    "lines.linewidth": 1.6,
    "image.origin": "lower",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def set_style(**overrides) -> None:
    """Apply the paper plotting style (optionally with rcParams overrides)."""
    mpl.rcParams.update({**STYLE, **overrides})


set_style()

VEL_CMAP = "seismic"
SIG_CMAP = "magma"
FLUX_CMAP = "rainbow"


def _cmap(name: str, fallback: str):
    """cmasher colour map ``cmr.<name>`` if cmasher is installed, else ``fallback``."""
    try:
        import cmasher  # noqa: F401  (registers the cmr.* colour maps)
        return plt.get_cmap(f"cmr.{name}")
    except Exception:
        return plt.get_cmap(fallback)


def fraction_cmap(which: str = "f_disc"):
    """Colour map for fraction maps (cmasher iceburn), oriented so that star formation is
    always orange and the AGN outflow always blue:

    * ``"f_disc"`` (disc fraction): high values orange;
    * ``"f_out"`` (AGN outflow fraction): high values blue (reversed map).
    """
    if which == "f_out":
        return _cmap("iceburn_r", "RdYlBu")
    return _cmap("iceburn", "RdYlBu_r")


def style_axes(ax, minor: bool = True) -> None:
    """Inward ticks on all four sides with minor ticks (paper style)."""
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=7, width=1.2)
    ax.tick_params(which="minor", length=3.5, width=0.9)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    for s in ax.spines.values():
        s.set_linewidth(1.3)


def style_colorbar(cbar, label: Optional[str] = None) -> None:
    cbar.ax.tick_params(which="both", direction="in")
    cbar.ax.tick_params(which="major", length=6, width=1.1)
    cbar.ax.tick_params(which="minor", length=3, width=0.8)
    cbar.outline.set_linewidth(1.2)
    try:
        cbar.ax.minorticks_on()
    except Exception:
        pass
    if label:
        cbar.set_label(label)


def _grid(n, ncols, size=3.3, extra_w=0.0):
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols + extra_w, size * nrows),
                             squeeze=False, constrained_layout=True)
    axes = list(axes.flat)
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[:n]:
        style_axes(ax)
    return fig, axes


def _agn(ax, color="k"):
    ax.plot(0, 0, "x", color=color, ms=9, mew=2.2)


def _sorted(d):
    return sorted(d.values(), key=lambda o: o.ip if hasattr(o, "ip") else o.line.ip)


def _map_labels(fig):
    fig.supxlabel(r"$\Delta x$ [pc]")
    fig.supylabel(r"$\Delta y$ [pc]")


def _panel_text(ax, label, ip=None, color="k"):
    txt = label if ip is None else f"{label}\nIP = {ip:g} eV"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top", fontsize=11, color=color)


# ----------------------------------------------------------------------
# Data overview
# ----------------------------------------------------------------------
def _continuum_peak(cube):
    from .cubes import find_agn_position
    return find_agn_position(cube.continuum_image())


def plot_cubes_overview(cubes: Dict, ncols: int = 5, mark: str = "continuum"):
    """Median continuum image of every cube, marking the continuum peak.

    mark : ``"continuum"`` (cross at the continuum peak), ``"adopted"`` (the AGN position used
        in the analysis, incl. overrides) or ``"both"`` (cross = peak, circle = adopted).
    """
    cs = sorted(cubes.values(), key=lambda c: c.wave_range[0])
    fig, axes = _grid(len(cs), ncols, size=3.0)
    for ax, c in zip(axes, cs):
        img = c.continuum_image()
        pos = img[img > 0]
        ax.imshow(np.where(img > 0, img, np.nan), cmap=FLUX_CMAP,
                  norm=LogNorm(*np.percentile(pos, [5, 99.8])) if pos.size else None)
        if mark in ("continuum", "both"):
            ax.plot(*_continuum_peak(c), "x", color="k", ms=9, mew=2)
        if mark in ("adopted", "both") and c.agn_xy is not None:
            ax.plot(*c.agn_xy, "o", mfc="none", mec="w" if mark == "both" else "k", ms=10, mew=1.8)
        ax.set_title(f"{c.name}\n{c.wave_range[0]:.2f}-{c.wave_range[1]:.2f} $\\mu$m", fontsize=11)
    fig.supxlabel("x [pixel]")
    fig.supylabel("y [pixel]")
    return fig


# ----------------------------------------------------------------------
# Step 1 maps
# ----------------------------------------------------------------------
_QUANT = {
    #  name      getter                                  label                                          cmap       vlim
    "flux":    (lambda lm: lm.flux * 1e15, r"Flux [$10^{-15}$ erg s$^{-1}$ cm$^{-2}$]", FLUX_CMAP, None),
    "mom0":    (lambda lm: lm.flux * 1e15, r"Moment 0 [$10^{-15}$ erg s$^{-1}$ cm$^{-2}$]", FLUX_CMAP, None),
    "velocity": (lambda lm: lm.single["v"], r"Velocity (single Gaussian) [km s$^{-1}$]", VEL_CMAP, (-400, 400)),
    "sigma":   (lambda lm: lm.single["sigma"], r"$\sigma$ (single Gaussian) [km s$^{-1}$]", SIG_CMAP, (0, 400)),
    "mom1":    (lambda lm: lm.mom1, r"Moment 1 [km s$^{-1}$]", VEL_CMAP, (-400, 400)),
    "mom2":    (lambda lm: lm.mom2, r"Moment 2 [km s$^{-1}$]", SIG_CMAP, (0, 400)),
    "v_n":     (lambda lm: lm.double["v_n"], r"Narrow-component velocity [km s$^{-1}$]", VEL_CMAP, (-400, 400)),
    "v_b":     (lambda lm: lm.double["v_b"], r"Broad-component velocity [km s$^{-1}$]", VEL_CMAP, (-400, 400)),
    "sigma_n": (lambda lm: lm.double["sigma_n"], r"Narrow-component $\sigma$ [km s$^{-1}$]", SIG_CMAP, (0, 400)),
    "sigma_b": (lambda lm: lm.double["sigma_b"], r"Broad-component $\sigma$ [km s$^{-1}$]", SIG_CMAP, (0, 400)),
}
_SHORT = {"flux": "flux", "mom0": "mom 0", "velocity": "$v$", "sigma": r"$\sigma$", "mom1": "mom 1",
          "mom2": "mom 2", "v_n": "narrow $v$", "v_b": "broad $v$", "sigma_n": r"narrow $\sigma$",
          "sigma_b": r"broad $\sigma$"}


def _get(lm, q, masked=True):
    arr = np.array(_QUANT[q][0](lm), dtype=float)
    if masked:
        arr[~lm.good] = np.nan
    return arr


def plot_line_maps(line_maps: Dict[str, LineMaps], quantity: str = "flux", masked: bool = True,
                   vlim: Optional[Tuple[float, float]] = None, ncols: int = 6, cmap=None,
                   gamma: float = 0.5, share_scale: bool = False):
    """Grid of one quantity for all lines, ordered by IP (paper Figs. S1, S2).

    quantity : ``flux``/``mom0``, ``velocity``, ``sigma``, ``mom1``, ``mom2``,
        ``v_n``, ``v_b``, ``sigma_n``, ``sigma_b``.
    share_scale : flux maps share one colour scale (else each has its own colour bar).
    """
    get, label, cm, vdef = _QUANT[quantity]
    lms = _sorted(line_maps)
    is_flux = quantity in ("flux", "mom0")
    fig, axes = _grid(len(lms), ncols, size=3.1)
    vlim = vlim or vdef
    if is_flux and share_scale and vlim is None:
        vlim = (0, max(np.nanmax(_get(lm, quantity)) for lm in lms))
    im = None
    for ax, lm in zip(axes, lms):
        arr = _get(lm, quantity, masked)
        kw = dict(extent=lm.extent_pc(), cmap=cmap or cm)
        if is_flux:
            kw["norm"] = PowerNorm(gamma, vmin=0, vmax=vlim[1] if vlim else np.nanmax(arr))
        else:
            kw.update(vmin=vlim[0], vmax=vlim[1])
        im = ax.imshow(arr, **kw)
        _agn(ax)
        ax.set_title(f"{lm.label}  ({lm.ip:g} eV)", fontsize=12)
        if is_flux and not share_scale:
            style_colorbar(fig.colorbar(im, ax=ax, shrink=0.8))
    if not (is_flux and not share_scale):
        style_colorbar(fig.colorbar(im, ax=axes[:len(lms)], shrink=0.6), label)
    else:
        fig.suptitle(label)
    _map_labels(fig)
    return fig


def _kind(q):
    return "flux" if q in ("flux", "mom0") else ("vel" if q in ("velocity", "mom1", "v_n", "v_b") else "sig")


_KIND_LABEL = {"vel": r"Velocity [km s$^{-1}$]", "sig": r"Velocity dispersion [km s$^{-1}$]",
               "flux": "Moment 0 (normalised to the peak of each line)"}


def plot_line_quantities(line_maps: Dict[str, LineMaps], quantities: Sequence[str] = ("v_n", "v_b"),
                         lines_per_row: int = 3, masked: bool = True, vlims: Optional[Dict] = None):
    """Several quantities side by side for every line (one block of panels per line).

    Examples
    --------
    double-Gaussian velocities : ``quantities=("v_n", "v_b")``
    double-Gaussian dispersions : ``quantities=("sigma_n", "sigma_b")``
    moment maps : ``quantities=("mom0", "mom1", "mom2")``

    vlims : ``{quantity: (lo, hi)}`` colour ranges (defaults: +/-400 km/s, 0-400 km/s).
    Quantities of the same kind (velocity / dispersion / flux) share one colour bar.
    """
    lms = _sorted(line_maps)
    nq = len(quantities)
    nl = min(lines_per_row, len(lms))
    ncols = nq * nl
    nrows = int(np.ceil(len(lms) / lines_per_row))
    fig = plt.figure(figsize=(2.75 * ncols, 2.75 * nrows + 1.1))
    axs = np.atleast_2d(fig.subplots(nrows, ncols, squeeze=False))
    bottom = 1.0 / (2.75 * nrows + 1.1)          # leave ~1 inch for colour bars
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=bottom + 0.06, wspace=0.12, hspace=0.32)
    for ax in axs.flat:
        ax.axis("off")
    vlims = vlims or {}
    kinds = {}
    for k, lm in enumerate(lms):
        r, c0 = divmod(k, lines_per_row)
        for j, q in enumerate(quantities):
            ax = axs[r, c0 * nq + j]
            ax.axis("on")
            style_axes(ax)
            ax.tick_params(labelsize=10)
            get, label, cm, vdef = _QUANT[q]
            arr = _get(lm, q, masked)
            if _kind(q) == "flux":
                arr = arr / np.nanmax(arr)
                im = ax.imshow(arr, extent=lm.extent_pc(), cmap=cm, norm=PowerNorm(0.5, vmin=0, vmax=1))
            else:
                lo, hi = vlims.get(q, vdef)
                im = ax.imshow(arr, extent=lm.extent_pc(), cmap=cm, vmin=lo, vmax=hi)
            _agn(ax)
            ax.set_title(f"{lm.label} {_SHORT[q]}", fontsize=11)
            if j > 0:
                ax.set_yticklabels([])
            kinds.setdefault(_kind(q), im)
    nk = len(kinds)
    w = 0.8 / nk
    for i, (kind, im) in enumerate(kinds.items()):
        cax = fig.add_axes([0.1 + i * w + 0.03, bottom * 0.55, w - 0.06, 0.018])
        style_colorbar(fig.colorbar(im, cax=cax, orientation="horizontal"), _KIND_LABEL[kind])
    fig.text(0.5, bottom + 0.012, r"$\Delta x$ [pc]", ha="center", fontsize=14)
    fig.text(0.012, 0.5 + bottom / 2, r"$\Delta y$ [pc]", va="center", rotation=90, fontsize=14)
    return fig


def plot_spaxel_fits(lm: LineMaps, spaxels: Optional[Sequence[Tuple[int, int]]] = None,
                     n: int = 6, seed: int = 0):
    """Data + single/double Gaussian fits in a few spaxels (row, col). Default: the
    brightest spaxel plus random good ones - use this to check the initial guesses."""
    from .linefit import spaxel_fit_curves
    if spaxels is None and getattr(lm, "fitted_spaxels", None):
        spaxels = lm.fitted_spaxels[:n]
    if spaxels is None:
        good = np.argwhere(lm.good)
        rng = np.random.default_rng(seed)
        peak = np.unravel_index(np.nanargmax(np.where(lm.good, lm.flux, -np.inf)), lm.shape)
        pick = good[rng.choice(len(good), size=min(n - 1, len(good)), replace=False)]
        spaxels = [tuple(peak)] + [tuple(p) for p in pick]
    fig, axes = _grid(len(spaxels), 3, size=3.6)
    for ax, (i, j) in zip(axes, spaxels):
        c = spaxel_fit_curves(lm, i, j)
        ax.step(c["v"], c["data"] * 1e3, where="mid", color="k", lw=1.2, label="data")
        if "single" in c:
            ax.plot(c["v"], c["single"] * 1e3, color="orange", label="single")
        if "double" in c:
            ax.plot(c["v"], c["double"] * 1e3, "--", color="grey", label="double")
            ax.plot(c["v"], c["narrow"] * 1e3, color="darkviolet", lw=1.2, label="narrow")
            ax.plot(c["v"], c["broad"] * 1e3, color="limegreen", lw=1.2, label="broad")
        ax.set_title(f"{lm.label}  spaxel ({i}, {j})", fontsize=11)
        ax.set_xlim(lm.vel_axis[lm.in_line].min() * 1.3, lm.vel_axis[lm.in_line].max() * 1.3)
    axes[0].legend(fontsize=9)
    fig.supxlabel(r"Velocity [km s$^{-1}$]")
    fig.supylabel("Flux density [mJy]")
    return fig


def plot_line_window(cube, line, n_brightest: int = 20, ax=None, catalogue=None):
    """Check a line's wavelength settings: the summed spectrum of the brightest spaxels over the
    fit window, with the line region (shaded), the continuum fit, the line's rest wavelength and
    any other catalogue lines nearby (dotted, labelled) that could blend with it."""
    from .linefit import subtract_continuum
    from .lines import LINE_CATALOGUE
    lo, hi = line.get_fit_window()
    lr = line.get_line_region()
    pad = 0.5 * (hi - lo)
    sel = (cube.wave >= lo - pad) & (cube.wave <= hi + pad)
    win = (cube.wave[sel] >= lo) & (cube.wave[sel] <= hi)
    wave = cube.wave[sel]
    cut = np.asarray(cube.data[sel], float)
    lsel = (wave >= lr[0]) & (wave <= lr[1])
    img = cut[lsel].sum(axis=0)
    idx = np.argsort(img.ravel())[::-1][:n_brightest]
    spec = cut.reshape(len(wave), -1)[:, idx].sum(axis=1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    style_axes(ax)
    ax.step(wave, spec * 1e3, where="mid", color="k", lw=1.1)
    ax.axvspan(*lr, color="orange", alpha=0.25, label="line region")
    for x in (lo, hi):
        ax.axvline(x, color="grey", ls="--", lw=1)
    sub = subtract_continuum(wave[win], cut[win][:, None, :].reshape(win.sum(), 1, -1)[:, :, idx].sum(axis=2)[:, :, None],
                             lr, line.cont_order)[:, 0, 0]
    ax.plot(wave[win], (spec[win] - sub) * 1e3, color="C0", lw=1.4, label="continuum fit")
    ax.axvline(line.wave, color="crimson", lw=1.2)
    for other in (catalogue or LINE_CATALOGUE).values():
        if other.name != line.name and wave[0] <= other.wave <= wave[-1]:
            ax.axvline(other.wave, color="purple", ls=":", lw=1.4)
            ax.text(other.wave, 0.97, other.label, transform=ax.get_xaxis_transform(), rotation=90,
                    ha="right", va="top", fontsize=9, color="purple")
    ax.set_title(f"{line.label} in {cube.name}", fontsize=12)
    ax.set_xlabel(r"Rest wavelength [$\mu$m]")
    ax.set_ylabel("Flux density [mJy]")
    ax.legend(fontsize=9, loc="upper left")
    return ax.figure


def plot_dispersion_histograms(line_maps: Dict[str, LineMaps], sigmas: Optional[Dict] = None,
                               sigma_range=(15, 600), bins: int = 60, ncols: int = 5,
                               xlim: Optional[Tuple[float, float]] = (0, 350),
                               min_ratio: Optional[float] = None):
    """Paper Fig. S3: disc (narrow, purple) and outflow (broad, green) dispersions.
    ``min_ratio`` excludes single-component spaxels (see :func:`dispersion_samples`)."""
    lms = _sorted(line_maps)
    fig, axes = _grid(len(lms), ncols, size=3.2)
    edges = np.linspace(*sigma_range, bins)
    for ax, lm in zip(axes, lms):
        sd, so = dispersion_samples(lm, sigma_range, min_ratio)
        ax.hist(sd, edges, color="darkviolet", alpha=0.65)
        ax.hist(so, edges, color="limegreen", alpha=0.65)
        if sigmas and lm.name in sigmas:
            for key, col, ypos in (("disc", "darkviolet", 0.95), ("outflow", "green", 0.83)):
                val = sigmas[lm.name][key]
                if np.isfinite(val):
                    ax.axvline(val, color=col, ls="--", lw=1.5)
                    ax.text(0.96, ypos, f"{val:.0f} km s$^{{-1}}$", transform=ax.transAxes, ha="right",
                            va="top", color=col, fontsize=11)
        ax.set_title(f"{lm.label} ({lm.ip:g} eV)", fontsize=12)
        if xlim:
            ax.set_xlim(*xlim)
    fig.supxlabel(r"Velocity dispersion [km s$^{-1}$]")
    fig.supylabel("Number of spaxels")
    return fig


# ----------------------------------------------------------------------
# Step 2
# ----------------------------------------------------------------------
def plot_velocity_panels(maps: Dict[str, np.ndarray], extent, vlim: float = 600, ncols: int = 3,
                         titles: Optional[Dict[str, str]] = None):
    """Any set of velocity maps on the same grid, with one shared colour bar."""
    fig, axes = _grid(len(maps), ncols, size=3.8)
    im = None
    for ax, (k, m) in zip(axes, maps.items()):
        im = ax.imshow(m, extent=extent, cmap=VEL_CMAP, vmin=-vlim, vmax=vlim)
        _agn(ax)
        ax.set_title((titles or {}).get(k, k), fontsize=12)
    style_colorbar(fig.colorbar(im, ax=axes[:len(maps)], shrink=0.6), r"Velocity [km s$^{-1}$]")
    _map_labels(fig)
    return fig


def plot_kinematic_fit(fit, lm: LineMaps, vlim: float = 600):
    """Paper Fig. S4: observed, model, residual (+ disc and outflow components)."""
    maps = fit.model_maps(lm)
    titles = {"observed": f"Observed velocity {lm.label}", "model": f"Best-fit {fit.component}",
              "residual": "Residuals", "disc": "Disc component", "outflow": "Outflow component"}
    keys = ["observed", "model", "residual"] + (["disc", "outflow"] if fit.component == "combined" else [])
    return plot_velocity_panels({k: maps[k] for k in keys}, lm.extent_pc(), vlim, titles=titles)


def plot_params_vs_ip(fits: Dict, keys: Sequence[str], relations: Optional[Dict] = None,
                      style: str = "box", annotate: bool = True):
    """Best-fit parameters of several fits against IP.

    style : ``"box"`` - box-and-whisker from the posterior samples (box: 25-75th
        percentile, line: median, whiskers: 16-84th percentile); ``"errorbar"`` -
        median with 16-84th percentile error bars. Least-squares fits (no samples)
        are always drawn as error bars.
    relations : ``{param: f(IP)}`` lines to overplot (e.g. a trend you adopt).
    """
    fs = sorted(fits.values(), key=lambda f: f.ip)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.9), constrained_layout=True,
                             squeeze=False)
    for ax, k in zip(axes[0], keys):
        style_axes(ax)
        sel = [f for f in fs if k in f.free]
        ips = np.array([f.ip for f in sel])
        for f in sel:
            col = f.free.index(k)
            if style == "box" and f.samples is not None:
                s = f.samples[:, col]
                p = np.percentile(s, [16, 25, 50, 75, 84])
                ax.bxp([dict(med=p[2], q1=p[1], q3=p[3], whislo=p[0], whishi=p[4], fliers=[])],
                       positions=[f.ip], widths=2.5, showfliers=False, manage_ticks=False,
                       boxprops=dict(color="C0", lw=1.3), medianprops=dict(color="k", lw=1.6),
                       whiskerprops=dict(color="C0", lw=1.2), capprops=dict(color="C0", lw=1.2))
            else:
                lo, hi = f.errors[k]
                ax.errorbar(f.ip, f.params[k], yerr=[[lo], [hi]], fmt="o", color="C0", capsize=3)
            if annotate:
                ax.annotate(f.label, (f.ip, f.params[k]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        if relations and k in relations and len(ips):
            xx = np.linspace(0, ips.max() * 1.05, 100)
            ax.plot(xx, [relations[k](x) for x in xx], "r-", lw=1.5)
        ax.set_xlabel("IP [eV]")
        ax.set_ylabel(k)
        if len(ips):
            ax.set_xlim(0, ips.max() * 1.08)
    return fig


# ----------------------------------------------------------------------
# Step 3
# ----------------------------------------------------------------------
def plot_decomposition_spaxels(dec, lm: LineMaps, spaxels=None, n: int = 9, seed: int = 0):
    """Constrained fits in a few spaxels: data, disc (blue), outflow (red), total."""
    from .utils import gaussian
    if spaxels is None:
        ok = np.argwhere(np.isfinite(dec.f_disc))
        rng = np.random.default_rng(seed)
        spaxels = [tuple(p) for p in ok[rng.choice(len(ok), size=min(n, len(ok)), replace=False)]]
    fig, axes = _grid(len(spaxels), 3, size=3.4)
    v = dec.vel_axis if getattr(dec, "vel_axis", None) is not None else lm.vel_axis
    y = dec.contsub if getattr(dec, "contsub", None) is not None else lm.contsub
    for ax, (i, j) in zip(axes, spaxels):
        d = gaussian(v, dec.A_disc[i, j], dec.v_disc[i, j], dec.S_disc[i, j])
        o = gaussian(v, dec.A_out[i, j], dec.v_out[i, j], dec.S_out[i, j])
        ax.step(v, y[:, i, j] * 1e3, where="mid", color="k", lw=1.1)
        ax.plot(v, d * 1e3, color="royalblue", label="disc")
        ax.plot(v, o * 1e3, color="crimson", label="outflow")
        ax.plot(v, (d + o) * 1e3, "--", color="orange", label="total")
        ax.set_title(f"({i}, {j})  $f_{{\\rm disc}}$ = {dec.f_disc[i, j]:.2f}", fontsize=11)
        ax.set_xlim(-1500, 1500)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"{dec.label}: constrained decomposition")
    fig.supxlabel(r"Velocity [km s$^{-1}$]")
    fig.supylabel("Flux density [mJy]")
    return fig


def plot_fraction_maps(decomps: Dict, which: str = "f_disc", ncols: int = 5):
    """Paper Fig. S6: per-line disc (or outflow, ``which="f_out"``) fraction maps."""
    ds = _sorted(decomps)
    fig, axes = _grid(len(ds), ncols, size=3.2)
    cm = fraction_cmap(which)
    im = None
    for ax, d in zip(axes, ds):
        im = ax.imshow(getattr(d, which), extent=d.extent_pc(), cmap=cm, vmin=0, vmax=1)
        _agn(ax, "w")
        ax.set_title(f"{d.label} ({d.ip:g} eV)", fontsize=12)
    style_colorbar(fig.colorbar(im, ax=axes[:len(ds)], shrink=0.6),
                   "Disc flux fraction" if which == "f_disc" else "AGN outflow flux fraction")
    _map_labels(fig)
    return fig


def plot_residual_maps(decomps: Dict, ncols: int = 5, vlim: float = 0.5):
    """Fractional residual flux per spaxel - coherent structure hints at an unmodelled
    component (paper Sect. 4.2.1)."""
    ds = _sorted(decomps)
    fig, axes = _grid(len(ds), ncols, size=3.2)
    im = None
    for ax, d in zip(axes, ds):
        im = ax.imshow(d.residual_frac, extent=d.extent_pc(), cmap="PuOr", vmin=-vlim, vmax=vlim)
        _agn(ax)
        ax.set_title(d.label, fontsize=12)
    style_colorbar(fig.colorbar(im, ax=axes[:len(ds)], shrink=0.6), "Residual / data flux")
    _map_labels(fig)
    return fig


# ----------------------------------------------------------------------
# Step 4
# ----------------------------------------------------------------------
def plot_stacked_map(stack, which: str = "f_out", lim_pc: Optional[float] = None, ax=None,
                     title: Optional[str] = None, units: str = "pc", pc_per_arcsec: Optional[float] = None,
                     scalebar_pc: Optional[float] = None):
    """Paper Fig. 2: the stacked AGN-outflow (``f_out``) or disc (``f_disc``) fraction.

    units : ``"pc"`` or ``"arcsec"`` (then give ``pc_per_arcsec``).
    scalebar_pc : draw a scale bar of this length (pc) in the lower right.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    style_axes(ax)
    f = 1.0 if units == "pc" else 1.0 / pc_per_arcsec
    ext = [e * f for e in stack.extent]
    im = ax.imshow(getattr(stack, which), extent=ext, cmap=fraction_cmap(which), vmin=0, vmax=1)
    _agn(ax, "w")
    if lim_pc:
        ax.set_xlim(-lim_pc * f, lim_pc * f)
        ax.set_ylim(-lim_pc * f, lim_pc * f)
    lab = "[pc]" if units == "pc" else "[arcsec]"
    ax.set_xlabel(r"$\Delta x$ " + lab if units == "pc" else "RA offset " + lab)
    ax.set_ylabel(r"$\Delta y$ " + lab if units == "pc" else "Dec offset " + lab)
    if scalebar_pc:
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        xe = x2 - 0.06 * (x2 - x1)
        ye = y1 + 0.07 * (y2 - y1)
        ax.plot([xe - scalebar_pc * f, xe], [ye, ye], color="k", lw=3)
        ax.text(xe - scalebar_pc * f / 2, ye + 0.02 * (y2 - y1), f"{scalebar_pc:g} pc", ha="center", va="bottom")
    if title:
        ax.set_title(title)
    style_colorbar(plt.colorbar(im, ax=ax, pad=0.02),
                   "AGN outflow fraction" if which == "f_out" else "Disc fraction")
    return ax.figure


# ----------------------------------------------------------------------
# Step 5
# ----------------------------------------------------------------------
def plot_extraction_regions(spec, cube):
    """Continuum image of ``cube`` with the AGN (cyan) and SF (red) spaxels marked."""
    img = cube.continuum_image()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax in axes:
        style_axes(ax)
    pos = img[img > 0]
    axes[0].imshow(np.where(img > 0, img, np.nan), cmap=FLUX_CMAP,
                   norm=LogNorm(*np.percentile(pos, [1, 99.5])) if pos.size else None)
    if spec.agn_mask is not None:
        for m, c, lab in ((spec.agn_mask, "cyan", "AGN"), (spec.sf_mask, "red", "SF")):
            yy, xx = np.where(m)
            axes[0].scatter(xx, yy, marker="x", color=c, s=14, label=f"{lab} ({m.sum()})")
        axes[0].legend(fontsize=10, frameon=True)
    axes[0].plot(*_continuum_peak(cube), "x", color="w", ms=10, mew=2.2)
    axes[0].set_title(f"{cube.name}: extraction regions")
    fo = 1 - spec.f_disc_map
    if getattr(spec, "footprint", None) is not None:
        fo = np.where(spec.footprint, fo, np.nan)        # only this cube's observed field
        for ax in axes:
            ax.contour(spec.footprint.astype(float), levels=[0.5], colors="k", linewidths=1.0)
    im = axes[1].imshow(fo, cmap=fraction_cmap("f_out"), vmin=0, vmax=1)
    axes[1].set_title(r"Stacked $f_{\rm out}$ within this cube's field")
    style_colorbar(plt.colorbar(im, ax=axes[1]))
    for ax in axes:
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
    return fig


def plot_component_spectra(wave_agn, flux_agn, wave_sf, flux_sf, smooth: float = 0, sf_scale: float = 1.0,
                           ax=None, xlim=None, err_agn=None, err_sf=None, err_scale: float = 1.0,
                           err_alpha: float = 0.3):
    """AGN and star-forming spectra on a log scale (optionally Gaussian-smoothed).

    err_agn, err_sf : 1-sigma errors, drawn as shaded bands (flux +/- err_scale x error). Errors of
        bright spectra are often too small to see at full scale - ``err_scale`` exaggerates them
        for display only (it is noted in the legend).
    """
    from scipy.ndimage import gaussian_filter1d
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    style_axes(ax)
    fa = gaussian_filter1d(flux_agn, smooth) if smooth else flux_agn
    fs = gaussian_filter1d(flux_sf, smooth) if smooth else flux_sf
    for w, f, e, col, sc in ((wave_agn, fa, err_agn, "cornflowerblue", 1.0), (wave_sf, fs, err_sf, "crimson", sf_scale)):
        if e is not None:
            e = np.asarray(e, float) * err_scale * sc
            lo = np.clip(f * sc - e, np.nanmin(f[f > 0]) * sc * 1e-2 if np.any(f > 0) else 1e-30, None)
            ax.fill_between(w, lo, f * sc + e, color=col, alpha=err_alpha, lw=0,
                            label=f"1$\\sigma$ error" + (f" ($\\times${err_scale:g})" if err_scale != 1 else "")
                            if col == "cornflowerblue" else None)
    ax.plot(wave_agn, fa, color="cornflowerblue", lw=1.6, label="AGN-dominated")
    ax.plot(wave_sf, fs * sf_scale, color="crimson", lw=1.6,
            label="Star-formation-dominated" + (f" $\\times${sf_scale:g}" if sf_scale != 1 else ""))
    ax.set_yscale("log")
    ax.set_xlabel(r"Rest-frame wavelength [$\mu$m]")
    ax.set_ylabel("Flux density [Jy]")
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend()
    return ax.figure


def plot_continuum_decomposition(res: Dict, cube, title: str = "continuum"):
    """Paper Fig. S7: total / disc / outflow continuum maps."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    pos = res["total"][np.isfinite(res["total"]) & (res["total"] > 0)]
    norm = LogNorm(*np.percentile(pos, [1, 99.5]))
    im = None
    for ax, k in zip(axes, ("total", "disc", "outflow")):
        style_axes(ax)
        im = ax.imshow(np.where(res[k] > 0, res[k], np.nan), cmap=FLUX_CMAP, norm=norm)
        ax.plot(*cube.agn_xy, "x", color="k", ms=9, mew=2)
        ax.set_title(f"{title}: {k}")
        ax.set_xlabel("x [pixel]")
    axes[0].set_ylabel("y [pixel]")
    style_colorbar(plt.colorbar(im, ax=list(axes), shrink=0.85), "Flux density [Jy pixel$^{-1}$]")
    return fig


def plot_truth_comparison(decomps: Dict, truth: Dict, ncols: int = 4):
    """(Mock data only) recovered minus true disc fraction for each line."""
    ds = _sorted(decomps)
    fig, axes = _grid(len(ds), ncols, size=3.2)
    im = None
    for ax, d in zip(axes, ds):
        im = ax.imshow(d.f_disc - truth["f_disc"][d.line_name], extent=d.extent_pc(), cmap="PuOr",
                       vmin=-0.2, vmax=0.2)
        _agn(ax)
        ax.set_title(d.label, fontsize=12)
    style_colorbar(fig.colorbar(im, ax=axes[:len(ds)], shrink=0.6),
                   r"$f_{\rm disc}$ recovered $-$ true")
    _map_labels(fig)
    return fig

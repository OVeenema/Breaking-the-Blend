"""
Emission-line catalogue and per-line settings.

Every line used in the analysis is described by a :class:`LineSpec`. The
catalogue below holds sensible defaults for the mid-IR / near-IR ionic lines
used in Veenema et al. (2026). Copy a line and change any setting with
``line.update(...)``, or create a new ``LineSpec`` for a line not listed.

Wavelengths are rest-frame microns. Ionisation potentials (IP, eV) are the
energy needed to *reach* the ionisation stage producing the transition.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple, Dict, Iterable

C_KMS = 299792.458  # speed of light [km/s]


@dataclass(frozen=True)
class LineSpec:
    """Settings for one emission line.

    Parameters
    ----------
    name : short identifier used as a dictionary key (e.g. ``"NeII"``).
    label : pretty label for plots (e.g. ``"[Ne II]"``).
    wave : rest wavelength [micron].
    ip : ionisation potential [eV].
    cube : name of the cube to measure the line in (e.g. ``"ch3-short"``).
        ``None`` = choose automatically from the loaded cubes.
    line_region : (lo, hi) rest wavelength range [micron] containing the line.
        Excluded from the continuum fit and integrated for the flux.
        ``None`` = ``wave`` +/- ``line_halfwidth_kms``.
    fit_window : (lo, hi) rest wavelength range [micron] cut out of the cube
        (line + continuum). ``None`` = ``line_region`` +/- ``cont_width_um``.
    cont_width_um : continuum on each side of ``line_region`` [micron]. ``None`` = 0.02 micron
        (as in the paper), or half the width of ``line_region`` if that is larger (long
        wavelengths, where 0.02 micron is only a few channels).
    cont_order : polynomial order of the local continuum (paper: 3).
    p0_sigma_kms : initial (narrow, broad) dispersions for the double-Gaussian fit.
    p0_dv_kms : initial (narrow, broad) velocity offsets for the double-Gaussian fit.
    p0_amp_frac : initial (narrow, broad) amplitudes as a fraction of the peak.
    sigma_bounds_kms : allowed dispersion range in the Gaussian fits.
    vel_bounds_kms : allowed centroid velocity range in the Gaussian fits.
    mask_percentile : spaxels with line flux below this percentile are masked
        (paper: 60, i.e. keep the brightest 40 per cent).
    """

    name: str
    label: str
    wave: float
    ip: float
    cube: Optional[str] = None
    line_region: Optional[Tuple[float, float]] = None
    fit_window: Optional[Tuple[float, float]] = None
    cont_order: int = 3
    line_halfwidth_kms: float = 1000.0
    cont_width_um: Optional[float] = None
    p0_sigma_kms: Tuple[float, float] = (80.0, 200.0)
    p0_dv_kms: Tuple[float, float] = (0.0, 0.0)
    p0_amp_frac: Tuple[float, float] = (0.6, 0.3)
    sigma_bounds_kms: Tuple[float, float] = (10.0, 1000.0)
    vel_bounds_kms: Tuple[float, float] = (-1500.0, 1500.0)
    mask_percentile: float = 60.0

    # ------------------------------------------------------------------
    def update(self, **kwargs) -> "LineSpec":
        """Return a copy with some settings changed, e.g. ``line.update(cube="ch3-short")``."""
        return replace(self, **kwargs)

    def get_line_region(self) -> Tuple[float, float]:
        if self.line_region is not None:
            return tuple(self.line_region)
        dl = self.wave * self.line_halfwidth_kms / C_KMS
        return (self.wave - dl, self.wave + dl)

    def get_fit_window(self) -> Tuple[float, float]:
        if self.fit_window is not None:
            return tuple(self.fit_window)
        lo, hi = self.get_line_region()
        cw = self.cont_width_um if self.cont_width_um is not None else max(0.02, 0.5 * (hi - lo))
        return (lo - cw, hi + cw)

    def __str__(self) -> str:  # nicer printing in notebooks
        return f"{self.label:<9s} {self.wave:8.4f} um   IP = {self.ip:6.1f} eV   cube = {self.cube}"


# ----------------------------------------------------------------------
# Default catalogue (wavelengths as adopted in the NGC 5728 analysis)
# ----------------------------------------------------------------------
_CATALOGUE = [
    LineSpec("NiII",     "[Ni II]",      6.6360,   7.6),
    LineSpec("FeII",     "[Fe II]",      5.3390,   7.9, p0_sigma_kms=(110, 225)),
    LineSpec("ClII",     "[Cl II]",     14.3678,  13.0),
    LineSpec("Pfa",      "Pf$\\alpha$",  7.4599,  13.6),
    LineSpec("ArII",     "[Ar II]",      6.9845,  15.8, p0_sigma_kms=(85, 215)),
    LineSpec("NeII",     "[Ne II]",     12.8140,  21.6, p0_sigma_kms=(105, 165)),
    LineSpec("SIII",     "[S III]",     18.7120,  23.3, p0_sigma_kms=(110, 150)),
    LineSpec("ArIII",    "[Ar III]",     8.9905,  27.6, p0_sigma_kms=(100, 200)),
    LineSpec("SIV",      "[S IV]",      10.5100,  34.9, p0_sigma_kms=(85, 255)),
    LineSpec("NeIII",    "[Ne III]",    15.5580,  41.0, p0_sigma_kms=(105, 175)),
    LineSpec("NaIII",    "[Na III]",     7.3177,  47.3),
    LineSpec("OIV",      "[O IV]",      25.8900,  54.9, p0_sigma_kms=(80, 130)),
    LineSpec("ArV_7.90", "[Ar V] 7.90",  7.9016,  59.6),
    LineSpec("ArV",      "[Ar V]",      13.1040,  59.6, p0_sigma_kms=(90, 160)),
    LineSpec("ArVI",     "[Ar VI]",      4.5300,  74.8, p0_sigma_kms=(65, 165)),
    LineSpec("MgIV",     "[Mg IV]",      4.4870,  80.1, p0_sigma_kms=(135, 265)),
    LineSpec("NeV",      "[Ne V]",      14.3200,  97.2),
    LineSpec("MgV",      "[Mg V]",       5.6105, 109.2),
    LineSpec("NeVI",     "[Ne VI]",      7.6524, 126.2, p0_sigma_kms=(80, 175)),
]

LINE_CATALOGUE: Dict[str, LineSpec] = {l.name: l for l in _CATALOGUE}


def get_lines(names: Optional[Iterable[str]] = None, sort_by_ip: bool = True) -> Dict[str, LineSpec]:
    """Return a dict ``{name: LineSpec}`` from the catalogue.

    ``names=None`` returns the full catalogue. The dict is ordered by IP
    (unless ``sort_by_ip=False``) so loops and plots run from low to high IP.
    """
    if names is None:
        names = list(LINE_CATALOGUE)
    names = list(names)
    missing = [n for n in names if n not in LINE_CATALOGUE]
    if missing:
        raise KeyError(f"Unknown line(s) {missing}. Available: {list(LINE_CATALOGUE)}. "
                       "Create a new LineSpec for lines that are not in the catalogue.")
    lines = [LINE_CATALOGUE[n] for n in names]
    if sort_by_ip:
        lines = sorted(lines, key=lambda l: l.ip)
    return {l.name: l for l in lines}


def ip_of(obj) -> float:
    """IP of a LineSpec or of any result object carrying ``.line``/``.ip``."""
    if hasattr(obj, "ip"):
        return obj.ip
    return obj.line.ip


def sort_by_ip(objs):
    """Sort a dict (or list) of results by the IP of their line."""
    if isinstance(objs, dict):
        return dict(sorted(objs.items(), key=lambda kv: ip_of(kv[1])))
    return sorted(objs, key=ip_of)

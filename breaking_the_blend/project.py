"""
One :class:`Project` per galaxy / analysis.

It knows

* the galaxy name and redshift (or distance),
* the folder holding the observed cubes (searched recursively for ``*s3d.fits*``),
* a results folder ``<results_dir>/<name>/`` where every step is check-pointed,
  figures are written and the analysis settings are recorded in ``config.json``.

So the same notebook can be re-run for many targets by changing three lines,
and a slow step (line fitting, MCMC) never has to be repeated::

    proj = btb.Project("NGC5728", data_dir="/data/NGC5728/JWST", z=0.00935)
    cubes = proj.load_cubes()
    line_maps = proj.checkpoint("step1_line_maps", btb.fit_lines, cubes, lines)
    proj.savefig(fig, "flux_maps")

Re-opening an existing analysis only needs the name (and results_dir)::

    proj = btb.Project.open("NGC5728")
    line_maps = proj.load("step1_line_maps")
"""
from __future__ import annotations

import json
import os
import re
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from .cubes import Cube, guess_cube_name, load_cube, set_agn_positions
from .utils import save as _save, load as _load


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name.strip())


def _jsonable(obj):
    """Convert settings (tuples, numpy scalars, LineSpecs...) to JSON-friendly types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if callable(obj):
        return getattr(obj, "__name__", repr(obj))
    return obj


class Project:
    """Book-keeping for one galaxy / analysis (see module docstring).

    Parameters
    ----------
    name : galaxy or analysis name, e.g. ``"NGC5728"`` or ``"NGC5728_neon_only"``.
        Used as the results sub-folder.
    data_dir : folder containing the ``s3d`` cubes (any sub-folder structure, e.g. a MAST download).
    z : redshift (de-redshifting and, without ``distance_mpc``, the physical scale).
    distance_mpc : optional distance [Mpc] for the physical scale.
    results_dir : parent folder for all analyses (default ``./results``).
    save_figures : what :meth:`savefig` writes: ``"png"``, ``"pdf"``, ``"both"`` or ``"none"``
        (or a list of formats, e.g. ``["png", "svg"]``). Change later with :meth:`set_figure_output`.
    figure_dpi : resolution of saved raster figures.
    """

    def __init__(self, name: str, data_dir: Optional[str] = None, z: Optional[float] = None,
                 distance_mpc: Optional[float] = None, results_dir: str = "results",
                 save_figures="both", figure_dpi: int = 200):
        self.name = name
        self.path = Path(results_dir).expanduser() / _safe(name)
        (self.path / "figures").mkdir(parents=True, exist_ok=True)
        old = self.config() if (self.path / "config.json").exists() else {}
        # values not given now are taken from an earlier run of the same analysis
        data_dir = data_dir if data_dir is not None else old.get("data_dir")
        self.data_dir = None if data_dir is None else str(Path(data_dir).expanduser())
        self.z = z if z is not None else old.get("z")
        self.distance_mpc = distance_mpc if distance_mpc is not None else old.get("distance_mpc")
        self.set_figure_output(save_figures, figure_dpi)
        self.cube_files: Dict[str, str] = dict(old.get("cube_files", {}))
        self.cubes: Dict[str, Cube] = {}
        self.save_config(name=name, data_dir=self.data_dir, z=self.z, distance_mpc=self.distance_mpc)
        print(f"Project '{name}': results in {self.path.resolve()}")

    @classmethod
    def open(cls, name: str, results_dir: str = "results") -> "Project":
        """Re-open an existing analysis from its ``config.json``."""
        cfg_file = Path(results_dir).expanduser() / _safe(name) / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"No analysis '{name}' in {results_dir} ({cfg_file} missing)")
        return cls(name, results_dir=results_dir)

    # ------------------------------------------------------------------
    # Cubes
    # ------------------------------------------------------------------
    def find_cubes(self, pattern: str = "*s3d.fits*", include: Optional[Iterable[str]] = None,
                   exclude: Optional[Iterable[str]] = None, verbose: bool = True) -> Dict[str, str]:
        """Search ``data_dir`` (recursively) for cubes and name them by band.

        include / exclude : lists of glob patterns matched against the file path
        (e.g. ``exclude=["*bkg*"]``). If two files give the same band name the
        first (alphabetically) is kept and a warning printed - use ``exclude`` or
        :meth:`set_cube_files` to choose.
        """
        if self.data_dir is None:
            raise ValueError("This project has no data_dir")
        root = Path(self.data_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"data_dir '{root}' does not exist")
        files = sorted(str(p) for p in root.rglob(pattern) if p.is_file())
        if include:
            files = [f for f in files if any(fnmatch(f, pat) for pat in include)]
        if exclude:
            files = [f for f in files if not any(fnmatch(f, pat) for pat in exclude)]
        found: Dict[str, str] = {}
        for f in files:
            n = guess_cube_name(f)
            if n in found:
                print(f"  WARNING: '{n}' matches several files; keeping {os.path.basename(found[n])}, "
                      f"ignoring {os.path.basename(f)}")
                continue
            found[n] = f
        if not found:
            raise FileNotFoundError(f"No files matching '{pattern}' under {root}")
        self.cube_files = found
        if verbose:
            print(f"Found {len(found)} cubes in {root}:")
            for n, f in found.items():
                print(f"  {n:>14s}: {os.path.relpath(f, root)}")
        self.save_config(cube_files=found)
        return found

    def set_cube_files(self, files: Dict[str, str]) -> None:
        """Give the cube files by hand: ``{name: path}`` (paths may be relative to data_dir)."""
        root = Path(self.data_dir) if self.data_dir else Path(".")
        self.cube_files = {n: str(p if Path(p).is_absolute() else root / p) for n, p in files.items()}
        self.save_config(cube_files=self.cube_files)

    def load_cubes(self, names: Optional[Iterable[str]] = None, load_err: bool = False,
                   agn_overrides: Optional[Dict[str, tuple]] = None,
                   pixel_scale_overrides: Optional[Dict[str, float]] = None) -> Dict[str, Cube]:
        """Load the cubes (all found ones, or ``names``) and set their AGN positions.

        agn_overrides : ``{cube: (x, y)}`` pixel positions where the continuum peak is wrong.
        pixel_scale_overrides : ``{cube: arcsec}`` to replace the header pixel size
            (normally not needed - the header value is correct for pipeline cubes).
        """
        if self.z is None:
            raise ValueError("Set the redshift: Project(..., z=...)")
        if not self.cube_files:
            self.find_cubes()
        names = list(names) if names is not None else list(self.cube_files)
        for n in names:
            if n not in self.cubes:
                self.cubes[n] = load_cube(self.cube_files[n], self.z, name=n,
                                          distance_mpc=self.distance_mpc, load_err=load_err)
                print(self.cubes[n])
        cubes = {n: self.cubes[n] for n in names}
        for n, ps in (pixel_scale_overrides or {}).items():
            if n in cubes:
                print(f"  {n}: pixel scale {cubes[n].pixel_scale:.3f}\" (header) -> {ps:.3f}\" (override)")
                cubes[n].pixel_scale = float(ps)
        set_agn_positions(cubes, overrides=agn_overrides)
        self.save_config(pixel_scales={n: c.pixel_scale for n, c in cubes.items()})
        self.save_config(agn_positions={n: c.agn_xy for n, c in cubes.items()},
                         agn_overrides=agn_overrides or {})
        return cubes

    # ------------------------------------------------------------------
    # Saving / check-pointing
    # ------------------------------------------------------------------
    def file(self, key: str, ext: str = ".pkl") -> Path:
        return self.path / f"{key}{ext}"

    def exists(self, key: str) -> bool:
        return self.file(key).exists()

    def save(self, obj: Any, key: str) -> None:
        """Pickle a result as ``<results>/<name>/<key>.pkl``."""
        _save(obj, str(self.file(key)))

    def load(self, key: str) -> Any:
        if not self.exists(key):
            raise FileNotFoundError(f"'{key}' not saved yet for {self.name} ({self.file(key)})")
        return _load(str(self.file(key)))

    def checkpoint(self, key: str, func: Callable, *args, rerun: bool = False,
                   settings: Optional[dict] = None, **kwargs) -> Any:
        """Return the saved result ``key`` if it exists, otherwise run
        ``func(*args, **kwargs)``, save and return it.

        rerun : force a re-run.
        settings : small dict describing the inputs (line settings, priors, ...). It is
            stored next to the result; if it differs next time, the step is re-run
            automatically so a stale result is never re-used by mistake.
        """
        sfile = self.file(key, ".settings.json")
        new = json.dumps(_jsonable(settings), sort_keys=True) if settings is not None else None
        if self.exists(key) and not rerun:
            old = sfile.read_text() if sfile.exists() else None
            if new is None or new == old:
                print(f"[{self.name}] loading saved '{key}' (rerun=True to recompute)")
                return self.load(key)
            print(f"[{self.name}] settings of '{key}' changed -> recomputing")
        t0 = time.time()
        out = func(*args, **kwargs)
        print(f"[{self.name}] '{key}' done in {time.time() - t0:.0f} s")
        self.save(out, key)
        if new is not None:
            sfile.write_text(new)
        return out

    def set_figure_output(self, save_figures="both", dpi: Optional[int] = None) -> None:
        """Choose which figure files :meth:`savefig` writes: ``"png"``, ``"pdf"``, ``"both"``,
        ``"none"``, or a list of matplotlib formats."""
        choice = {"both": ("png", "pdf"), "png": ("png",), "pdf": ("pdf",), "none": (), None: (),
                  False: (), True: ("png", "pdf")}
        if isinstance(save_figures, (list, tuple)):
            self.figure_formats = tuple(save_figures)
        elif save_figures in choice:
            self.figure_formats = choice[save_figures]
        else:
            raise ValueError("save_figures must be 'png', 'pdf', 'both', 'none' or a list of formats")
        if dpi is not None:
            self.figure_dpi = int(dpi)

    def savefig(self, fig, name: str, formats=None, dpi: Optional[int] = None) -> None:
        """Save a figure to ``<results>/<name>/figures/`` in the formats chosen with
        ``save_figures`` (nothing is written if that is ``"none"``)."""
        formats = self.figure_formats if formats is None else formats
        for fmt in formats:
            fig.savefig(self.path / "figures" / f"{name}.{fmt}", dpi=dpi or self.figure_dpi, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Settings record
    # ------------------------------------------------------------------
    def save_config(self, **settings) -> None:
        """Merge ``settings`` into ``config.json`` (a human-readable record of the run)."""
        f = self.path / "config.json"
        cfg = json.loads(f.read_text()) if f.exists() else {}
        cfg.update(_jsonable(settings))
        cfg["last_modified"] = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write_text(json.dumps(cfg, indent=2))

    def config(self) -> dict:
        return json.loads((self.path / "config.json").read_text())

    def summary(self) -> None:
        """List what has been saved for this analysis."""
        print(f"{self.name}: {self.path}")
        for p in sorted(self.path.glob("*")):
            if p.is_file():
                print(f"  {p.name:<40s} {p.stat().st_size / 1e6:8.1f} MB")
        nfig = len(list((self.path / "figures").glob("*")))
        print(f"  figures/ ({nfig} files)")

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def export_stacked_map(self, stack, key: str = "stacked_fraction_map") -> None:
        """Write the final map as ``.npz`` (always) and ``.fits`` (if astropy is installed)."""
        np.savez(self.file(key, ".npz"), x_pc=stack.x_pc, y_pc=stack.y_pc, f_disc=stack.f_disc,
                 f_out=stack.f_out, n_lines=stack.n_lines)
        try:
            from astropy.io import fits
            hdr = fits.Header()
            hdr["OBJECT"] = self.name
            for ax, arr in ((1, stack.x_pc), (2, stack.y_pc)):
                hdr[f"CTYPE{ax}"] = ("OFFSET", "offset from AGN")
                hdr[f"CUNIT{ax}"] = "pc"
                hdr[f"CRPIX{ax}"] = 1.0
                hdr[f"CRVAL{ax}"] = float(arr[0])
                hdr[f"CDELT{ax}"] = float(arr[1] - arr[0])
            hdr["SCHEME"] = stack.scheme
            hdr["COMMENT"] = "PRIMARY: AGN outflow fraction; F_DISC: disc fraction;"
            hdr["COMMENT"] = "N_LINES: number of emission lines contributing to each pixel"
            fits.HDUList([fits.PrimaryHDU(stack.f_out.astype(np.float32), hdr),
                          fits.ImageHDU(stack.f_disc.astype(np.float32), hdr, name="F_DISC"),
                          fits.ImageHDU(stack.n_lines.astype(np.int16), hdr, name="N_LINES")]
                         ).writeto(self.file(key, ".fits"), overwrite=True)
        except ImportError:
            pass
        print(f"exported -> {self.file(key, '.npz')}")

    def export_spectra(self, spectra: Dict[str, tuple], key: str = "component_spectra") -> None:
        """Write spectra ``{label: (wave, flux)}`` to one text file per label."""
        for lab, (w, f) in spectra.items():
            out = self.file(f"{key}_{_safe(lab)}", ".txt")
            np.savetxt(out, np.column_stack([w, f]), header="rest_wavelength_um  flux_Jy")
            print(f"exported -> {out}")

    def __repr__(self):
        return f"Project('{self.name}', data_dir='{self.data_dir}', z={self.z}, results={self.path})"


def list_projects(results_dir: str = "results") -> List[str]:
    """Names of all analyses saved under ``results_dir``."""
    root = Path(results_dir)
    return sorted(p.name for p in root.glob("*") if (p / "config.json").exists())

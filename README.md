# Breaking the Blend

Multi-tracer kinematic decomposition of integral-field spectroscopy into a **star-forming disc** and an
**AGN-driven outflow**, spaxel by spaxel — the method of
*Veenema et al. (2026), "Breaking the Blend: A multi-tracer kinematic decomposition method for IFS data
applied to disentangling the AGN outflow and circumnuclear ring in NGC 5728 with JWST"* (arXiv:2609.30232).
doi:[10.1093/mnras/stag1917](https://doi.org/10.1093/mnras/stag1917)

Written for JWST NIRSpec IFU and MIRI/MRS cubes (either or both), but nothing is instrument-specific:
any IFS cube with emission lines tracing both components can be used.

## What it is for

Integral-field spectra of active galaxies usually mix emission from several physical components in every spaxel -
here a star-forming disc/ring and an AGN-driven outflow. Breaking the Blend separates them: it models the velocity
field of each component, fixes each component's velocity and dispersion in every spaxel, fits only their amplitudes,
and combines many emission lines into one map of the fraction of emission from each component. From that map it
builds component-specific spectra. It was developed for JWST NIRSpec and MIRI/MRS cubes but works with any IFS data
in which emission lines trace both components.

## Quick start

**1. Get the code.** From GitHub,

```bash
git clone https://github.com/OVeenema/Breaking-the-Blend.git
cd Breaking-the-Blend
```

(or use the green **Code -> Download ZIP** button on the repository page and unzip it).

**2. Install** (needs [conda](https://docs.conda.io/en/latest/miniconda.html) or mamba):

```bash
conda env create -f environment.yml
conda activate breaking-the-blend
python -m ipykernel install --user --name breaking-the-blend --display-name "Breaking the Blend"
python tests/test_pipeline.py          # should end with "all tests passed"
```

**3. Run the tutorials.** Start `jupyter lab`, open a notebook in `notebooks/` and select the
**"Breaking the Blend"** kernel:

| Notebook | Data needed | Purpose |
|---|---|---|
| `00_quickstart_mock.ipynb` | none (synthetic) | the whole method in a few seconds, checked against a known answer |
| `01_NGC5728.ipynb` | NGC 5728 cubes (below) | fully worked real-data example |
| `02_template_new_galaxy.ipynb` | your cubes | clean template for any galaxy (cells to edit marked **EDIT**) |
| `03_check_line_settings.ipynb` | your cubes | quick check of line windows and fits before a full run |

**4. Data for notebook 01.** Download the public JWST cubes of NGC 5728 (programmes 1670 and 5017) from MAST:

```bash
python scripts/download_mast_cubes.py --target "NGC 5728" --programs 1670 5017 --out data/NGC5728
```

or follow the portal instructions in [`data/NGC5728/README.md`](data/NGC5728/README.md).
Results of every notebook are written to `notebooks/results/<name>/`.

## The method in five steps

| Step | What | Function(s) |
|---|---|---|
| 1 | Continuum-subtract each line, flux map, single- and double-Gaussian fits; median σ of each component | `fit_lines`, `dispersion_medians` |
| 2 | Fit a thin rotating disc, a hollow bicone, or both (α-weighted) to the velocity maps — or skip and give literature values | `fit_kinematics`, `fit_many`, `trend_with_ip` |
| 3 | In every spaxel fix both Gaussians' **velocities** (models) and **widths** (Step 1), fit only the amplitudes → disc fraction per line | `line_params`, `decompose_lines` |
| 4 | Weighted stack of all lines on a common pc grid → final AGN-outflow fraction map | `stack_fraction_maps` |
| 5 | AGN- and star-formation-dominated spectra across all bands (optionally: split a continuum map, `decompose_continuum`) | `extract_all`, `stitch_spectra` |

## Install

Use a dedicated environment so the analysis always runs with the same packages.

**With conda / mamba (recommended):**

```bash
git clone https://github.com/<you>/breaking-the-blend.git
cd breaking-the-blend
conda env create -f environment.yml          # creates "breaking-the-blend" and installs this package (editable)
conda activate breaking-the-blend
python -m ipykernel install --user --name breaking-the-blend --display-name "Breaking the Blend"
python tests/test_pipeline.py                # ~1 min, runs on synthetic data
```

**With venv + pip:**

```bash
cd breaking-the-blend
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
pip install -e ".[extras,test]"
python -m ipykernel install --user --name breaking-the-blend --display-name "Breaking the Blend"
```

Then start `jupyter lab` and pick the **"Breaking the Blend"** kernel in each notebook
(Kernel -> Change kernel). The package is installed in editable mode (`-e`), so edits to the `.py`
files are picked up after restarting the kernel. To update the environment after changing
`environment.yml`: `conda env update -f environment.yml --prune`.

(The notebooks also fall back to the package copy in the repository if it is not installed, but the
dependencies, e.g. astropy and emcee, still need to be present.)

## Getting started

* **`notebooks/00_quickstart_mock.ipynb`** — the whole method in a few minutes on a synthetic galaxy with a known
  answer. No data needed. Start here (and use it for hands-on sessions; it ends with exercises).
* **`notebooks/01_NGC5728.ipynb`** — a fully worked, annotated example on NGC 5728. It adopts several settings of the
  original by-hand analysis (marked in the notebook) so the published maps are reproduced closely.
* **`notebooks/02_template_new_galaxy.ipynb`** — **start here for a new target**: the same workflow with no
  galaxy-specific settings, generic starting values, and the cells to edit clearly marked (**EDIT**).
* **`notebooks/03_check_line_settings.ipynb`** — a fast checker for a new dataset: shows each line's wavelength
  window (line region, continuum fit, neighbouring lines) and fits a handful of spaxels per line, flagging failed or
  degenerate fits. Tune `LINE_SETTINGS` here, then copy them into notebook 02.

Every notebook has a `SAVE_FIGURES` switch at the top: `"png"`, `"pdf"`, `"both"` or `"none"`
(figures go to `results/<name>/figures/`).

## One analysis per target

Everything for a galaxy is handled by a `Project`: where the cubes are, the redshift, and a results folder.

```python
import breaking_the_blend as btb

proj  = btb.Project("NGC5728", data_dir="/data/NGC5728", z=0.00935)   # results -> results/NGC5728/
proj.find_cubes()                       # finds every *s3d.fits* (any sub-folder), names them ch3-short, g395h-f290lp, ...
cubes = proj.load_cubes()               # Jy/pixel, rest-frame wavelengths, AGN position per cube

line_maps = proj.checkpoint("step1_line_maps", btb.fit_lines, cubes, lines,
                            settings=dict(lines=lines))    # computed once, then re-loaded
proj.savefig(fig, "flux_maps")          # -> results/NGC5728/figures/flux_maps.png/.pdf
```

* `results/<name>/config.json` records every setting used (lines, priors, dispersions, AGN positions...).
* `checkpoint(...)` saves each step; if its `settings` change it is **recomputed automatically**.
* Re-open later with `btb.Project.open("NGC5728")`, list all analyses with `btb.list_projects()`.
* To analyse a new galaxy, copy `02_template_new_galaxy.ipynb` and work through the cells marked **EDIT**.
  Different analyses of the same galaxy just use different names (e.g. `"NGC5728_neon_only"`).
* `Project(..., save_figures="both" | "png" | "pdf" | "none")` controls what `proj.savefig` writes.

## Choosing lines and settings

```python
lines = btb.get_lines(["NeII", "NeIII", "NeV"])          # from btb.LINE_CATALOGUE, ordered by IP
lines["NeII"] = lines["NeII"].update(cube="ch3-short",
                                     line_region=(12.800, 12.830),   # rest-frame micron
                                     p0_sigma_kms=(105, 165),        # double-Gaussian initial guesses
                                     mask_percentile=60)             # keep the brightest 40%
my_line = btb.LineSpec("H2_S1", "H$_2$ S(1)", 17.0348, 0.0)          # anything not in the catalogue
```

## Model parameters

One dict is used everywhere. **A number fixes a parameter, a tuple `(lo, hi)` frees it** (uniform prior):

```python
params = dict(inc_disc=43, pa_disc=194, inc_out=48, pa_out=110,    # fixed geometry
              v_disc=(50, 800), R_disc=(20, 1500),                 # free
              v_out=(100, 1500), R_out=(50, 3000), theta_in=(0, 60), theta_out=(30, 120), alpha=(0, 1))
fit = btb.fit_kinematics(line_maps["SIII"], "combined", params, method="emcee")   # or "least_squares"
```

Radii are in **pc**, velocities in km/s, angles in degrees. **Position angles are east of north** (receding side of
the disc / redshifted cone). Maps are stored as in the cube and drawn with `origin="lower"`; for pipeline
`skyalign` cubes (north up, east left) the conversion to the angle from +x is `PA + 90` (`pa_offset`).

## Options worth knowing

* **Double-Gaussian fitting** (`fit_lines`): `fitter="trf"` (bounded, default), `"lm"` or `"capfit"` (unbounded,
  as in the original analysis); `weighting="none" | "err" | "sqrt"`; `p0_center="moment" | "systemic"`.
* **Kinematic fits**: `method="emcee" | "least_squares"`; `sigma_v="residual"` (scatter about the best fit,
  realistic errors) or `"std"` (scatter of the map, as in the paper).
* **Imported models**: `btb.load_external_map(...)` brings in velocity/dispersion maps from 3DBarolo, MOKA3D, etc.
  (give the AGN pixel position and pixel size of that map); pass them to `decompose_lines(..., external={...})`.
* **Decomposition**: per-line or common model parameters, measured or fixed dispersions, optional re-extraction
  of the spectra with other windows (`recut`), fixed or free dispersions (`sigma_mode`).
* **Plots**: `breaking_the_blend.plotting` sets a publication style on import (`bplt.set_style(...)` to change it).
  Fraction maps use cmasher *iceburn*, oriented so star formation is always orange and the AGN outflow
  always blue (disc-fraction maps: high = orange; outflow-fraction maps: high = blue).
* **Dispersions**: `dispersion_medians(..., min_ratio=1.3)` ignores spaxels where the double-Gaussian fit is
  degenerate (a single-component profile gives two Gaussians of nearly the same width).
* **Spectra**: detector gaps are left out, bands with too few selected spaxels are skipped (`min_spaxels`), and
  `stitch_spectra` scales all bands relative to a reference band (default: the one covering the largest sky area).
* **Exported map**: `results/<name>/stacked_fraction_map.fits` holds the AGN outflow fraction (primary), the disc
  fraction (`F_DISC`) and `N_LINES`, the number of lines contributing to each pixel.

## Package layout

```
breaking_the_blend/
  project.py     Project: data folder, results folder, checkpoints, config, exports
  lines.py       LineSpec + catalogue of NIR/MIR ionic lines
  cubes.py       load s3d cubes, pc scale, AGN position, line -> cube assignment
  linefit.py     Step 1   continuum, flux, single/double Gaussians, moments
  dispersion.py  Step 1b  disc / outflow dispersion medians
  models.py      disc, bicone and combined velocity fields
  kinfit.py      Step 2   emcee / least-squares fits, trends with IP
  decompose.py   Step 3   constrained amplitude fits -> f_disc
  stacking.py    Step 4   weighted multi-line map, robustness checks
  extraction.py  Step 5   component spectra, stitching, continuum split
  external.py    optional imported velocity / dispersion maps (3DBarolo, MOKA3D, ...)
  plotting.py    one function per figure of the paper
  mock.py        synthetic ring + bicone cubes in the real MIRI/MRS bands, with known truth
```

## Notes on differences from the original analysis notebooks

* Gaussian fits are done in velocity space; the default is a bounded trust-region fit, while
  `fitter="capfit", weighting="sqrt", p0_center="systemic"` reproduces the original fitting.
* All model radii are in pc (not pixels), so lines from channels with different pixel scales share parameters.
* The rotation curve is always the tanh form of the paper.
* `percentile_over="all"` reproduces the original flux masking (percentile over the whole array, incl. empty
  padding); `"fov"` uses only spaxels inside the field of view.
* Fit weights: unweighted by default; `weighting="err"` (ERR extension) or `"sqrt"` (original) available.

## Citation

If you use this code, please cite the Breaking the Blend paper:

> Veenema O., et al., 2026, MNRAS, "Breaking the Blend: A multi-tracer kinematic decomposition method for IFS data
> applied to disentangling the AGN outflow and circumnuclear ring in NGC 5728 with JWST",
> doi:[10.1093/mnras/stag1917](https://doi.org/10.1093/mnras/stag1917), arXiv:[2609.30232](https://arxiv.org/abs/2609.30232)

## Disclaimer

This code is shared to support research, as is and without warranty of any kind. Its results depend on the data,
the settings and the modelling choices made by the user, and they are not guaranteed to be correct, complete or
appropriate for any particular purpose. Users are responsible for checking the output and for any conclusions drawn
from it, including in publications; the authors accept no liability for errors, misinterpretations or other
consequences arising from its use.

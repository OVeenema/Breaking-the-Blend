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
| `00_quickstart_mock.ipynb` | none (synthetic) | the whole method in a few minutes, checked against a known answer |
| `01_NGC5728.ipynb` | NGC 5728 cubes (below) | fully worked real-data example |
| `02_template_new_galaxy.ipynb` | your cubes | clean template for any galaxy (cells to edit marked **EDIT**) |
| `03_check_line_settings.ipynb` | your cubes | quick check of line windows and fits before a full run |
| `04_geometry_explorer.ipynb` | none | schematic of any disc + bicone geometry relative to the line of sight |

**4. Data for notebook 01.** Download the public JWST cubes of NGC 5728 (programmes 1670 and 5017) from MAST:

```bash
python scripts/download_mast_cubes.py --target "NGC 5728" --programs 1670 5017 --out data/NGC5728
```

or follow the portal instructions in [`data/NGC5728/README.md`](data/NGC5728/README.md).
Results of every notebook are written to `notebooks/results/<name>/`.

## The method in five steps, plus corrected spectra

| Step | What | Function(s) |
|---|---|---|
| 1 | Continuum-subtract each line, flux map, single- and double-Gaussian fits; median σ of each component | `fit_lines`, `dispersion_medians` |
| 2 | Fit a thin rotating disc, a hollow bicone, or both (α-weighted) to the velocity maps — or skip and give literature values; draw the geometry | `fit_kinematics`, `fit_many`, `trend_with_ip`, `plot_geometry` |
| 3 | In every spaxel fix both Gaussians' **velocities** (models) and **widths** (Step 1), fit only the amplitudes → disc fraction per line | `line_params`, `decompose_lines` |
| 4 | Weighted stack of all lines on a common pc grid → final AGN-outflow fraction map, and the share of it dominated by each component | `stack_fraction_maps`, `fraction_summary` |
| 5 | AGN- and star-formation-dominated spectra (with errors) across all bands, stitched and saved as observed (optionally: split a continuum map, `decompose_continuum`) | `extract_all`, `stitch_spectra` |
| + | **Corrected spectra**: τ<sub>9.8</sub> per spaxel from the H₂ S(1)-S(7) rotation diagram; every region spaxel corrected for extinction, then for the AGN light spread by the PSF (nuclear aperture, AGN point source, PSF model), per band; then stitched → AGN-psf-corrected, AGN-outflow-corrected and SF-corrected spectra, and a τ<sub>9.8</sub> map | `h2_line_maps`, `fit_tau`, `tau_on_cube`, `extract_apertures`, `psf_correct`, `corrected_region_spectra`, `stitch_bands` |

## Install

Use a dedicated environment so the analysis always runs with the same packages.

**With conda / mamba (recommended):**

```bash
git clone https://github.com/OVeenema/Breaking-the-Blend.git
cd Breaking-the-Blend
conda env create -f environment.yml          # creates "breaking-the-blend" and installs this package (editable)
conda activate breaking-the-blend
python -m ipykernel install --user --name breaking-the-blend --display-name "Breaking the Blend"
python tests/test_pipeline.py                # ~1-2 min, runs on synthetic data
```

**With venv + pip:**

```bash
cd Breaking-the-Blend
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
  answer. No data needed. Start here (it ends with some things to try).
* **`notebooks/01_NGC5728.ipynb`** — a fully worked, annotated example on NGC 5728. It adopts several settings of the
  original by-hand analysis (marked in the notebook) so the published maps are reproduced closely. Its last section
  (also in notebook 02) turns the decomposition into corrected spectra: the AGN point source (PSF-corrected), the AGN
  outflow and the star formation, corrected spaxel by spaxel first for extinction and then for the AGN's PSF light.
* **`notebooks/02_template_new_galaxy.ipynb`** — **start here for a new target**: the same workflow with no
  galaxy-specific settings, generic starting values, and the cells to edit clearly marked (**EDIT**).
* **`notebooks/03_check_line_settings.ipynb`** — a fast checker for a new dataset: shows each line's wavelength
  window (line region, continuum fit, neighbouring lines) and fits a handful of spaxels per line, flagging failed or
  degenerate fits. Tune `LINE_SETTINGS` here, then copy them into notebook 02.
* **`notebooks/04_geometry_explorer.ipynb`** — draw any disc + hollow-bicone geometry as seen by the observer and in
  3-D (in the style of Fischer et al. 2013), with worked examples for NGC 5728 and Circinus. No data needed.

Every notebook has a `SAVE_FIGURES` switch at the top: `"png"`, `"pdf"`, `"both"` or `"none"`
(figures go to `notebooks/results/<name>/figures/`).

## One analysis per target

Everything for a galaxy is handled by a `Project`: where the cubes are, the redshift, and a results folder.

```python
import breaking_the_blend as btb

proj  = btb.Project("NGC5728", data_dir="../data/NGC5728", z=0.00935)   # results -> results/NGC5728/
proj.find_cubes()                       # finds every *s3d.fits* (any sub-folder), names them ch3-short, g395h-f290lp, ...
cubes = proj.load_cubes(load_err=True)  # Jy/pixel, rest-frame wavelengths, AGN position per cube (+ ERR extensions)

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
the disc / redshifted cone). `inc_out` is the angle between the bicone axis and the **line of sight** (0 = pole-on);
`inc_disc` is the usual disc inclination (0 = face-on). Maps are stored as in the cube and drawn with
`origin="lower"`; for pipeline `skyalign` cubes (north up, east left) the conversion to the angle from +x is
`PA + 90` (`pa_offset`). Free parameters that end on the edge of their prior range are flagged in the fit summary.

## Options worth knowing

* **Double-Gaussian fitting** (`fit_lines`): `fitter="trf"` (bounded, default), `"lm"` or `"capfit"` (unbounded,
  as in the original analysis); `weighting="none" | "err" | "sqrt"`; `p0_center="moment" | "systemic"`.
* **Kinematic fits**: `method="emcee" | "least_squares"`; `sigma_v="residual"` (scatter about the best fit,
  realistic errors) or `"std"` (scatter of the map, as in the paper).
* **Imported models**: `btb.load_external_map(...)` brings in velocity/dispersion maps from 3DBarolo, MOKA3D, etc.
  (give the AGN pixel position and pixel size of that map); pass them to `decompose_lines(..., external={...})`.
* **Decomposition**: per-line or common model parameters, measured or fixed dispersions, optional re-extraction
  of the spectra with other windows (`recut`), fixed or free dispersions (`sigma_mode`).
* **Geometry schematic**: `btb.plot_geometry(params)` draws the disc and hollow bicone as seen by the observer (parts
  of the cones behind the disc as a wire-frame, parts in front shaded, cone/disc intersections marked) and in 3-D.
  Used in notebooks 01 and 02 for the best-fit geometry and the geometry adopted in the decomposition
  (`btb.mean_geometry(...)`). By default the outer opening angle is drawn as `theta_out / 2`, because the model
  applies its opening angles to sky-projected positions (`halve_theta_out=False` draws it unchanged, e.g. for
  3-D half-opening angles from the literature). Which side of the disc is nearer to us is set with
  `disc_near_side=+1/-1` (not constrained by the kinematics).
* **Dominated fractions**: `btb.fraction_summary(stack, sf_max=0.2, agn_min=0.8)` prints the share of the stacked map
  that is disc / star-formation dominated, mixed, and outflow / AGN dominated.
* **Plots**: `breaking_the_blend.plotting` sets a publication style on import (`bplt.set_style(...)` to change it).
  Fraction maps use cmasher *iceburn*, oriented so star formation is always orange and the AGN outflow
  always blue (disc-fraction maps: high = orange; outflow-fraction maps: high = blue).
* **Dispersions**: `dispersion_medians(..., min_ratio=1.3)` ignores spaxels where the double-Gaussian fit is
  degenerate (a single-component profile gives two Gaussians of nearly the same width).
* **Spectra**: detector gaps are left out, bands with too few selected spaxels are skipped (`min_spaxels`), and
  `stitch_spectra` scales all bands relative to a reference band (default: the one covering the largest sky area);
  `stitch_bands` (used for the corrected spectra) starts from the bluest band instead.
* **Spectrum errors**: `extract_all(..., err_method=...)` gives a 1σ error per channel - `"cube"` (the cubes' ERR
  extensions summed in quadrature; load the cubes with `load_err=True`; a lower limit, as neighbouring spaxels of a
  resampled cube share noise), `"empirical"` (the measured channel-to-channel scatter of each spectrum), or `"max"`
  (the larger of the two). `stitch_spectra(..., with_errors=True)` carries them through the stitching, and the
  exported `component_spectra_*.txt` files have three columns: rest wavelength [µm], flux [Jy], error [Jy].
* **Corrected spectra** (notebooks 01 and 02, Section 8). The Step 5 spectra are sums of observed spaxels; Section 8
  corrects every AGN-dominated (f_out > 0.8) and SF-dominated (f_out < 0.2) spaxel, band by band, first for extinction
  and then for the light of the unresolved AGN spread by the PSF, and only then stitches the bands (applying
  band-dependent corrections to an already stitched spectrum leaves jumps at the band edges). τ_9.8 comes from the H₂
  lines, which the AGN does not emit, so it is measured on the observed cubes with no PSF model involved. Each spaxel is
  corrected, D_i' = D_i exp(+τ_9.8,i τ(λ)), and its PSF contamination, C_i = P_i S exp(+τ_9.8,i τ(λ)) (P_i the fraction of
  the AGN point source the PSF model puts in spaxel i, S the point-source spectrum of that band; the AGN light in the
  spaxel was scaled by the same correction), is removed: F_R(λ) = Σ_i [D_i'(λ) − C_i(λ)]. Outputs in
  `results/<name>/`: `spectrum_AGN-raw.txt` (0.3" nuclear aperture), `spectrum_AGN-psf-corrected.txt` (aperture / PSF
  encircled energy), `spectrum_AGN-outflow-corrected.txt`, `spectrum_SF-corrected.txt` (columns: rest wavelength,
  flux, random error, PSF-model systematic) and the τ_9.8 map on the grid of the outflow fraction map
  (`tau98_map.npz/.fits`, `figures/tau98_map.pdf`). The steps as functions, in the order used:
  * `btb.h2_line_maps(cubes, target="S1")` - H₂ S(1)-S(7) flux maps, PSF-matched on the MRS channel-3 grid of S(1)
    (S(7) deblended from [Mg VII]); `btb.fit_tau(maps, method="powerlaw" | "smoothness")` - τ_9.8 in every spaxel from
    the dip of S(3) (in the 9.8 µm silicate feature) in the rotation diagram (Donnan et al. 2024, MNRAS 529, 1386,
    Fig. 7), with Monte-Carlo errors; `btb.tau_on_stack(tau_map, stack)` puts it on the grid of the outflow map and
    `btb.tau_on_cube(tau_map, cube)` on any cube;
  * extinction law: `btb.extinction_law(wave, law="donnan2023")` - Donnan et al. (2023a): 0.1 (9.8/λ)^1.7 + 0.9 τ_Sil
    with the empirical silicate profile of IRAS 08572+3915 (bundled table from Donnan's
    [SPIRIT](https://github.com/FergusDonnan/SPIRIT), BSD-3); `"kvt"` (PAHFIT) or any `(wave, tau)` table also work;
  * `btb.extract_apertures(cubes, {"nuclear": dict(radius_arcsec=0.3)})` - aperture spectra (exact pixel weights,
    centre given as an east/north offset from the AGN, so the same sky region in every band);
  * `btb.psf_correct(cubes, spectra, aperture_spectra["nuclear"], model="webbpsf", point_source="fit")` - the AGN
    point-source spectrum of every band (PSF + smooth extended emission fitted around the nucleus, or aperture / EE)
    and the PSF fractions; STPSF (WebbPSF) instrument PSFs, or a Gaussian. STPSF is installed with the environment;
    its data files (~90 MB) download to `~/data/stpsf-data` on first use. `btb.psf_contamination` and
    `btb.growth_curve` are the checks shown in the notebooks;
  * `btb.corrected_region_spectra(cubes, spectra, psf_res, tau_maps=...)` - the per-spaxel, per-band corrections:
    extinction first, then the PSF contamination of the corrected spaxels (also extinction-only and PSF-only versions;
    with `psf_res={}` extinction only); `btb.stitch_bands(bands, reference="blue", min_area_frac=0.3)` - stitches
    from the short-wavelength end with factors measured on the corrected spectra, dropping bands that cover only a
    small part of a region.

  Older building blocks are still available: `btb.subtract_aperture` (region minus aperture on the pixels) and
  `btb.extinction_corrected_spectra` (extinction-only extraction).
* **Exported map**: `results/<name>/stacked_fraction_map.fits` holds the AGN outflow fraction (primary), the disc
  fraction (`F_DISC`) and `N_LINES`, the number of lines contributing to each pixel.

## Package layout

```
breaking_the_blend/
  project.py     Project: data folder, results folder, checkpoints, config, exports
  lines.py       LineSpec + catalogue of NIR/MIR ionic lines
  cubes.py       load s3d cubes, pc scale, AGN position, line -> cube assignment
  linefit.py     Step 1   continuum, flux, single/double Gaussians, moments, quick line checks
  dispersion.py  Step 1b  disc / outflow dispersion medians
  models.py      disc, bicone and combined velocity fields
  kinfit.py      Step 2   emcee / least-squares fits, trends with IP
  geometry.py    schematic of the disc + bicone geometry relative to the line of sight
  decompose.py   Step 3   constrained amplitude fits -> f_disc
  stacking.py    Step 4   weighted multi-line map, robustness checks, dominated fractions
  extraction.py  Step 5   component and aperture spectra with errors, stitching (incl. per-band corrected), continuum split
  psf.py         AGN point source and PSF model (STPSF / Gaussian); per-spaxel extinction-then-PSF corrected spectra
  extinction.py  H2 line maps, tau_9.8 per spaxel from the H2 rotation diagram, extinction laws, regridding
  data/          D23ExtCurve.txt: Donnan et al. (2023a) extinction curve (from SPIRIT, BSD-3 licence)
  external.py    optional imported velocity / dispersion maps (3DBarolo, MOKA3D, ...)
  plotting.py    one function per figure of the paper
  mock.py        synthetic ring + bicone cubes in the real MIRI/MRS bands, with known truth
scripts/
  download_mast_cubes.py   download the s3d cubes of a target from MAST
```

## Notes on differences from the original analysis notebooks

* Gaussian fits are done in velocity space; the default is a bounded trust-region fit, while
  `fitter="capfit", weighting="sqrt", p0_center="systemic"` reproduces the original fitting.
* All model radii are in pc (not pixels), so lines from channels with different pixel scales share parameters.
* The rotation curve is always the tanh form of the paper.
* `percentile_over="all"` reproduces the original flux masking (percentile over the whole array, incl. empty
  padding); `"fov"` uses only spaxels inside the field of view.
* Fit weights: unweighted by default; `weighting="err"` (ERR extension) or `"sqrt"` (original) available.

## Contact

Questions, problems or ideas? I am happy to help with applying the code to your data, and always interested in working
with users on new applications of the method. Please get in touch by email at
[oscar.veenema@physics.ox.ac.uk](mailto:oscar.veenema@physics.ox.ac.uk), or open an issue on this repository.

## Getting in touch

Questions, problems, or ideas for applying Breaking the Blend to your own data are very welcome - I am happy to help
with setting it up for a new target and open to collaborating. Please get in touch by email:
**Oscar Veenema** - [oscar.veenema@physics.ox.ac.uk](mailto:oscar.veenema@physics.ox.ac.uk).

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

"""
breaking_the_blend - multi-tracer kinematic decomposition of IFS data
(Veenema et al. 2026, "Breaking the Blend").

Workflow (one module per step):

1. ``cubes`` + ``linefit``   load cubes, fit every line (flux, 1- and 2-Gaussian maps)
   ``dispersion``            median disc / outflow dispersions per line
2. ``models`` + ``kinfit``   disc / bicone / combined velocity-field fits
3. ``decompose``             fixed-velocity, fixed-dispersion amplitude fits -> f_disc per line
4. ``stacking``              weighted multi-line f_disc / f_out map
5. ``extraction``            AGN- and SF-dominated spectra, continuum decomposition

``project.Project`` keeps one results folder per galaxy (checkpoints, figures,
config.json) so the same notebook runs on many targets.

Typical use::

    import breaking_the_blend as btb
    proj = btb.Project("NGC5728", data_dir="/data/NGC5728", z=0.00935)
    cubes = proj.load_cubes()
"""
from .lines import LineSpec, LINE_CATALOGUE, get_lines, sort_by_ip, C_KMS
from .cubes import (Cube, load_cube, load_cubes, set_agn_positions, assign_cubes,
                    find_agn_position, pc_per_arcsec)
from .linefit import LineMaps, fit_line, fit_lines, check_line, check_summary
from .dispersion import dispersion_medians, dispersion_samples
from .models import (disc_velocity, bicone_velocity, model_velocity, component_velocities,
                     PA_OFFSET_DEG)
from .kinfit import (KinematicFit, fit_kinematics, fit_many, fits_table, trend_with_ip,
                     mean_params, DEFAULT_PARAMS)
from .decompose import Decomposition, decompose_line, decompose_lines, line_params
from .stacking import (StackedMap, stack_fraction_maps, line_weight, line_to_line_scatter,
                       compare_stacks, fraction_summary)
from .extraction import (ComponentSpectra, extract_spectra, extract_all, stitch_spectra,
                         decompose_continuum, fraction_on_cube, empirical_errors,
                         ApertureSpectrum, aperture_weights, extract_aperture, extract_apertures,
                         reference_band, subtract_spectra, subtract_aperture)
from .external import ExternalMap, load_external_map
from .project import Project, list_projects
from .utils import save, load
from .geometry import plot_geometry, plot_sky_view, plot_3d_view, mean_geometry, geometry_params
from .mock import MRS_BANDS, NIRSPEC_BANDS
from . import plotting, mock

__version__ = "0.1.0"

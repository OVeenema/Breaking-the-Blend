"""End-to-end test on mock cubes (runs without JWST data, astropy or emcee).

Run with ``pytest tests`` or ``python tests/test_pipeline.py``.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # works without install
import breaking_the_blend as btb

BANDS = {k: btb.MRS_BANDS[k] for k in ("ch1-short", "ch3-short", "ch3-medium", "ch3-long", "ch4-long")}
NAMES = ["FeII", "NeII", "NeIII", "OIV"]


def _setup():
    cubes, truth = btb.mock.make_mock_cubes(BANDS, fov_arcsec=5.5)
    btb.set_agn_positions(cubes, verbose=False)
    lines = {n: l.update(mask_percentile=40) for n, l in btb.get_lines(NAMES + ["NeV"]).items()}
    lines["NeV"] = lines["NeV"].update(line_region=(14.295, 14.345), fit_window=(14.265, 14.350))
    lines = btb.assign_cubes(lines, cubes, verbose=False)
    lm = btb.fit_lines(cubes, lines, show_progress=False)
    return cubes, truth, lm


def test_full_pipeline(tmp_path):
    cubes, truth, lm = _setup()

    # Step 1b: dispersions recovered
    sig = btb.dispersion_medians(lm, verbose=False)
    for n in ["NeII", "NeIII"]:
        assert abs(sig[n]["disc"] - truth["sigma_disc"]) < 15
        assert abs(sig[n]["outflow"] - truth["sigma_out"]) < 30

    # Step 2: combined fit recovers the outflow speed roughly
    geo = {k: truth["params"][k] for k in ("inc_disc", "pa_disc", "inc_out", "pa_out")}
    free = dict(v_disc=(50, 800), R_disc=(20, 2000), v_out=(100, 1500), R_out=(50, 3000),
                theta_in=(0, 50), theta_out=(30, 110))
    fit = btb.fit_kinematics(lm["NeIII"], "combined", {**geo, **free}, method="least_squares",
                             n_starts=5, seed=1, verbose=False)
    assert fit.mean_abs_residual < 60

    # Step 3: with the true model the disc fractions are recovered
    params = btb.line_params({n: lm[n] for n in NAMES}, truth["params"], verbose=False)
    decs = btb.decompose_lines(lm, params, sig, names=NAMES, show_progress=False)
    for n, d in decs.items():
        assert np.nanmedian(np.abs(d.f_disc - truth["f_disc"][n])) < 0.05

    # external map (truth disc velocity on a coarser grid) + recut + sigma_fraction
    from breaking_the_blend.models import disc_velocity
    yy, xx = np.indices((25, 25), dtype=float)
    s_ext = 60.0
    p = truth["params"]
    vmap = disc_velocity((xx - 12) * s_ext, (yy - 12) * s_ext, p["v_disc"], p["R_disc"], p["inc_disc"], p["pa_disc"])
    ext = btb.load_external_map(vmap, agn_xy=(12, 12), pixel_scale_pc=s_ext, name="truth disc")
    n = "NeIII"
    d_ext = btb.decompose_line(lm[n], params[n], sig[n]["disc"], sig[n]["outflow"], external={"v_disc": ext},
                               cube=cubes[lm[n].cube_name],
                               recut=dict(window=(15.45, 15.65), line_region=(15.52, 15.60), cont_order=1),
                               show_progress=False)
    assert d_ext.external == ("v_disc",)
    assert np.nanmedian(np.abs(d_ext.f_disc - truth["f_disc"][n])) < 0.08

    # Step 4
    stack = btb.stack_fraction_maps(decs, verbose=False)
    assert np.isfinite(stack.f_out).sum() > 100
    assert np.nanmin(stack.f_out) < 0.2 and np.nanmax(stack.f_out) > 0.8

    # Step 5
    spec = btb.extract_all(cubes, stack)
    assert all(s.agn_mask.sum() > 0 and s.sf_mask.sum() > 0 for s in spec.values())
    w, f, _ = btb.stitch_spectra(spec, "agn", verbose=False)
    assert np.all(np.diff(w) >= 0)

    # Project bookkeeping
    proj = btb.Project("TestGalaxy", z=0.0093, results_dir=str(tmp_path))
    out = proj.checkpoint("stack", lambda: stack, settings={"a": 1})
    again = proj.checkpoint("stack", lambda: None, settings={"a": 1})
    assert again is not None
    rerun = proj.checkpoint("stack", lambda: "new", settings={"a": 2})
    assert rerun == "new"
    proj.export_stacked_map(stack)
    assert btb.Project.open("TestGalaxy", results_dir=str(tmp_path)).z == 0.0093

    # plots run
    bplt = btb.plotting
    for fig in (bplt.plot_line_maps(lm, "flux"), bplt.plot_dispersion_histograms(lm, sig),
                bplt.plot_kinematic_fit(fit, lm["NeIII"]), bplt.plot_fraction_maps(decs),
                bplt.plot_stacked_map(stack), bplt.plot_cubes_overview(cubes)):
        assert fig is not None


def test_unbounded_fitters():
    cubes, truth = btb.mock.make_mock_cubes({"ch3-short": btb.MRS_BANDS["ch3-short"]}, fov_arcsec=4)
    btb.set_agn_positions(cubes, verbose=False)
    line = btb.get_lines(["NeII"])["NeII"].update(cube="ch3-short", mask_percentile=50)
    lm = btb.fit_line(cubes["ch3-short"], line, fitter="lm", weighting="sqrt", p0_center="systemic",
                      show_progress=False)
    sig = btb.dispersion_medians({"NeII": lm}, verbose=False)["NeII"]
    assert abs(sig["disc"] - 70) < 20 and abs(sig["outflow"] - 200) < 60


def test_check_and_stitch():
    cubes, truth = btb.mock.make_mock_cubes({k: btb.MRS_BANDS[k] for k in ("ch3-short", "ch3-medium")}, fov_arcsec=7)
    btb.set_agn_positions(cubes, verbose=False)
    line = btb.get_lines(["NeII"])["NeII"].update(cube="ch3-short", mask_percentile=40)
    chk = btb.check_line(cubes["ch3-short"], line, n_spaxels=4)
    assert len(chk.fitted_spaxels) == 4
    assert np.isfinite(chk.double["sigma_n"]).sum() == 4
    s = btb.check_summary({"NeII": chk}, verbose=False)["NeII"]
    assert s["failed"] == 0
    # a detector gap (zeros) becomes NaN in the extracted spectrum
    cubes["ch3-short"].data[100:110] = 0.0
    lm = btb.fit_lines(cubes, {"NeII": line}, show_progress=False)
    d = btb.decompose_lines(lm, btb.line_params(lm, truth["params"], verbose=False),
                            {"NeII": {"disc": 70, "outflow": 200}}, show_progress=False)
    stack = btb.stack_fraction_maps(d, min_lines=1, verbose=False)
    spec = btb.extract_all(cubes, stack, method="weighted")
    assert np.all(np.isnan(spec["ch3-short"].agn[100:110]))
    w, f, sc = btb.stitch_spectra(spec, "agn", verbose=False)
    assert np.all(np.isfinite(f)) and np.all(np.diff(w) >= 0)


def test_models_pa_convention():
    # disc receding side lies along pa_disc (east of north; +x = west, +y = north)
    p = dict(v_disc=200, R_disc=100, inc_disc=60, pa_disc=0)  # receding to the north
    assert btb.disc_velocity(0.0, 500.0, **p) > 0 and btb.disc_velocity(0.0, -500.0, **p) < 0
    p["pa_disc"] = 90                                          # receding to the east (-x)
    assert btb.disc_velocity(-500.0, 0.0, **p) > 0


if __name__ == "__main__":
    import tempfile, pathlib
    test_models_pa_convention()
    test_unbounded_fitters()
    test_check_and_stitch()
    test_full_pipeline(pathlib.Path(tempfile.mkdtemp()))
    print("all tests passed")

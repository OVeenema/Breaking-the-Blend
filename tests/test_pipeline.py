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


def test_geometry_and_fractions():
    import matplotlib.pyplot as plt
    from breaking_the_blend.geometry import outflow_axis, disc_frame
    # redshifted cone points away from the observer, along its PA on the sky
    a = outflow_axis(inc_out=60, pa_out=90)
    assert a[2] < 0 and a[0] > 0 and abs(a[1]) < 1e-9
    m, u, n = disc_frame(inc_disc=30, pa_disc=0)
    assert np.isclose(n[2], np.cos(np.deg2rad(30))) and np.isclose(abs(m @ n), 0)
    p = dict(inc_disc=43, pa_disc=194, inc_out=48, pa_out=110, theta_in=21, theta_out=80, R_out=600)
    fig = btb.plot_geometry(p, print_params=False)
    assert len(fig.axes) == 2
    plt.close(fig)
    g = btb.mean_geometry({"a": {**p, "pa_out": 350}, "b": {**p, "pa_out": 10}})
    assert min(abs(g["pa_out"]), abs(g["pa_out"] - 360)) < 1e-6        # circular mean of 350 and 10 is 0
    # fraction summary on a toy stacked map
    from breaking_the_blend.stacking import StackedMap
    f = np.array([[0.1, 0.5], [0.9, np.nan]])
    st = StackedMap(x_pc=np.array([0.0, 10.0]), y_pc=np.array([0.0, 10.0]), f_disc=1 - f, n_lines=np.ones((2, 2)),
                    weights={}, mean_f_disc={}, scheme="mean/linear")
    r = btb.fraction_summary(st, verbose=False)
    assert np.isclose(r["disc_dominated"], 1 / 3) and np.isclose(r["outflow_dominated"], 1 / 3)


def test_apertures():
    # exact weights reproduce the circle area; centre-method apertures equal a direct pixel sum
    w = btb.aperture_weights((40, 40), 19.3, 20.7, 2.3)
    assert abs(w.sum() - np.pi * 2.3 ** 2) < 0.1
    cubes, truth = btb.mock.make_mock_cubes({k: btb.MRS_BANDS[k] for k in ("ch3-short", "ch3-medium")}, fov_arcsec=5)
    btb.set_agn_positions(cubes, verbose=False)
    c = cubes["ch3-short"]
    s = btb.extract_aperture(c, 0.3, method="center", err_method=None)
    assert np.allclose(s.flux, np.asarray(c.data, float)[:, s.weights > 0].sum(axis=1))
    ap = btb.extract_apertures(cubes, {"nuc": dict(radius_arcsec=0.3), "off": dict(radius_arcsec=0.4, offset_arcsec=(0.6, 0.2))},
                               verbose=False)
    assert ap["off"]["ch3-short"].center_xy[0] < c.agn_xy[0]          # east = -x
    wn, fn, en, sc = btb.stitch_spectra(ap["nuc"], "flux", with_errors=True, verbose=False)
    assert np.all(np.isfinite(fn)) and np.all(en > 0)
    wr, dr, er, s_ = btb.subtract_spectra(wn, fn, en, wn, 0.5 * fn, 0.5 * en, scale="window", window=(12.0, 12.5),
                                          verbose=False)
    assert abs(s_ - 2.0) < 1e-6 and np.nanmax(np.abs(dr)) < 1e-9
    # region minus aperture, per band: equals the plain difference when the aperture lies inside the region
    from breaking_the_blend.extraction import ComponentSpectra
    reg = np.zeros(c.shape2d, bool)
    reg[max(int(c.agn_xy[1]) - 6, 0):int(c.agn_xy[1]) + 7, max(int(c.agn_xy[0]) - 6, 0):int(c.agn_xy[0]) + 7] = True
    data = np.asarray(c.data, float)
    cs = ComponentSpectra("ch3-short", c.wave, data[:, reg].sum(axis=1), np.full(c.wave.size, np.nan), reg, ~reg,
                          np.zeros(c.shape2d), "threshold", c.pixel_scale, None, err_method=None)
    diff = btb.subtract_aperture(cubes, {"ch3-short": cs}, ap["nuc"], "agn", err_method=None, verbose=False)
    assert np.allclose(diff["ch3-short"].flux, cs.agn - ap["nuc"]["ch3-short"].flux)


def test_empirical_errors():
    # noise is recovered on a steep continuum with resolved lines, and never returns zeros on slopes
    from breaking_the_blend.extraction import empirical_errors
    rng = np.random.default_rng(0)
    x = np.linspace(5, 28, 8000)
    f = 0.5 * (x / 10) ** 2 + 3 * np.exp(-0.5 * ((x - 12.8) / 0.02) ** 2) + rng.normal(0, 0.01, x.size)
    e = empirical_errors(f)
    assert np.all(e > 0) and abs(np.median(e) / 0.01 - 1) < 0.1


def test_psf_correction():
    # inject a point source with a known spectrum (Gaussian PSF, same model as the correction) into mock cubes and
    # compare with the same correction applied without it: the AGN spectrum must gain exactly the point source,
    # and the AGN-outflow / SF spectra must not change (the point source and its PSF leakage are removed)
    from breaking_the_blend.psf import psf_on_cube
    cubes, truth = btb.mock.make_mock_cubes({k: btb.MRS_BANDS[k] for k in ("ch3-short", "ch3-medium")}, fov_arcsec=10)
    btb.set_agn_positions(cubes, verbose=False)
    lines = btb.assign_cubes({n: l.update(mask_percentile=40) for n, l in btb.get_lines(["NeII", "NeV"]).items()},
                             cubes, verbose=False, drop_missing=True)
    lm = btb.fit_lines(cubes, lines, show_progress=False, fit_double=False)
    d = btb.decompose_lines(lm, btb.line_params(lm, truth["params"], verbose=False),
                            {n: {"disc": 70, "outflow": 200} for n in lm}, show_progress=False)
    stack = btb.stack_fraction_maps(d, verbose=False)

    def run(annulus):
        sp = btb.extract_all(cubes, stack, err_method="cube")
        ap = btb.extract_apertures(cubes, {"nuclear": dict(radius_arcsec=0.3)}, err_method="cube", verbose=False)
        return btb.psf_correct(cubes, sp, ap["nuclear"], model="gaussian", n_wave=8, annulus_arcsec=annulus,
                               verbose=False)[0]
    r0 = {a: run(a) for a in (None, (0.6, 0.9))}
    for c in cubes.values():                                                  # add 0.05 Jy x (lam/12)^2
        S = 0.05 * (c.wave / 12.0) ** 2
        for i in range(0, len(c.wave), 25):
            P = psf_on_cube(c, c.wave[i] * (1 + c.z), "gaussian")
            c.data[i:i + 25] += (S[i:i + 25, None, None] * P[None]).astype(c.data.dtype)
    b = "ch3-short"
    S = 0.05 * (cubes[b].wave / 12.0) ** 2
    for a in (None, (0.6, 0.9)):
        r1 = run(a)
        assert np.nanmedian(np.abs((r1["AGN"][b].flux - r0[a]["AGN"][b].flux) / S - 1)) < 0.02
        for name in ("AGN-outflow", "SF"):
            if b in r1[name]:
                ref = np.nanmedian(np.abs(r0[a][name][b].flux)) + 1e-12
                assert np.nanmedian(np.abs(r1[name][b].flux - r0[a][name][b].flux)) < 0.02 * max(ref, np.nanmedian(S))
        assert np.all(np.isfinite(r1["AGN-outflow"][b].flux_err))


def test_models_pa_convention():
    # disc receding side lies along pa_disc (east of north; +x = west, +y = north)
    p = dict(v_disc=200, R_disc=100, inc_disc=60, pa_disc=0)  # receding to the north
    assert btb.disc_velocity(0.0, 500.0, **p) > 0 and btb.disc_velocity(0.0, -500.0, **p) < 0
    p["pa_disc"] = 90                                          # receding to the east (-x)
    assert btb.disc_velocity(-500.0, 0.0, **p) > 0


def _mock_h2_cubes(tau_fn, bands=("ch2-short", "ch2-medium", "ch3-short", "ch3-long")):
    """Mock MRS cubes with H2 S(1)-S(4) on a power-law excitation diagram, behind a screen tau_fn(x_pc, y_pc)."""
    from breaking_the_blend import extinction as ex
    cubes, truth = btb.mock.make_mock_cubes(bands={b: btb.MRS_BANDS[b] for b in bands}, noise=0.01, fov_arcsec=5.0)
    for name, c in cubes.items():
        c.agn_xy = truth["agn_xy"][name]
        x, y = c.offsets_pc()
        omega = (c.pixel_scale / ex.ARCSEC_PER_RAD) ** 2
        data = np.array(c.data, float)
        for ln in ex.get_h2_lines(["S1", "S2", "S3", "S4"]).values():
            if not (c.wave[0] + 0.03 < ln.wave < c.wave[-1] - 0.03):
                continue
            N = 1e18 * (ln.Eu / 1000.0) ** -3.0 * ln.g * 1e4
            F = N * ln.A * ex.H_PLANCK * ex.C_MS * omega / (4 * np.pi * ln.wave * 1e-6)
            amp = F / (np.sqrt(2 * np.pi) * 90.0 / btb.C_KMS * ex.C_MS / (ln.wave * 1e-6) * 1e-26)
            v = btb.C_KMS * (c.wave - ln.wave) / ln.wave
            data += amp * np.exp(-0.5 * (v[:, None, None] / 90.0) ** 2)
        c.data = (data * np.exp(-tau_fn(x, y)[None] * ex.extinction_law(c.wave)[:, None, None])).astype(np.float32)
    return cubes


def test_extinction_from_h2():
    law = btb.extinction_law(np.array([9.8, 9.665, 17.03, 5.5]))
    assert abs(law[0] - 1) < 1e-9 and 0.95 < law[1] <= 1.0 and law[2] < 0.5 and law[3] < 0.5
    tau_fn = lambda x, y: 0.5 + 1.0 * np.exp(-0.5 * (x ** 2 + y ** 2) / 400.0 ** 2)
    cubes = _mock_h2_cubes(tau_fn)
    maps = btb.h2_line_maps(cubes, ["S1", "S2", "S3", "S4"], target="S1", verbose=False)
    assert all(m.grid.name == "ch3-long" for m in maps.values())
    for method in ("powerlaw", "smoothness"):
        tm = btb.fit_tau(maps, method=method, n_mc=20, verbose=False)
        x, y = tm.grid.offsets_pc()
        d = (tm.tau - tau_fn(x, y))[tm.ok]
        assert tm.ok.sum() > 100
        assert abs(np.median(d)) < 0.1 and np.std(d) < 0.15, (method, np.median(d), np.std(d))
    # the tau map on another grid, and the corrected cube recovers the unextincted continuum shape
    c = cubes["ch2-medium"]
    t = btb.tau_on_cube(tm, c)
    assert t.shape == c.shape2d and np.isfinite(t).all()
    xc, yc = c.offsets_pc()
    assert np.median(np.abs(t - tau_fn(xc, yc))) < 0.1
    cc = btb.correct_cube(c, t)
    assert cc.data.shape == c.data.shape
    assert np.nanmedian(np.asarray(cc.data) / np.where(c.data != 0, c.data, np.nan)) > 1.5


def test_per_spaxel_psf_and_extinction():
    """F_R = sum_i w_i [D_i - P_i S] exp(+tau_i k) recovers the intrinsic extended emission of a region exactly
    when the point source and the screen are known; per-band stitching from the blue end joins offset bands."""
    from breaking_the_blend.extraction import ComponentSpectra
    from breaking_the_blend.psf import psf_on_cube
    ny = nx = 31
    wave = np.linspace(8.0, 12.0, 120)
    cube = btb.Cube("ch2-medium", np.zeros((wave.size, ny, nx)), wave, 0.2, 100.0, err=np.full((wave.size, ny, nx), 1e-7),
                    agn_xy=(15.3, 14.8))
    yy, xx = np.indices((ny, nx))
    r = np.hypot(xx - 15.3, yy - 14.8)
    tau = 0.3 + 0.05 * (xx - 15)
    tau = np.clip(tau, 0, None)
    intr = (np.exp(-r / 8.0) * 1e-3)[None] * (wave / 10.0)[:, None, None]
    S = 2e-2 * (wave / 10.0) ** 2
    lam_s = np.linspace(wave.min(), wave.max(), 6)
    Ps = np.array([psf_on_cube(cube, l, "gaussian") for l in lam_s])
    P = np.array([np.interp(l, lam_s, np.arange(6)) for l in wave])
    i0 = np.minimum(np.floor(P).astype(int), 5); t = P - i0; i1 = np.minimum(i0 + 1, 5)
    Pl = (1 - t)[:, None, None] * Ps[i0] + t[:, None, None] * Ps[i1]
    k = btb.extinction_law(wave)
    cube.data = intr * np.exp(-tau[None] * k[:, None, None]) + S[:, None, None] * Pl
    agn_mask, sf_mask = r < 4, r > 9
    sp = ComponentSpectra(cube.name, wave, np.ones(wave.size), np.ones(wave.size), agn_mask, sf_mask,
                          np.zeros((ny, nx)), pixel_scale=0.2, footprint=np.ones((ny, nx), bool))
    src = btb.ApertureSpectrum("AGN-continuum", cube.name, wave, S, np.zeros_like(S), agn_mask.astype(float))
    out = btb.corrected_region_spectra({cube.name: cube}, {cube.name: sp}, {"AGN-continuum": {cube.name: src}},
                                       tau_maps={cube.name: tau}, model="gaussian", verbose=False)
    for R, m in (("AGN", agn_mask), ("SF", sf_mask)):
        truth = intr[:, m].sum(axis=1)
        got = out[f"{R}-corr"][cube.name].flux
        assert np.allclose(got, truth, rtol=1e-6, atol=1e-12), (R, np.max(np.abs(got / truth - 1)))
        assert np.all(out[f"{R}-raw"][cube.name].flux > 0)
    # blue-anchored stitching of two offset bands
    w1, w2 = np.linspace(5, 7, 100), np.linspace(6.5, 9, 120)
    a = btb.ApertureSpectrum("x", "b1", w1, w1 ** 2, 0.01 * w1, np.ones((2, 2)))
    b = btb.ApertureSpectrum("x", "b2", w2, 1.5 * w2 ** 2, 0.01 * w2, np.ones((2, 2)))
    w, f, e, s_, sc, ref = btb.stitch_bands({"b1": a, "b2": b}, verbose=False)
    assert ref == "b1" and abs(sc["b2"] - 1 / 1.5) < 1e-3 and np.allclose(f, w ** 2, rtol=1e-3)


if __name__ == "__main__":
    import tempfile, pathlib
    test_models_pa_convention()
    test_unbounded_fitters()
    test_check_and_stitch()
    test_geometry_and_fractions()
    test_apertures()
    test_empirical_errors()
    test_psf_correction()
    test_extinction_from_h2()
    test_per_spaxel_psf_and_extinction()
    test_full_pipeline(pathlib.Path(tempfile.mkdtemp()))
    print("all tests passed")

"""
STEP 2 - Fit the disc / outflow / combined velocity models to observed velocity maps
(paper Sect. 3.2.3).

Specifying parameters
---------------------
Every model parameter (see :mod:`breaking_the_blend.models`) is given in one dict:

* a number  -> the parameter is **fixed** to that value;
* a tuple (lo, hi) -> the parameter is **free** with a uniform prior lo..hi.

Anything you do not give falls back to :data:`DEFAULT_PARAMS`. Example::

    params = {"inc_disc": 43, "pa_disc": 194,          # fixed (literature)
              "v_disc": (100, 800), "R_disc": (50, 1500)}  # free
    fit = fit_kinematics(line_maps["NeII"], "disc", params)

``init`` (same keys, (lo, hi)) optionally narrows where the MCMC walkers /
least-squares starts are drawn from.

Methods
-------
``"least_squares"`` : fast multi-start optimisation, good for exploring.
``"emcee"`` : affine-invariant MCMC as in the paper (needs ``pip install emcee``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .linefit import LineMaps
from .models import COMPONENT_PARAMS, PA_OFFSET_DEG, model_velocity

DEFAULT_PARAMS: Dict[str, object] = {
    "v_sys": 0.0,
    # disc
    "v_disc": (20.0, 800.0), "R_disc": (10.0, 3000.0), "inc_disc": (10.0, 80.0), "pa_disc": (0.0, 360.0),
    # outflow
    "v_out": (50.0, 2000.0), "R_out": (10.0, 5000.0), "inc_out": (10.0, 80.0), "pa_out": (0.0, 360.0),
    "theta_in": (0.0, 60.0), "theta_out": (20.0, 120.0),
    # combined
    "alpha": (0.0, 1.0),
}


def _is_free(v) -> bool:
    return isinstance(v, (tuple, list)) and len(v) == 2


@dataclass
class KinematicFit:
    """Result of :func:`fit_kinematics`.

    params : best-fit values (median of the posterior for emcee) of *all* model
        parameters (fixed ones included) - pass this straight to later steps.
    errors : ``{name: (minus, plus)}`` for the free parameters.
    """

    line_name: str
    label: str
    ip: float
    component: str
    params: Dict[str, float]
    free: List[str]
    errors: Dict[str, Tuple[float, float]]
    method: str
    velocity_source: str
    pa_offset: float
    mean_abs_residual: float
    samples: Optional[np.ndarray] = field(default=None, repr=False)
    at_bound: List[str] = field(default_factory=list)   # free parameters ending on a prior limit

    def model_maps(self, lm: LineMaps) -> Dict[str, np.ndarray]:
        """Observed, disc, outflow, model and residual velocity maps on ``lm``'s grid."""
        x, y = lm.offsets_pc()
        obs = lm.velocity_map(self.velocity_source)
        out = {"observed": obs}
        p = self.params
        if self.component in ("disc", "combined"):
            out["disc"] = model_velocity(x, y, p, "disc", self.pa_offset)
        if self.component in ("outflow", "combined"):
            out["outflow"] = model_velocity(x, y, p, "outflow", self.pa_offset)
        out["model"] = model_velocity(x, y, p, self.component, self.pa_offset)
        out["residual"] = obs - out["model"]
        for k in out:
            out[k] = np.where(lm.good, out[k], np.nan)
        return out

    def summary(self) -> str:
        lines = [f"{self.label} (IP={self.ip} eV) - {self.component} model, {self.method}"]
        for k in COMPONENT_PARAMS[self.component]:
            v = self.params[k]
            if k in self.free:
                lo, hi = self.errors[k]
                flag = "   <-- at prior limit" if k in getattr(self, "at_bound", []) else ""
                lines.append(f"  {k:>9s} = {v:9.2f}  +{hi:.2f} / -{lo:.2f}{flag}")
            else:
                lines.append(f"  {k:>9s} = {v:9.2f}  (fixed)")
        lines.append(f"  mean |residual| = {self.mean_abs_residual:.1f} km/s")
        if getattr(self, "at_bound", []):
            lines.append(f"  WARNING: {', '.join(self.at_bound)} at the edge of the prior range - "
                         "the prior, not the data, sets these values; widen the range(s)")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


def _setup(lm, component, params, velocity_source):
    if component not in COMPONENT_PARAMS:
        raise ValueError("component must be 'disc', 'outflow' or 'combined'")
    merged = {**DEFAULT_PARAMS, **(params or {})}
    names = COMPONENT_PARAMS[component]
    free = [n for n in names if _is_free(merged[n])]
    fixed = {n: float(merged[n]) for n in names if not _is_free(merged[n])}
    bounds = np.array([merged[n] for n in free], dtype=float)
    v = lm.velocity_map(velocity_source)
    x, y = lm.offsets_pc()
    ok = np.isfinite(v)
    return free, fixed, bounds, x[ok], y[ok], v[ok]


def fit_kinematics(lm: LineMaps, component: str = "combined", params: Optional[dict] = None,
                   init: Optional[dict] = None, method: str = "emcee",
                   nwalkers: int = 32, nsteps: int = 5000, burnin: int = 1000,
                   n_starts: int = 20, velocity_source: str = "single", sigma_v="std",
                   pa_offset: float = PA_OFFSET_DEG, seed: Optional[int] = None,
                   show_progress: bool = True, verbose: bool = True,
                   max_samples: int = 20000) -> KinematicFit:
    """Fit a velocity model to the observed velocity map of one line.

    Parameters
    ----------
    lm : :class:`LineMaps` of the line.
    component : ``"disc"``, ``"outflow"`` or ``"combined"``.
    params : fixed values / (lo, hi) priors (see module docstring).
    init : optional (lo, hi) ranges to draw starting points from.
    method : ``"emcee"`` or ``"least_squares"``.
    nwalkers, nsteps, burnin : MCMC settings (paper: 32, 150000, 50000).
    n_starts : number of random starts for least squares.
    velocity_source : ``"single"`` (single-Gaussian velocities) or ``"moment"``.
    sigma_v : velocity uncertainty per spaxel in the likelihood (sets the size of the errors):
        ``"std"`` = std of the observed velocity map (as in the paper; conservative, large errors),
        ``"residual"`` = rms scatter of the data about a quick best fit (smaller, more realistic
        errors), or a number in km/s.
    max_samples : posterior samples kept in the result (random subset, keeps files small).
    """
    rng = np.random.default_rng(seed)
    free, fixed, bounds, x, y, v_obs = _setup(lm, component, params, velocity_source)
    if len(free) == 0:
        raise ValueError("No free parameters - give at least one (lo, hi) prior")
    sig = np.nanstd(v_obs) if sigma_v in ("std", "residual") else float(sigma_v)

    init = init or {}
    init_b = np.array([init.get(n, bounds[k]) for k, n in enumerate(free)], dtype=float)
    init_b[:, 0] = np.maximum(init_b[:, 0], bounds[:, 0])
    init_b[:, 1] = np.minimum(init_b[:, 1], bounds[:, 1])

    def full(theta):
        p = dict(fixed)
        p.update(zip(free, theta))
        return p

    def valid(p):
        return not ("theta_in" in p and "theta_out" in p and p["theta_out"] <= p["theta_in"])

    def model(theta):
        return model_velocity(x, y, full(theta), component, pa_offset)

    def log_prob(theta):
        if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]) or not valid(full(theta)):
            return -np.inf
        r = (v_obs - model(theta)) / sig
        return -0.5 * np.sum(r * r)

    def draw(n):
        pts, tries = [], 0
        while len(pts) < n:
            t = rng.uniform(init_b[:, 0], init_b[:, 1])
            tries += 1
            if np.isfinite(log_prob(t)):
                pts.append(t)
            elif tries > 1000 * n:
                raise RuntimeError("Could not draw valid starting points - check that the priors / "
                                   "init ranges allow theta_out > theta_in")
        return np.array(pts)

    def run_lsq(n):
        best, best_cost, best_res = None, np.inf, None
        for t0 in draw(n):
            res = least_squares(lambda t: (v_obs - model(t)) / sig, t0,
                                bounds=(bounds[:, 0], bounds[:, 1]), method="trf")
            if res.cost < best_cost and valid(full(res.x)):
                best, best_cost, best_res = res.x, res.cost, res
        return best, best_res

    if sigma_v == "residual":
        # likelihood width = scatter of the data about a quick best fit (not of the raw map)
        b0, _ = run_lsq(max(5, n_starts // 2))
        sig = float(np.std(v_obs - model(b0)))
        if verbose:
            print(f"  sigma_v (residual scatter) = {sig:.1f} km/s")

    samples = None
    if method == "emcee":
        try:
            import emcee
        except ImportError as e:
            raise ImportError("method='emcee' needs emcee (pip install emcee); "
                              "or use method='least_squares'") from e
        ndim = len(free)
        nw = max(nwalkers, 2 * ndim + 2)
        sampler = emcee.EnsembleSampler(nw, ndim, log_prob)
        sampler.run_mcmc(draw(nw), nsteps, progress=show_progress)
        samples = sampler.get_chain(discard=burnin, flat=True)
        if len(samples) > max_samples:   # keep saved results small
            samples = samples[rng.choice(len(samples), max_samples, replace=False)]
        p16, p50, p84 = np.percentile(samples, [16, 50, 84], axis=0)
        best = p50
        errors = {n: (p50[k] - p16[k], p84[k] - p50[k]) for k, n in enumerate(free)}
    elif method == "least_squares":
        best, best_res = run_lsq(n_starts)
        try:
            cov = np.linalg.pinv(best_res.jac.T @ best_res.jac)
            err = np.sqrt(np.clip(np.diag(cov), 0, None))
        except Exception:  # pragma: no cover
            err = np.full(len(free), np.nan)
        errors = {n: (err[k], err[k]) for k, n in enumerate(free)}
    else:
        raise ValueError("method must be 'emcee' or 'least_squares'")

    p = full(best)
    span = bounds[:, 1] - bounds[:, 0]
    at_bound = [n for k, n in enumerate(free)
                if min(best[k] - bounds[k, 0], bounds[k, 1] - best[k]) < 0.01 * span[k]]
    resid = np.nanmean(np.abs(v_obs - model_velocity(x, y, p, component, pa_offset)))
    fit = KinematicFit(line_name=lm.name, label=lm.label, ip=lm.ip, component=component,
                       params=p, free=free, errors=errors, method=method,
                       velocity_source=velocity_source, pa_offset=pa_offset,
                       mean_abs_residual=float(resid), samples=samples, at_bound=at_bound)
    if verbose:
        print(fit.summary())
    return fit


def fit_many(line_maps: Dict[str, LineMaps], names: List[str], component: str, **kwargs) -> Dict[str, KinematicFit]:
    """Run :func:`fit_kinematics` on several lines with the same settings."""
    return {n: fit_kinematics(line_maps[n], component, **kwargs) for n in names}


def fits_table(fits: Dict[str, KinematicFit], keys: Optional[List[str]] = None) -> None:
    """Print best-fit values of several fits side by side (sorted by IP)."""
    fits = dict(sorted(fits.items(), key=lambda kv: kv[1].ip))
    if keys is None:
        keys = sorted({k for f in fits.values() for k in f.free}, key=lambda k: COMPONENT_PARAMS["combined"].index(k))
    print(f"{'line':>9s} {'IP':>6s} " + " ".join(f"{k:>16s}" for k in keys))
    for f in fits.values():
        row = []
        for k in keys:
            if k in f.free:
                row.append(f"{f.params[k]:8.1f} +/-{np.mean(f.errors[k]):5.1f}")
            elif k in f.params:
                row.append(f"{f.params[k]:8.1f} (fix)   ")
            else:
                row.append(" " * 16)
        print(f"{f.label:>9s} {f.ip:6.1f} " + " ".join(f"{r:>16s}" for r in row))


def trend_with_ip(fits: Dict[str, KinematicFit], key: str, max_ip: Optional[float] = None) -> Tuple[float, float]:
    """Linear fit ``key = slope * IP + intercept`` across lines (paper: R_out and alpha vs IP).

    Returns (slope, intercept). Uses the mean of the +/- errors as weights.
    """
    ip, val, err = [], [], []
    for f in fits.values():
        if key in f.params and (max_ip is None or f.ip <= max_ip):
            ip.append(f.ip)
            val.append(f.params[key])
            err.append(np.mean(f.errors.get(key, (1.0, 1.0))) or 1.0)
    w = 1.0 / np.maximum(np.array(err), 1e-6)
    slope, intercept = np.polyfit(ip, val, 1, w=w)
    print(f"{key} = {slope:.3f} * IP + {intercept:.2f}")
    return slope, intercept


def mean_params(fits: Dict[str, KinematicFit], keys: List[str]) -> Dict[str, float]:
    """Average best-fit values of the free parameters ``keys`` over several fits (circular mean
    for position angles). Parameters that were fixed in all fits are skipped."""
    out = {}
    for k in keys:
        vals = np.array([f.params[k] for f in fits.values() if k in f.free])
        if vals.size == 0:
            continue
        if k.startswith("pa_"):            # angles: circular mean / scatter
            a = np.deg2rad(vals)
            m = np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))
            out[k] = float(np.rad2deg(m) % 360)
            sd = float(np.rad2deg(np.sqrt(-2 * np.log(np.hypot(np.mean(np.sin(a)), np.mean(np.cos(a)))))))
        else:
            out[k] = float(np.mean(vals))
            sd = float(np.std(vals))
        print(f"{k:>9s} = {out[k]:8.1f} +/- {sd:.1f}  (N={len(vals)})")
    return out

#!/usr/bin/env python3
"""
Robust numerical sweep for a discrete quaternionic orientational lattice model.

Purpose
-------
This script is intended to replace the earlier single-size/single-protocol
spectrum script when testing whether the effective exponent alpha is robust.
It performs a finite-size and multi-seed sweep over boundary degree Q.

It saves all runs, summaries by (N,Q), best minima by (N,Q), fits alpha(N),
and morphology fields for representative best configurations.

Example
-------
Quick test:
    python quaternionic_lattice_simulation.py --N-values 16 24 --Q-values 1 2 3 4 --seeds 4 --steps 3000

Main run, more serious:
    python quaternionic_lattice_simulation.py --N-values 16 24 32 --Q-values 1 2 3 4 5 6 7 8 9 10 --seeds 8 --steps 9000 --adaptive --min-steps 3000 --delta-E-tol 1e-5 --patience 5

Optional parameter test at fixed N:
    python quaternionic_lattice_simulation.py --N-values 32 --Q-values 1 2 3 4 5 6 7 8 --seeds 8 --kappa 0.1 --lambda-pot 5.0
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad


# ============================================================
# OUTPUT PATHS
# ============================================================

OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
CSV_DIR = OUT_DIR / "csv"
DATA_DIR = OUT_DIR / "data"
for _d in (FIG_DIR, CSV_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ALL_RUNS_CSV = CSV_DIR / "robust_all_runs.csv"
SUMMARY_CSV = CSV_DIR / "robust_summary_by_N_Q.csv"
BEST_CSV = CSV_DIR / "robust_best_by_N_Q.csv"
FITS_CSV = CSV_DIR / "robust_fits_by_N.csv"
ALPHA_EXTRAP_CSV = CSV_DIR / "robust_alpha_extrapolation.csv"
PARAMS_CSV = CSV_DIR / "robust_run_parameters.csv"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 400,
})


# ============================================================
# BASIC OUTPUT HELPERS
# ============================================================

def savefig_publication(fig: plt.Figure, stem: str) -> None:
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=400)
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved {png}")
    print(f"Saved {pdf}")


def save_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"Saved {path}")


# ============================================================
# QUATERNIONS / SU(2)
# ============================================================

def normalize_q(q, eps=1e-8):
    return q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), eps)


def qmul(a, b):
    w1, x1, y1, z1 = jnp.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = jnp.moveaxis(b, -1, 0)
    return jnp.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def qinv(q):
    w, x, y, z = jnp.moveaxis(q, -1, 0)
    return jnp.stack([w, -x, -y, -z], axis=-1)


def identity_links(N: int):
    Gx = jnp.zeros((N, N, N, 4))
    Gy = jnp.zeros((N, N, N, 4))
    Gz = jnp.zeros((N, N, N, 4))
    Gx = Gx.at[..., 0].set(1.0)
    Gy = Gy.at[..., 0].set(1.0)
    Gz = Gz.at[..., 0].set(1.0)
    return Gx, Gy, Gz


def rotate_vec(q, v):
    zero = jnp.zeros(v.shape[:-1] + (1,))
    vq = jnp.concatenate([zero, v], axis=-1)
    rq = qmul(qmul(q, vq), qinv(q))
    return rq[..., 1:]


# ============================================================
# GRID AND INITIAL CONDITIONS
# ============================================================

def make_grid(N: int):
    xs = jnp.linspace(-1.0, 1.0, N)
    X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
    R = jnp.sqrt(X**2 + Y**2 + Z**2)
    return X, Y, Z, R


def rational_map_degree_k(X, Y, Z, k: int):
    """Boundary orientational map based on R(z)=z^k."""
    eps = 1e-8
    Rr = jnp.sqrt(X**2 + Y**2 + Z**2)
    nx = X / jnp.maximum(Rr, eps)
    ny = Y / jnp.maximum(Rr, eps)
    nz = Z / jnp.maximum(Rr, eps)
    denom = jnp.maximum(1.0 - nz, eps)
    z = (nx + 1j * ny) / denom
    Rz = z ** k
    abs2 = jnp.real(Rz * jnp.conj(Rz))
    n1 = 2.0 * jnp.real(Rz) / (1.0 + abs2)
    n2 = 2.0 * jnp.imag(Rz) / (1.0 + abs2)
    n3 = (abs2 - 1.0) / (1.0 + abs2)
    n = jnp.stack([n1, n2, n3], axis=-1)
    n = jnp.where(Rr[..., None] < 1e-6, jnp.array([0.0, 0.0, 1.0]), n)
    return n


def core_centers(k: int, seed_id: int, mode: str = "mixed") -> List[Tuple[float, float, float]]:
    """
    Generate core centers. The boundary degree is set by the rational map;
    these centers only define the initial amplitude profile.
    """
    rng = np.random.default_rng(1000 * k + seed_id)
    if k == 1:
        if mode == "jitter":
            return [(float(rng.normal(0, 0.04)), float(rng.normal(0, 0.04)), float(rng.normal(0, 0.04)))]
        return [(0.0, 0.0, 0.0)]

    centers: List[Tuple[float, float, float]] = []
    selected = seed_id % 4 if mode == "mixed" else {"ring": 0, "irregular": 1, "line": 2, "compact": 3}.get(mode, 0)

    if selected == 0:
        radius = 0.42
        phase = rng.uniform(0, 2 * math.pi)
        for m in range(k):
            ang = phase + 2.0 * math.pi * m / k
            centers.append((radius * math.cos(ang), radius * math.sin(ang), 0.0))
    elif selected == 1:
        radius = 0.42
        phase = rng.uniform(0, 2 * math.pi)
        for m in range(k):
            ang = phase + 2.0 * math.pi * m / k + rng.normal(0, 0.18)
            r = radius * (1.0 + rng.normal(0, 0.12))
            centers.append((r * math.cos(ang), r * math.sin(ang), rng.normal(0, 0.06)))
    elif selected == 2:
        span = 0.75
        for m in range(k):
            t = -0.5 + m / max(1, k - 1)
            centers.append((span * t, rng.normal(0, 0.04), rng.normal(0, 0.04)))
    else:
        for _ in range(k):
            centers.append((rng.normal(0, 0.22), rng.normal(0, 0.22), rng.normal(0, 0.12)))
    return [(float(a), float(b), float(c)) for a, b, c in centers]


def amplitude_profile(X, Y, Z, centers: List[Tuple[float, float, float]], sharpness: float = 7.0):
    profile = jnp.ones(X.shape)
    for cx, cy, cz in centers:
        r = jnp.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        profile = profile * jnp.tanh(sharpness * r)
    return profile


def initial_phi(N: int, q_value: int, seed_id: int, V: float, init_mode: str):
    X, Y, Z, _ = make_grid(N)
    n = rational_map_degree_k(X, Y, Z, q_value)
    centers = core_centers(q_value, seed_id, init_mode)
    profile = amplitude_profile(X, Y, Z, centers)
    phi = V * profile[..., None] * n
    return phi, centers


def apply_phi_boundary(phi, boundary_phi):
    phi = phi.at[0, :, :, :].set(boundary_phi[0, :, :, :])
    phi = phi.at[-1, :, :, :].set(boundary_phi[-1, :, :, :])
    phi = phi.at[:, 0, :, :].set(boundary_phi[:, 0, :, :])
    phi = phi.at[:, -1, :, :].set(boundary_phi[:, -1, :, :])
    phi = phi.at[:, :, 0, :].set(boundary_phi[:, :, 0, :])
    phi = phi.at[:, :, -1, :].set(boundary_phi[:, :, -1, :])
    return phi


# ============================================================
# ENERGY AND OBSERVABLES
# ============================================================

def plaquette_xy(Gx, Gy):
    A = Gx[:-1, :-1, :, :]
    B = Gy[1:, :-1, :, :]
    C = qinv(Gx[:-1, 1:, :, :])
    D = qinv(Gy[:-1, :-1, :, :])
    return qmul(qmul(A, B), qmul(C, D))


def plaquette_xz(Gx, Gz):
    A = Gx[:-1, :, :-1, :]
    B = Gz[1:, :, :-1, :]
    C = qinv(Gx[:-1, :, 1:, :])
    D = qinv(Gz[:-1, :, :-1, :])
    return qmul(qmul(A, B), qmul(C, D))


def plaquette_yz(Gy, Gz):
    A = Gy[:, :-1, :-1, :]
    B = Gz[:, 1:, :-1, :]
    C = qinv(Gy[:, :-1, 1:, :])
    D = qinv(Gz[:, :-1, :-1, :])
    return qmul(qmul(A, B), qmul(C, D))


def covariant_energy(phi, Gx, Gy, Gz):
    Ex = jnp.sum((phi[:-1, :, :, :] - rotate_vec(Gx[:-1, :, :, :], phi[1:, :, :, :])) ** 2)
    Ey = jnp.sum((phi[:, :-1, :, :] - rotate_vec(Gy[:, :-1, :, :], phi[:, 1:, :, :])) ** 2)
    Ez = jnp.sum((phi[:, :, :-1, :] - rotate_vec(Gz[:, :, :-1, :], phi[:, :, 1:, :])) ** 2)
    return Ex + Ey + Ez


def curvature_energy(Gx, Gy, Gz):
    Pxy = plaquette_xy(Gx, Gy)
    Pxz = plaquette_xz(Gx, Gz)
    Pyz = plaquette_yz(Gy, Gz)
    return jnp.sum(1.0 - Pxy[..., 0]) + jnp.sum(1.0 - Pxz[..., 0]) + jnp.sum(1.0 - Pyz[..., 0])


def potential_energy(phi, V):
    norm2 = jnp.sum(phi**2, axis=-1)
    return jnp.sum((norm2 - V**2) ** 2)


def total_energy(phi, Gx, Gy, Gz, alpha_c, kappa, lambda_pot, V):
    return (
        alpha_c * covariant_energy(phi, Gx, Gy, Gz)
        + kappa * curvature_energy(Gx, Gy, Gz)
        + lambda_pot * potential_energy(phi, V)
    )


def phi_norm(phi):
    return jnp.sqrt(jnp.sum(phi**2, axis=-1))


def curvature_density(Gx, Gy, Gz):
    N = Gx.shape[0]
    rho = jnp.zeros((N, N, N))
    Pxy = plaquette_xy(Gx, Gy)
    Pxz = plaquette_xz(Gx, Gz)
    Pyz = plaquette_yz(Gy, Gz)
    rho = rho.at[:-1, :-1, :].add(1.0 - Pxy[..., 0])
    rho = rho.at[:-1, :, :-1].add(1.0 - Pxz[..., 0])
    rho = rho.at[:, :-1, :-1].add(1.0 - Pyz[..., 0])
    return rho / 3.0


def find_top_peaks(rho_jax, n_peaks: int, min_dist: int):
    rho = np.array(rho_jax)
    N = rho.shape[0]
    work = rho.copy()
    coords = []
    vals = []
    for _ in range(n_peaks):
        idx = int(np.argmax(work))
        val = float(work.reshape(-1)[idx])
        if val <= 1e-8:
            break
        c = np.unravel_index(idx, rho.shape)
        coords.append(tuple(int(v) for v in c))
        vals.append(val)
        x0, y0, z0 = c
        for i in range(max(0, x0 - min_dist), min(N, x0 + min_dist + 1)):
            for j in range(max(0, y0 - min_dist), min(N, y0 + min_dist + 1)):
                for k in range(max(0, z0 - min_dist), min(N, z0 + min_dist + 1)):
                    if np.linalg.norm(np.array([i - x0, j - y0, k - z0])) <= min_dist:
                        work[i, j, k] = -1.0
    return coords, vals


def geometry_label(vals):
    if not vals:
        return "none", 0
    maxv = max(vals)
    strong_vals = [v for v in vals if v >= 0.15 * maxv]
    n = len(strong_vals)
    if n == 1:
        return "single-core", n
    if n == 2:
        return "double-core", n
    if n == 3:
        return "triangular/three-core", n
    return f"{n}-core/extended", n


def pairwise_distances(coords):
    dists = []
    for a in range(len(coords)):
        for b in range(a + 1, len(coords)):
            dists.append(float(np.linalg.norm(np.array(coords[a]) - np.array(coords[b]))))
    return dists


def solid_angle(a, b, c):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)
    num = np.dot(a, np.cross(b, c))
    den = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    return 2.0 * math.atan2(num, den)


def boundary_degree(phi_jax):
    phi = np.array(phi_jax)
    N = phi.shape[0]
    n = phi / np.maximum(np.linalg.norm(phi, axis=-1, keepdims=True), 1e-12)
    total = 0.0

    def add_quad(v00, v10, v11, v01):
        nonlocal total
        total += solid_angle(v00, v10, v11)
        total += solid_angle(v00, v11, v01)

    k = N - 1
    for i in range(N - 1):
        for j in range(N - 1):
            add_quad(n[i, j, k], n[i+1, j, k], n[i+1, j+1, k], n[i, j+1, k])
    k = 0
    for i in range(N - 1):
        for j in range(N - 1):
            add_quad(n[i, j+1, k], n[i+1, j+1, k], n[i+1, j, k], n[i, j, k])
    i = N - 1
    for j in range(N - 1):
        for k in range(N - 1):
            add_quad(n[i, j, k], n[i, j+1, k], n[i, j+1, k+1], n[i, j, k+1])
    i = 0
    for j in range(N - 1):
        for k in range(N - 1):
            add_quad(n[i, j, k+1], n[i, j+1, k+1], n[i, j+1, k], n[i, j, k])
    j = N - 1
    for i in range(N - 1):
        for k in range(N - 1):
            add_quad(n[i, j, k], n[i, j, k+1], n[i+1, j, k+1], n[i+1, j, k])
    j = 0
    for i in range(N - 1):
        for k in range(N - 1):
            add_quad(n[i+1, j, k], n[i+1, j, k+1], n[i, j, k+1], n[i, j, k])
    return total / (4.0 * math.pi)


def stats(phi, Gx, Gy, Gz, alpha_c, kappa, lambda_pot, V, n_peaks, min_peak_dist):
    rho = curvature_density(Gx, Gy, Gz)
    nrm = phi_norm(phi)
    coords, vals = find_top_peaks(rho, n_peaks, min_peak_dist)
    geom, nstrong = geometry_label(vals)
    E_cov = float(covariant_energy(phi, Gx, Gy, Gz))
    E_curv = float(curvature_energy(Gx, Gy, Gz))
    E_pot = float(potential_energy(phi, V))
    E = float(alpha_c * E_cov + kappa * E_curv + lambda_pot * E_pot)
    return {
        "E": E,
        "E_cov": E_cov,
        "E_curv": E_curv,
        "E_pot": E_pot,
        "E_cov_weighted": alpha_c * E_cov,
        "E_curv_weighted": kappa * E_curv,
        "E_pot_weighted": lambda_pot * E_pot,
        "rho_mean": float(jnp.mean(rho)),
        "rho_max": float(jnp.max(rho)),
        "rho_std": float(jnp.std(rho)),
        "phi_min": float(jnp.min(nrm)),
        "phi_mean": float(jnp.mean(nrm)),
        "num_peaks": nstrong,
        "peak_coords": coords,
        "peak_vals": vals,
        "peak_dists": pairwise_distances(coords),
        "geometry": geom,
    }


# ============================================================
# RELAXATION STEP
# ============================================================

@jax.jit
def relax_step(phi, Gx, Gy, Gz, boundary_phi, lr, alpha_c, kappa, lambda_pot, V):
    grads = grad(total_energy, argnums=(0, 1, 2, 3))(phi, Gx, Gy, Gz, alpha_c, kappa, lambda_pot, V)
    phi_new = phi - lr * grads[0]
    Gx_new = normalize_q(Gx - lr * grads[1])
    Gy_new = normalize_q(Gy - lr * grads[2])
    Gz_new = normalize_q(Gz - lr * grads[3])
    phi_new = apply_phi_boundary(phi_new, boundary_phi)
    return phi_new, Gx_new, Gy_new, Gz_new


# ============================================================
# SINGLE RUN
# ============================================================

def run_case(
    N: int,
    q_value: int,
    seed_id: int,
    args,
):
    phi0, centers = initial_phi(N, q_value, seed_id, args.V, args.init_mode)
    boundary_phi = phi0
    phi = apply_phi_boundary(phi0, boundary_phi)
    Gx, Gy, Gz = identity_links(N)
    q_initial = boundary_degree(phi)
    last_E = None
    stable_count = 0
    converged = False
    final_step = args.steps
    final_delta_E = np.nan

    for step in range(args.steps):
        phi, Gx, Gy, Gz = relax_step(
            phi, Gx, Gy, Gz, boundary_phi,
            args.lr, args.alpha_c, args.kappa, args.lambda_pot, args.V,
        )

        if args.adaptive and (step % args.check_every == 0 or step == args.steps - 1):
            s = stats(phi, Gx, Gy, Gz, args.alpha_c, args.kappa, args.lambda_pot, args.V, args.n_peaks, args.min_peak_dist)
            if last_E is not None:
                dE = abs(s["E"] - last_E)
                final_delta_E = dE
                if step >= args.min_steps and dE < args.delta_E_tol:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= args.patience:
                    converged = True
                    final_step = step
                    break
            last_E = s["E"]

    s = stats(phi, Gx, Gy, Gz, args.alpha_c, args.kappa, args.lambda_pot, args.V, args.n_peaks, args.min_peak_dist)
    q_boundary = boundary_degree(phi)
    result = {
        "N": N,
        "Q_target": q_value,
        "seed_id": seed_id,
        "Q_initial": q_initial,
        "Q_boundary": q_boundary,
        "Q_boundary_abs_error": abs(q_boundary - q_value),
        "steps": final_step,
        "final_delta_E": final_delta_E,
        "converged": converged if args.adaptive else "fixed_steps",
        "alpha_c": args.alpha_c,
        "kappa": args.kappa,
        "lambda_pot": args.lambda_pot,
        "V": args.V,
        "init_mode": args.init_mode,
        "centers": centers,
        **s,
    }
    return result, phi, Gx, Gy, Gz


# ============================================================
# FITTING AND SUMMARY
# ============================================================

def fit_power_law_from_rows(rows: List[dict], energy_key: str = "E_min"):
    Q = np.array([float(r["Q_target"]) for r in rows], dtype=float)
    E = np.array([float(r[energy_key]) for r in rows], dtype=float)
    mask = (Q > 0) & (E > 0)
    Q = Q[mask]
    E = E[mask]
    x = np.log(Q)
    y = np.log(E)
    n = len(x)
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    sxx = float(np.sum((x - xbar) ** 2))
    sxy = float(np.sum((x - xbar) * (y - ybar)))
    alpha = sxy / sxx
    logC = ybar - alpha * xbar
    pred = logC + alpha * x
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - ybar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    sigma2 = ss_res / max(n - 2, 1)
    alpha_err = math.sqrt(sigma2 / sxx) if sxx > 0 else np.nan
    logC_err = math.sqrt(sigma2 * (1.0 / n + xbar * xbar / sxx)) if sxx > 0 else np.nan
    C = math.exp(logC)
    C_err = C * logC_err if np.isfinite(logC_err) else np.nan
    return {"C": C, "C_err": C_err, "alpha": alpha, "alpha_err": alpha_err, "R2": r2, "residuals": resid}


def summarize(all_rows: List[dict]):
    summary = []
    best = []
    key_groups: Dict[Tuple[int, int], List[dict]] = {}
    for r in all_rows:
        key_groups.setdefault((int(r["N"]), int(r["Q_target"])), []).append(r)

    for (N, Q), rows in sorted(key_groups.items()):
        Es = np.array([float(r["E"]) for r in rows], dtype=float)
        best_row = min(rows, key=lambda r: float(r["E"]))
        q_errors = np.array([abs(float(r["Q_boundary"]) - float(r["Q_target"])) for r in rows], dtype=float)
        summary.append({
            "N": N,
            "Q_target": Q,
            "n_seeds": len(rows),
            "E_min": float(np.min(Es)),
            "E_mean": float(np.mean(Es)),
            "E_std": float(np.std(Es, ddof=1)) if len(Es) > 1 else 0.0,
            "E_median": float(np.median(Es)),
            "E_max": float(np.max(Es)),
            "best_seed_id": best_row["seed_id"],
            "best_geometry": best_row["geometry"],
            "best_Q_boundary": best_row["Q_boundary"],
            "Q_boundary_mean_abs_error": float(np.mean(q_errors)),
            "Q_boundary_max_abs_error": float(np.max(q_errors)),
        })
        best.append(best_row)
    return summary, best


def fit_by_N(summary_rows: List[dict]) -> List[dict]:
    by_N: Dict[int, List[dict]] = {}
    for r in summary_rows:
        by_N.setdefault(int(r["N"]), []).append(r)
    fits = []
    for N, rows in sorted(by_N.items()):
        rows_sorted = sorted(rows, key=lambda x: int(x["Q_target"]))
        fit_min = fit_power_law_from_rows(rows_sorted, "E_min")
        fit_mean = fit_power_law_from_rows(rows_sorted, "E_mean")
        q_mean_err = float(np.mean([float(r.get("Q_boundary_mean_abs_error", np.nan)) for r in rows_sorted]))
        q_max_err = float(np.max([float(r.get("Q_boundary_max_abs_error", np.nan)) for r in rows_sorted]))
        fits.append({
            "N": N,
            "C_min": fit_min["C"],
            "C_min_err": fit_min["C_err"],
            "alpha_min": fit_min["alpha"],
            "alpha_min_err": fit_min["alpha_err"],
            "R2_min": fit_min["R2"],
            "C_mean": fit_mean["C"],
            "C_mean_err": fit_mean["C_err"],
            "alpha_mean": fit_mean["alpha"],
            "alpha_mean_err": fit_mean["alpha_err"],
            "R2_mean": fit_mean["R2"],
            "Q_boundary_mean_abs_error": q_mean_err,
            "Q_boundary_max_abs_error": q_max_err,
        })
    return fits


# ============================================================
# MORPHOLOGY SAVING
# ============================================================

def save_morphology_npz(N: int, q_value: int, seed_id: int, phi, Gx, Gy, Gz, result: dict, canonical: bool = False):
    rho = np.array(curvature_density(Gx, Gy, Gz))
    defect = 1.0 - np.array(phi_norm(phi))
    mid = N // 2
    filename = f"morphology_N{N}_Q{q_value}.npz"
    path = DATA_DIR / filename
    np.savez_compressed(
        path,
        N=N,
        Q=q_value,
        seed_id=seed_id,
        rho=rho,
        defect=defect,
        curv_xy=rho[:, :, mid].T,
        curv_xz=rho[:, mid, :].T,
        curv_yz=rho[mid, :, :].T,
        defect_xy=defect[:, :, mid].T,
        E=result.get("E", np.nan),
        Q_boundary=result.get("Q_boundary", np.nan),
        geometry=result.get("geometry", ""),
        peak_coords=str(result.get("peak_coords", "")),
        peak_vals=str(result.get("peak_vals", "")),
    )
    print(f"Saved morphology data: {path}")

    if canonical:
        canonical_path = DATA_DIR / f"morphology_Q{q_value}.npz"
        np.savez_compressed(
            canonical_path,
            N=N,
            Q=q_value,
            seed_id=seed_id,
            rho=rho,
            defect=defect,
            curv_xy=rho[:, :, mid].T,
            curv_xz=rho[:, mid, :].T,
            curv_yz=rho[mid, :, :].T,
            defect_xy=defect[:, :, mid].T,
            E=result.get("E", np.nan),
            Q_boundary=result.get("Q_boundary", np.nan),
            geometry=result.get("geometry", ""),
            peak_coords=str(result.get("peak_coords", "")),
            peak_vals=str(result.get("peak_vals", "")),
        )
        print(f"Saved canonical morphology data: {canonical_path}")


# ============================================================
# FIGURES
# ============================================================

def build_figure_scaling(summary_rows: List[dict], fits_rows: List[dict]):
    by_N: Dict[int, List[dict]] = {}
    for r in summary_rows:
        by_N.setdefault(int(r["N"]), []).append(r)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for N, rows in sorted(by_N.items()):
        rows = sorted(rows, key=lambda x: int(x["Q_target"]))
        Q = np.array([float(r["Q_target"]) for r in rows])
        Emin = np.array([float(r["E_min"]) for r in rows])
        Estd = np.array([float(r["E_std"]) for r in rows])
        ax.errorbar(Q, Emin, yerr=Estd, fmt="o", markersize=4, capsize=2, label=f"N={N}")
        fit = next(f for f in fits_rows if int(f["N"]) == N)
        qfit = np.linspace(Q.min(), Q.max(), 300)
        efit = float(fit["C_min"]) * qfit ** float(fit["alpha_min"])
        ax.plot(qfit, efit, "-", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"boundary degree $Q$")
    ax.set_ylabel(r"minimum energy $E_{\min}$")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    savefig_publication(fig, "robust_figure1_scaling_by_N")
    plt.close(fig)


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, yerr: np.ndarray | None = None) -> dict:
    """
    Fit y = slope*x + intercept.

    With only two lattice sizes, the line is exactly determined but the
    covariance matrix is not statistically meaningful. In that case we return
    finite slope/intercept and NaN uncertainties instead of failing. A proper
    alpha_infinity uncertainty requires at least three N values.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        mask &= np.isfinite(yerr) & (yerr > 0)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if yerr is not None:
        yerr = np.asarray(yerr[mask], dtype=float)

    if len(x) < 2:
        return {
            "intercept": np.nan, "intercept_err": np.nan,
            "slope": np.nan, "slope_err": np.nan, "R2": np.nan,
            "n_points": int(len(x)),
        }

    if len(x) == 2:
        coeff = np.polyfit(x, y, 1)
        slope, intercept = coeff
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            "intercept": float(intercept),
            "intercept_err": np.nan,
            "slope": float(slope),
            "slope_err": np.nan,
            "R2": r2,
            "n_points": int(len(x)),
        }

    if yerr is None:
        coeff, cov = np.polyfit(x, y, 1, cov=True)
    else:
        coeff, cov = np.polyfit(x, y, 1, w=1.0 / yerr, cov=True)

    slope, intercept = coeff
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "intercept": float(intercept),
        "intercept_err": float(np.sqrt(cov[1, 1])) if cov.shape == (2, 2) else np.nan,
        "slope": float(slope),
        "slope_err": float(np.sqrt(cov[0, 0])) if cov.shape == (2, 2) else np.nan,
        "R2": r2,
        "n_points": int(len(x)),
    }


def build_figure_alpha_vs_N(fits_rows: List[dict]) -> dict:
    Nvals = np.array([float(r["N"]) for r in fits_rows])
    alpha = np.array([float(r["alpha_min"]) for r in fits_rows])
    alpha_err = np.array([float(r["alpha_min_err"]) for r in fits_rows])
    x = 1.0 / Nvals
    fit = weighted_linear_fit(x, alpha, alpha_err)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.errorbar(x, alpha, yerr=alpha_err, fmt="o", capsize=2, label="fits by lattice size")
    if np.isfinite(fit["intercept"]):
        xfit = np.linspace(0.0, max(x) * 1.05, 200)
        yfit = fit["slope"] * xfit + fit["intercept"]
        ax.plot(xfit, yfit, "-", linewidth=1.2, label="linear extrapolation")
        if np.isfinite(fit.get("intercept_err", np.nan)):
            label_text = rf"$\alpha_{{\infty}} = {fit['intercept']:.3f} \pm {fit['intercept_err']:.3f}$"
        else:
            label_text = rf"$\alpha_{{\infty}} \approx {fit['intercept']:.3f}$" + "\n" + r"(2-point estimate)"
        ax.text(
            0.04, 0.96,
            label_text,
            transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75),
        )
    ax.set_xlabel(r"$1/N$")
    ax.set_ylabel(r"effective exponent $\alpha(N)$")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    savefig_publication(fig, "robust_figure_alpha_vs_inverse_N")
    plt.close(fig)
    return fit


def build_figure_boundary_error(summary_rows: List[dict]):
    by_N: Dict[int, List[dict]] = {}
    for r in summary_rows:
        by_N.setdefault(int(r["N"]), []).append(r)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for N, rows in sorted(by_N.items()):
        rows = sorted(rows, key=lambda x: int(x["Q_target"]))
        Q = np.array([float(r["Q_target"]) for r in rows])
        err = np.array([float(r["Q_boundary_max_abs_error"]) for r in rows])
        ax.plot(Q, err, "o-", markersize=4, linewidth=1.0, label=f"N={N}")
    ax.set_yscale("log")
    ax.set_xlabel(r"target boundary degree $Q$")
    ax.set_ylabel(r"max $|Q_{\mathrm{boundary}}-Q|$")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    savefig_publication(fig, "robust_figure_boundary_degree_error")
    plt.close(fig)


def build_figure_morphologies(q_values: Iterable[int], canonical_N: int):
    panels = []
    labels = []
    for q in q_values:
        path = DATA_DIR / f"morphology_N{canonical_N}_Q{q}.npz"
        if not path.exists():
            print(f"Missing morphology data for figure: {path}")
            return
        data = np.load(path, allow_pickle=True)
        panels.append(np.asarray(data["curv_xy"]))
        labels.append(f"Q = {q}")
    all_vals = np.concatenate([p.ravel() for p in panels])
    vmax = np.percentile(all_vals, 99)
    if vmax <= 0:
        vmax = float(np.max(all_vals)) if np.max(all_vals) > 0 else 1.0
    fig, axes = plt.subplots(1, len(panels), figsize=(2.15 * len(panels), 2.45), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    im = None
    for ax, panel, label in zip(axes, panels, labels):
        im = ax.imshow(panel, origin="lower", vmin=0.0, vmax=vmax, cmap="inferno")
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02)
    cbar.set_label(r"curvature density $\rho_{\mathrm{curv}}$")
    savefig_publication(fig, f"robust_figure_morphologies_N{canonical_N}")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Robust finite-size/multi-seed sweep for discrete orientational field model.")
    parser.add_argument("--N-values", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--Q-values", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--adaptive", action="store_true", help="Enable early stopping by energy convergence.")
    parser.add_argument("--min-steps", type=int, default=3000)
    parser.add_argument("--check-every", type=int, default=500)
    parser.add_argument("--delta-E-tol", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--alpha-c", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.2)
    parser.add_argument("--lambda-pot", type=float, default=5.0)
    parser.add_argument("--V", type=float, default=1.0)
    parser.add_argument("--init-mode", choices=["mixed", "ring", "irregular", "line", "compact", "jitter"], default="mixed")
    parser.add_argument("--n-peaks", type=int, default=8)
    parser.add_argument("--min-peak-dist", type=int, default=3)
    parser.add_argument("--morphology-Q-values", type=int, nargs="+", default=[1, 3, 5, 8, 10])
    parser.add_argument("--canonical-N", type=int, default=None, help="N used for morphology figure; defaults to largest N.")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    canonical_N = args.canonical_N if args.canonical_N is not None else max(args.N_values)

    param_rows = [{
        "N_values": " ".join(map(str, args.N_values)),
        "Q_values": " ".join(map(str, args.Q_values)),
        "seeds": args.seeds,
        "steps": args.steps,
        "adaptive": args.adaptive,
        "min_steps": args.min_steps,
        "lr": args.lr,
        "alpha_c": args.alpha_c,
        "kappa": args.kappa,
        "lambda_pot": args.lambda_pot,
        "V": args.V,
        "init_mode": args.init_mode,
    }]
    save_csv(PARAMS_CSV, param_rows, list(param_rows[0].keys()))

    all_rows: List[dict] = []
    fields_runtime = [
        "N", "Q_target", "seed_id", "Q_initial", "Q_boundary", "Q_boundary_abs_error", "steps", "final_delta_E", "converged",
        "alpha_c", "kappa", "lambda_pot", "V", "init_mode",
        "E", "E_cov", "E_curv", "E_pot", "E_cov_weighted", "E_curv_weighted", "E_pot_weighted",
        "rho_max", "rho_mean", "rho_std", "num_peaks", "geometry", "phi_min", "phi_mean",
        "peak_coords", "peak_vals", "peak_dists", "centers",
    ]

    best_fields_store: Dict[Tuple[int, int], Tuple[dict, object, object, object, object]] = {}

    for N in args.N_values:
        for q in args.Q_values:
            for seed_id in range(args.seeds):
                print(f"\n=== RUN N={N} Q={q} seed={seed_id} ===")
                result, phi, Gx, Gy, Gz = run_case(N, q, seed_id, args)
                all_rows.append(result)
                key = (N, q)
                if key not in best_fields_store or float(result["E"]) < float(best_fields_store[key][0]["E"]):
                    best_fields_store[key] = (result, phi, Gx, Gy, Gz)
                print(
                    f"N={N} Q={q} seed={seed_id} E={result['E']:.6f} "
                    f"Qb={result['Q_boundary']:.6f} geom={result['geometry']}"
                )
                # Incremental save protects against interruptions.
                save_csv(ALL_RUNS_CSV, all_rows, fields_runtime)

    summary_rows, best_rows = summarize(all_rows)
    fits_rows = fit_by_N(summary_rows)

    summary_fields = [
        "N", "Q_target", "n_seeds", "E_min", "E_mean", "E_std", "E_median", "E_max",
        "best_seed_id", "best_geometry", "best_Q_boundary",
        "Q_boundary_mean_abs_error", "Q_boundary_max_abs_error",
    ]
    best_fields = fields_runtime
    fits_fields = [
        "N", "C_min", "C_min_err", "alpha_min", "alpha_min_err", "R2_min",
        "C_mean", "C_mean_err", "alpha_mean", "alpha_mean_err", "R2_mean",
        "Q_boundary_mean_abs_error", "Q_boundary_max_abs_error",
    ]

    save_csv(SUMMARY_CSV, summary_rows, summary_fields)
    save_csv(BEST_CSV, best_rows, best_fields)
    save_csv(FITS_CSV, fits_rows, fits_fields)

    # Save representative morphology fields for selected Q at canonical_N.
    for q in args.morphology_Q_values:
        key = (canonical_N, q)
        if key in best_fields_store:
            result, phi, Gx, Gy, Gz = best_fields_store[key]
            save_morphology_npz(canonical_N, q, int(result["seed_id"]), phi, Gx, Gy, Gz, result, canonical=True)
        else:
            print(f"No best morphology available for N={canonical_N}, Q={q}")

    print("\n=== FITS BY N ===")
    for f in fits_rows:
        print(
            f"N={int(f['N']):3d}: alpha_min={float(f['alpha_min']):.6f} ± {float(f['alpha_min_err']):.6f}, "
            f"R2={float(f['R2_min']):.6f}; alpha_mean={float(f['alpha_mean']):.6f}"
        )

    extrap_fit = build_figure_alpha_vs_N(fits_rows) if not args.no_figures else weighted_linear_fit(
        np.array([1.0 / float(r["N"]) for r in fits_rows]),
        np.array([float(r["alpha_min"]) for r in fits_rows]),
        np.array([float(r["alpha_min_err"]) for r in fits_rows]),
    )
    extrap_rows = [{
        "fit": "alpha_min_vs_inverse_N_linear",
        "alpha_infinity": extrap_fit["intercept"],
        "alpha_infinity_err": extrap_fit["intercept_err"],
        "slope": extrap_fit["slope"],
        "slope_err": extrap_fit["slope_err"],
        "R2": extrap_fit["R2"],
        "n_points": extrap_fit.get("n_points", ""),
    }]
    save_csv(ALPHA_EXTRAP_CSV, extrap_rows, list(extrap_rows[0].keys()))

    if not args.no_figures:
        build_figure_scaling(summary_rows, fits_rows)
        build_figure_boundary_error(summary_rows)
        build_figure_morphologies(args.morphology_Q_values, canonical_N)


if __name__ == "__main__":
    main()

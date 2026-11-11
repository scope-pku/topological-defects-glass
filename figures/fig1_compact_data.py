#!/usr/bin/env python3
"""
fig1_compact_data.py

Data stage for the compact Fig. 1 (see plot_fig1_compact.py).
Reads the raw simulation output (same sources and parameters as
figures_v25/plotting_scripts/plot_topological_defects_PRL.py) and writes
everything the figure needs into one small cache file, so the layout can be
iterated on in seconds:

    fig1_compact_cache.npz   (next to this script)

Usage:  python fig1_compact_data.py
"""

import os
import pickle

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "fig1_compact_cache.npz")

# ---------------------------------------------------------------------------
# Parameters (identical to plot_topological_defects_PRL.py)
# ---------------------------------------------------------------------------
num_particles = 32000
system_size = 192.897128868165
pixel_density_multiplier = 2
conf_idx = 1
fraction = 0.10  # rDOS + defect pickles
fraction_field = 0.05  # eigenvector / longitudinal field files
cooling_rate = "10e-4"
num_modes_total = 2 * num_particles
num_confs, exclude_conf = 48, 45
freq_max, num_bins = 6.0, 200

# psi4 (N = 128000 system, as in the PRL script)
num_particles_128k, num_confs_128k = 128000, 12
fraction_128k = fraction_psi4_128k = 0.05
delta_omega, weight_threshold = 0.1, 0.20

mode_indices = [22, 271]  # omega ~ 0.4, 1.2
q_resolutions = [15, 30]  # k-space resolution of the phonon order parameter

base_dir = os.path.join(
    os.path.expanduser("~"), "research", "plastic_event", "IPL10",
    f"size_{num_particles}p", f"coolingrate_{cooling_rate}",
)
base_128k = os.path.join(
    os.path.expanduser("~"), "research", "plastic_event", "IPL10",
    f"size_{num_particles_128k}p", f"coolingrate_{cooling_rate}",
)
D = {
    "eigenvalues": os.path.join(base_dir, "eigensystems", "numerical", "eigenvalues"),
    "eigenvectors": os.path.join(base_dir, "eigensystems", "numerical", "eigenvectors"),
    "data": os.path.join(base_dir, "data"),
    "long": os.path.join(base_dir, "longitudinal_components"),
    "modulus": os.path.join(base_dir, "modulus", "deform_1e-06"),
    "defects_transverse": os.path.join(base_dir, "topological_defects_transverse"),
    "defects_longitudinal": os.path.join(base_dir, "topological_defects_longitudinal"),
}


# ---------------------------------------------------------------------------
# Helpers (copied from plot_topological_defects_PRL.py)
# ---------------------------------------------------------------------------
def read_lammps_data_2d(path, n_atoms):
    coords = []
    with open(path) as fh:
        for line in fh:
            if line.strip().startswith("Atoms"):
                next(fh)
                break
        for _ in range(n_atoms):
            d = fh.readline().split()
            coords.append((int(d[0]) - 1, float(d[2]), float(d[3])))
    coords.sort(key=lambda c: c[0])
    return np.array([[c[1], c[2]] for c in coords])


def read_moduli(directory, n_confs, excl=None):
    K, G = [], []
    for conf in range(1, n_confs + 1):
        if conf == excl:
            continue
        path = os.path.join(directory, f"{conf:03d}.modulus")
        if not os.path.isfile(path):
            continue
        with open(path) as fp:
            lines = fp.readlines()
        K.append(float(lines[11].split()[3]))
        G.append(float(lines[12].split()[3]))
    return float(np.mean(K)), float(np.mean(G))


def lognormal_func(omega, A, mu, sigma):
    return (A / omega) * np.exp(-((np.log(omega) - mu) ** 2) / (2 * sigma**2))


def fit_lognormal(bin_centers, rdos):
    m = (bin_centers > 0) & np.isfinite(rdos)
    i = np.argmax(rdos[m])
    p0 = [rdos[m][i], np.log(bin_centers[m][i]), 0.5]
    popt, _ = curve_fit(lognormal_func, bin_centers[m], rdos[m], p0=p0, maxfev=10000)
    return float(np.exp(popt[1] - popt[2] ** 2)), popt


def phonon_order_parameter(coords, disp, L, q_resolution):
    if q_resolution % 2 == 0:
        q_resolution += 1
    n = len(coords)
    n_max = (q_resolution - 1) // 2
    q = np.arange(-n_max, n_max + 1) * 2.0 * np.pi / L
    O_long = np.zeros((q_resolution, q_resolution))
    O_trans = np.zeros((q_resolution, q_resolution))
    for i, qx in enumerate(q):
        for j, qy in enumerate(q):
            qv = np.array([qx, qy])
            qn = np.linalg.norm(qv)
            if qn < 1e-9:
                continue
            qh = qv / qn
            F = np.sum(disp * np.exp(1j * coords @ qv)[:, None], axis=0)
            O_long[j, i] = np.abs(F @ qh) ** 2 / n
            O_trans[j, i] = np.abs(F @ np.array([-qh[1], qh[0]])) ** 2 / n
    return q, O_long, O_trans


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
def main():
    out = {}

    # ---- reduced VDOS (N = 32000, 47 configurations) ----------------------
    freqs, wl = [], []
    for conf in range(1, num_confs + 1):
        if conf == exclude_conf:
            continue
        ev = np.load(os.path.join(
            D["eigenvalues"], f"eigenvalues_fraction_{fraction:.2f}_conf_{conf:03d}.npy"))[2:]
        w = np.load(os.path.join(
            D["long"], "vdos_weights",
            f"longitudinal_vdos_weights_frac_{fraction:.2f}_conf_{conf:03d}.npy"))[2:]
        freqs.append(np.sqrt(ev))
        wl.append(w)
    n_valid = len(freqs)
    freqs, wl = np.concatenate(freqs), np.concatenate(wl)
    edges = np.linspace(0, freq_max, num_bins + 1)
    bc = 0.5 * (edges[:-1] + edges[1:])
    bw = edges[1] - edges[0]
    norm = n_valid * (num_modes_total - 2) * bw

    def rdos(weights=None):
        h, _ = np.histogram(freqs, bins=edges, weights=weights)
        return (h / norm) / bc

    out["bin_centers"] = bc
    out["rdos_total"] = rdos()
    out["rdos_trans"] = rdos(1.0 - wl)
    out["rdos_long"] = rdos(wl)
    K, G = read_moduli(D["modulus"], num_confs, exclude_conf)
    out["debye_trans"] = 1.0 / (4.0 * np.pi * G)
    out["debye_long"] = 1.0 / (4.0 * np.pi * (K + G))
    out["debye_total"] = out["debye_trans"] + out["debye_long"]
    omega_bp, popt = fit_lognormal(bc, out["rdos_total"])
    out["omega_bp"], out["lognormal_popt"] = omega_bp, popt
    print(f"[rDOS] {n_valid} confs, omega_BP = {omega_bp:.4f}")

    # ---- <psi4>(omega), N = 128000 ----------------------------------------
    d128 = {
        "eig": os.path.join(base_128k, "eigensystems", "numerical", "eigenvalues"),
        "w": os.path.join(base_128k, "longitudinal_components", "vdos_weights"),
        "psi_T": os.path.join(base_128k, "psi4_defects_transverse"),
        "psi_L": os.path.join(base_128k, "psi4_defects_longitudinal"),
        "psi_tot": os.path.join(base_128k, "psi4_defects_total"),
        "mod": os.path.join(base_128k, "modulus", "deform_1e-06"),
    }
    f_all, w_all, pT, pL, pTot = [], [], [], [], []
    for conf in range(1, num_confs_128k + 1):
        try:
            ev = np.load(os.path.join(
                d128["eig"], f"eigenvalues_fraction_{fraction_128k:.2f}_conf_{conf:03d}.npy"))
            n_off = len(ev)
            ev = ev[2:]
            w = np.load(os.path.join(
                d128["w"], f"longitudinal_vdos_weights_frac_{fraction_128k:.2f}_conf_{conf:03d}.npy"))[2:]
            tag = f"frac_{fraction_psi4_128k:.2f}_conf_{conf:03d}.npy"
            pt = np.load(os.path.join(d128["psi_T"], f"avg_psi4_defects_transverse_{tag}"))[2:n_off]
            pl = np.load(os.path.join(d128["psi_L"], f"avg_psi4_defects_longitudinal_{tag}"))[2:n_off]
            ptot = np.load(os.path.join(d128["psi_tot"], f"avg_psi4_defects_total_{tag}"))[2:n_off]
        except Exception as e:  # mirror the PRL script: skip missing configs
            print(f"  [128k] conf {conf}: {e}. Skipping.")
            continue
        f_all.append(np.sqrt(np.maximum(ev, 0.0)))
        w_all.append(w)
        pT.append(pt)
        pL.append(pl)
        pTot.append(ptot)
    f_all, w_all = np.concatenate(f_all), np.concatenate(w_all)
    pT, pL, pTot = np.concatenate(pT), np.concatenate(pL), np.concatenate(pTot)

    K128, G128 = read_moduli(d128["mod"], num_confs_128k)
    L128 = np.sqrt(num_particles_128k / 0.86)
    qmin = 2.0 * np.pi / L128
    wT_min, wL_min = np.sqrt(G128 / 0.86) * qmin, np.sqrt((K128 + G128) / 0.86) * qmin
    axis_T = np.linspace(wT_min, 4.0, 10)
    axis_L = np.linspace(wL_min, 4.0, 10)

    def bin_psi(axis, psi, filt):
        res = []
        for fc in axis:
            m = (f_all >= fc - delta_omega / 2) & (f_all < fc + delta_omega / 2)
            v = psi[m][filt[m] > weight_threshold] if filt is not None else psi[m]
            res.append(np.mean(v) if v.size else np.nan)
        return np.array(res)

    out["psi4_w_T"], out["psi4_T"] = axis_T, bin_psi(axis_T, pT, 1.0 - w_all)
    out["psi4_w_L"], out["psi4_L"] = axis_L, bin_psi(axis_L, pL, w_all)
    out["psi4_w_tot"], out["psi4_tot"] = axis_T, bin_psi(axis_T, pTot, None)
    print("[psi4] T:", np.round(out["psi4_T"], 3))
    print("[psi4] L:", np.round(out["psi4_L"], 3))

    # ---- eigenvector fields, defects, k-space maps (N = 32000, conf 1) ----
    coords = read_lammps_data_2d(
        os.path.join(D["data"], f"lammps_data_file_{conf_idx:03}.dat"), num_particles)
    out["coords"] = coords
    evals = np.load(os.path.join(
        D["eigenvalues"], f"eigenvalues_fraction_{fraction_field:.2f}_conf_{conf_idx:03}.npy"))
    evecs = np.load(os.path.join(
        D["eigenvectors"], f"eigenvectors_fraction_{fraction_field:.2f}_conf_{conf_idx:03}.npy"),
        mmap_mode="r")
    longi = np.load(os.path.join(
        D["long"], f"longitudinal_fraction_{fraction_field:.2f}_conf_{conf_idx:03}.npy"),
        mmap_mode="r")
    n_px = pixel_density_multiplier * int(np.ceil(np.sqrt(num_particles)))
    px = system_size / n_px

    pick = {}
    for comp, tag in (("transverse", "trans"), ("longitudinal", "long")):
        for sign in ("pos", "neg"):
            pick[(comp, sign)] = load_pickle(os.path.join(
                D[f"defects_{comp}"],
                f"{sign}_topo_defects_{tag}_fraction_{fraction:.2f}_conf_{conf_idx:03}.pickle"))

    out["mode_indices"] = np.array(mode_indices)
    out["mode_omegas"] = np.sqrt(evals[mode_indices])
    for idx, qres in zip(mode_indices, q_resolutions):
        whole = np.array(evecs[:, idx]).reshape(-1, 2)
        dl = np.array(longi[:, idx]).reshape(-1, 2)
        dt = whole - dl
        print(f"[mode {idx}] omega = {np.sqrt(evals[idx]):.4f}")
        for comp, disp in (("transverse", dt), ("longitudinal", dl)):
            key = f"{comp[0]}{idx}"
            out[f"disp_{key}"] = disp.astype(np.float32)
            for sign in ("pos", "neg"):
                d = pick[(comp, sign)][idx]
                out[f"{sign}_{key}"] = np.array(d, dtype=float).reshape(-1, 2) * px
            nrm = np.linalg.norm(disp)
            q, O_l, O_t = phonon_order_parameter(coords, disp / nrm, system_size, qres)
            out[f"q_{key}"] = q
            out[f"Ok_{key}"] = O_t if comp == "transverse" else O_l

    np.savez_compressed(CACHE, **out)
    print(f"wrote {CACHE}")


if __name__ == "__main__":
    main()

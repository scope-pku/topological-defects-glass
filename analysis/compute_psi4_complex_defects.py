"""
compute_psi4_complex_defects.py

This script computes the real and imaginary parts of the \psi_4 order parameter for
each topological defect in the vibrational modes of a 2D amorphous solid.  It processes
the transverse, total, and/or longitudinal components based on user-defined flags and
saves the results as pickled lists of 2D NumPy arrays, including defect positions
(x, y), Re(\psi_4), and Im(\psi_4).

The script reads box dimensions (Lx, Ly) directly from the LAMMPS data files so that it
is consistent with the glass model used by identify_topo_defects.py.

Parameters (command-line):
  num_particles   : Number of particles in the system.
  start_index     : Starting configuration index.
  end_index       : Ending configuration index.

Hard-coded parameters (match identify_topo_defects.py):
  fraction_td     : Fraction of total modes used for defect identification (default: 0.10).
  psi_fraction    : Fraction of total modes to process for psi4 (default: 0.10).
  pixel_multiplier: Multiplier for grid resolution (default: 2).
  compute_total / compute_transverse / compute_longitudinal : Boolean flags.

Input:
  - LAMMPS data files in 'data/lammps_data_file_{conf_idx:03d}.dat'.
  - Defect pickle files produced by identify_topo_defects.py:
      topological_defects_total/   pos/neg_topo_defects_{fraction_str}_conf_{idx:03}.pickle
      topological_defects_long/    pos/neg_topo_defects_long_{fraction_str}_conf_{idx:03}.pickle
      topological_defects_trans/   pos/neg_topo_defects_trans_{fraction_str}_conf_{idx:03}.pickle

Output:
  - Pickled lists of 2D NumPy arrays [x, y, Re(\psi_4), Im(\psi_4)] per mode, saved in:
      psi4_complex_defects_total/
      psi4_complex_defects_longitudinal/
      psi4_complex_defects_transverse/
"""

import argparse
import os
import pickle

import numpy as np

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compute psi4 complex order parameter at defects")
parser.add_argument("num_particles", type=int, help="Number of particles in the system")
parser.add_argument("start_index",   type=int, help="Starting configuration index")
parser.add_argument("end_index",     type=int, help="Ending configuration index")
args = parser.parse_args()

num_particles = args.num_particles
start_index   = args.start_index
end_index     = args.end_index

# ---------------------------------------------------------------------------
# Hard-coded parameters (keep in sync with identify_topo_defects.py)
# ---------------------------------------------------------------------------
fraction_td   = 0.05   # fraction of modes used for defect identification
psi_fraction  = 0.05   # fraction of total modes to compute psi4 for
pixel_multiplier = 2   # grid resolution multiplier

# Boolean flags
compute_total        = True
compute_transverse   = True
compute_longitudinal = True

# ---------------------------------------------------------------------------
# Derived mode counts
# ---------------------------------------------------------------------------
total_modes          = 2 * num_particles
num_modes_to_process = int(psi_fraction * total_modes) if psi_fraction is not None else total_modes

# fraction string that matches what identify_topo_defects.py writes
fraction_str = f"fraction_{fraction_td:.2f}"

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
base_dir = os.getcwd()

defects_dirs = {
    "total":        os.path.join(base_dir, "topological_defects_total"),
    "longitudinal": os.path.join(base_dir, "topological_defects_longitudinal"),
    "transverse":   os.path.join(base_dir, "topological_defects_transverse"),
}
output_dirs = {
    "total":        os.path.join(base_dir, "psi4_complex_defects_total"),
    "longitudinal": os.path.join(base_dir, "psi4_complex_defects_longitudinal"),
    "transverse":   os.path.join(base_dir, "psi4_complex_defects_transverse"),
}
for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

# File-name prefix for each component (matches identify_topo_defects.py)
component_abbr = {
    "total":        "",       # no component suffix for total
    "longitudinal": "long_",
    "transverse":   "trans_",
}

# Components to process
components_to_process = []
if compute_total:        components_to_process.append("total")
if compute_transverse:   components_to_process.append("transverse")
if compute_longitudinal: components_to_process.append("longitudinal")

# ---------------------------------------------------------------------------
# Helper: read LAMMPS data file for box size (same logic as identify_topo_defects.py)
# ---------------------------------------------------------------------------

def read_lammps_box(num_atoms: int, path_to_data: str):
    """Read box dimensions Lx and Ly from a LAMMPS data file."""
    Lx = Ly = 0.0
    with open(path_to_data, "r") as fh:
        for line in fh:
            data = line.split()
            if not data:
                continue
            if len(data) >= 4 and data[2] == "xlo" and data[3] == "xhi":
                Lx = float(data[1]) - float(data[0])
            elif len(data) >= 4 and data[2] == "ylo" and data[3] == "yhi":
                Ly = float(data[1]) - float(data[0])
    return Lx, Ly


# ---------------------------------------------------------------------------
# Helper: pixel_size from box + grid (matches identify_topo_defects.py grid logic)
# ---------------------------------------------------------------------------

def compute_grid_params(num_atoms: int, box_lx: float, box_ly: float, px_mult: int):
    """
    Compute the rectangular grid dimensions and (geometric-mean) pixel size
    using the same logic as identify_topo_defects.py::compute_orientational_field.

    Returns:
        grid_x, grid_y : int – number of grid cells along each axis.
        pixel_size      : float – geometric mean pixel size, used for coordinate conversion.
    """
    approx_linear_density = np.sqrt(num_atoms / (box_lx * box_ly))
    grid_x = px_mult * max(1, int(np.ceil(approx_linear_density * box_lx)))
    grid_y = px_mult * max(1, int(np.ceil(approx_linear_density * box_ly)))
    pixel_size_x = box_lx / grid_x
    pixel_size_y = box_ly / grid_y
    # geometric mean – consistent with how the smoothing kernel was isotropised
    pixel_size = np.sqrt(pixel_size_x * pixel_size_y)
    return grid_x, grid_y, pixel_size_x, pixel_size_y


# ---------------------------------------------------------------------------
# Psi-k computation helpers
# ---------------------------------------------------------------------------

def _compute_periodic_differences_rect(positions: np.ndarray, lx: float, ly: float):
    """Pairwise distances and angles under a rectangular PBC box."""
    x, y  = positions[:, 0], positions[:, 1]
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dx -= lx * np.round(dx / lx)
    dy -= ly * np.round(dy / ly)
    dist_matrix  = np.sqrt(dx**2 + dy**2)
    theta_matrix = np.arctan2(dy, dx)
    return dist_matrix, theta_matrix


def compute_psi_k_large_N(defect_positions: np.ndarray, lx: float, ly: float, k: int):
    """
    Compute ψ_k for each defect using the k nearest neighbours (minimum-image, rectangular box).
    Efficient for n > k.
    """
    dist_matrix, theta_matrix = _compute_periodic_differences_rect(defect_positions, lx, ly)
    neighbors       = np.argsort(dist_matrix, axis=1)[:, 1:k + 1]
    theta_neighbors = theta_matrix[np.arange(len(defect_positions))[:, np.newaxis], neighbors]
    return np.mean(np.exp(1j * k * theta_neighbors), axis=1)


def compute_psi_k_replication(defect_positions: np.ndarray, lx: float, ly: float, k: int):
    """
    Compute ψ_k for each defect using a 3×3 periodic-image replication.
    Handles n ≤ k (sparse defect cases).
    """
    n = len(defect_positions)
    if n == 0:
        return np.array([], dtype=complex)

    shift_x = np.arange(-1, 2) * lx
    shift_y = np.arange(-1, 2) * ly
    shifts  = np.array(np.meshgrid(shift_x, shift_y)).T.reshape(-1, 2)
    extended = np.vstack([defect_positions + s for s in shifts])

    psi_k_list = []
    for i in range(n):
        pos  = defect_positions[i]
        dx   = extended[:, 0] - pos[0]
        dy   = extended[:, 1] - pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        angles = np.arctan2(dy, dx)
        mask   = dist > 1e-10
        dist, angles = dist[mask], angles[mask]
        num_nb = min(k, len(dist))
        if num_nb == 0:
            psi_k_list.append(0j)
            continue
        idx = np.argsort(dist)[:num_nb]
        psi_k_list.append(np.mean(np.exp(1j * k * angles[idx])))

    return np.array(psi_k_list)


def compute_psi_k_defects(defect_positions: np.ndarray, lx: float, ly: float, k: int):
    """
    Dispatch to replication (n ≤ k) or minimum-image (n > k) ψ_k computation.
    """
    n = len(defect_positions)
    if n <= k:
        return compute_psi_k_replication(defect_positions, lx, ly, k)
    return compute_psi_k_large_N(defect_positions, lx, ly, k)


# ---------------------------------------------------------------------------
# Helper: convert grid indices → spatial coordinates (rectangular pixels)
# ---------------------------------------------------------------------------

def grid_to_coordinates(grid_indices: list, pixel_size_x: float, pixel_size_y: float):
    """
    Convert [i, j] grid indices to (x, y) spatial coordinates.

    Args:
        grid_indices (list): list of [row_i, col_j] pairs.
        pixel_size_x (float): pixel size along x (= Lx / grid_x).
        pixel_size_y (float): pixel size along y (= Ly / grid_y).

    Returns:
        np.ndarray: shape (n, 2) with columns [x, y].
    """
    arr = np.array(grid_indices, dtype=float)          # shape (n, 2): [row, col]
    # row index → x,  col index → y  (consistent with orientational field layout)
    x = arr[:, 0] * pixel_size_x
    y = arr[:, 1] * pixel_size_y
    return np.column_stack((x, y))


# ---------------------------------------------------------------------------
# Helper: load defect pickle files
# ---------------------------------------------------------------------------

def load_defects(component, conf_index, start_mode, end_mode):
    """
    Load positive and negative defect lists from pickle files produced by
    identify_topo_defects.py.

    File naming convention:
      total        : pos_topo_defects_{fraction_str}_conf_{conf_index:03}.pickle
      longitudinal : pos_topo_defects_long_{fraction_str}_conf_{conf_index:03}.pickle
      transverse   : pos_topo_defects_trans_{fraction_str}_conf_{conf_index:03}.pickle
    """
    abbr     = component_abbr[component]           # "", "long_", or "trans_"
    td_suffix = f"{fraction_str}_conf_{conf_index:03d}.pickle"
    pos_file  = f"pos_topo_defects_{abbr}{td_suffix}"
    neg_file  = f"neg_topo_defects_{abbr}{td_suffix}"

    path_pos = os.path.join(defects_dirs[component], pos_file)
    path_neg = os.path.join(defects_dirs[component], neg_file)

    with open(path_pos, "rb") as f:
        pos_defects = pickle.load(f)[start_mode:end_mode]
    with open(path_neg, "rb") as f:
        neg_defects = pickle.load(f)[start_mode:end_mode]
    return pos_defects, neg_defects


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

data_dir = os.path.join(base_dir, "data")

for conf_index in range(start_index, end_index + 1):
    print(f"Processing configuration {conf_index}...")

    # Read box geometry from LAMMPS data file (same source as identify_topo_defects.py)
    lammps_file = os.path.join(data_dir, f"lammps_data_file_{conf_index:03d}.dat")
    box_lx, box_ly = read_lammps_box(num_particles, lammps_file)
    print(f"  Box size: Lx={box_lx:.4f}, Ly={box_ly:.4f}")

    # Derive pixel sizes consistent with the orientational field grid
    _, _, pixel_size_x, pixel_size_y = compute_grid_params(
        num_particles, box_lx, box_ly, pixel_multiplier
    )

    for component in components_to_process:
        print(f"  Component: {component}")

        # Load defects
        pos_defects, neg_defects = load_defects(
            component, conf_index, 0, num_modes_to_process
        )

        if len(pos_defects) < num_modes_to_process:
            raise ValueError(
                f"Defect file for conf {conf_index}, component {component} has "
                f"{len(pos_defects)} modes, but {num_modes_to_process} requested."
            )

        # Compute ψ_4 for every mode
        psi_results = []
        for mode_index in range(num_modes_to_process):
            all_defects = pos_defects[mode_index] + neg_defects[mode_index]
            if len(all_defects) == 0:
                psi_results.append(
                    np.array([[np.nan, np.nan, np.nan, np.nan]], dtype=float)
                )
                continue

            defect_positions = grid_to_coordinates(all_defects, pixel_size_x, pixel_size_y)
            psi4 = compute_psi_k_defects(defect_positions, box_lx, box_ly, k=4)

            x          = defect_positions[:, 0]
            y          = defect_positions[:, 1]
            re_psi4    = psi4.real
            im_psi4    = psi4.imag
            psi_results.append(np.column_stack((x, y, re_psi4, im_psi4)))

        # Save
        frac_str_out = "all" if psi_fraction is None else f"{psi_fraction:.2f}"
        output_file  = os.path.join(
            output_dirs[component],
            f"psi4_complex_defects_{component}_frac_{frac_str_out}_conf_{conf_index:03d}.pickle",
        )
        with open(output_file, "wb") as f:
            pickle.dump(psi_results, f)
        print(f"    Saved to {output_file}")

print("All configurations processed.")

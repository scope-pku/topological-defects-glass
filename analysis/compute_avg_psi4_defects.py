"""
compute_avg_psi4_defects.py

Computes the averaged |psi_4| per mode by reading the complex psi_4 values
already computed by compute_psi4_complex_defects.py.

For each mode the input file contains a NumPy array of shape (n_defects, 4)
with columns [x, y, Re(psi4), Im(psi4)], or [[nan, nan, nan, nan]] for
topologically trivial (zero-defect) modes.

avg_psi4[mode] = |mean(psi4)|  over all defects in that mode
               = 0.0           for trivial modes (NaN row)

Usage:
  python compute_avg_psi4_defects.py <num_particles> <start_index> <end_index>

Input (written by compute_psi4_complex_defects.py):
  psi4_complex_defects_total/        psi4_complex_defects_total_frac_{frac}_conf_{idx:03d}.pickle
  psi4_complex_defects_longitudinal/ psi4_complex_defects_longitudinal_frac_{frac}_conf_{idx:03d}.pickle
  psi4_complex_defects_transverse/   psi4_complex_defects_transverse_frac_{frac}_conf_{idx:03d}.pickle

Output:
  psi4_defects_total/        avg_psi4_defects_total_frac_{frac}_conf_{idx:03d}.npy
  psi4_defects_longitudinal/ avg_psi4_defects_longitudinal_frac_{frac}_conf_{idx:03d}.npy
  psi4_defects_transverse/   avg_psi4_defects_transverse_frac_{frac}_conf_{idx:03d}.npy
"""

import argparse
import os
import pickle

import numpy as np

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compute averaged |psi4| from complex psi4 files")
parser.add_argument("num_particles", type=int, help="Number of particles in the system")
parser.add_argument("start_index",   type=int, help="Starting configuration index")
parser.add_argument("end_index",     type=int, help="Ending configuration index")
args = parser.parse_args()

num_particles = args.num_particles
start_index   = args.start_index
end_index     = args.end_index

# ---------------------------------------------------------------------------
# Hard-coded parameters — must match compute_psi4_complex_defects.py
# ---------------------------------------------------------------------------
psi4_fraction = 0.05   # fraction of modes that were processed

# Boolean flags
compute_total        = True
compute_transverse   = True
compute_longitudinal = True

# ---------------------------------------------------------------------------
# Derived
# ---------------------------------------------------------------------------
total_modes          = 2 * num_particles
num_modes_to_process = int(psi4_fraction * total_modes) if psi4_fraction is not None else total_modes
frac_str             = "all" if psi4_fraction is None else f"{psi4_fraction:.2f}"

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
base_dir = os.getcwd()

input_dirs = {
    "total":        os.path.join(base_dir, "psi4_complex_defects_total"),
    "longitudinal": os.path.join(base_dir, "psi4_complex_defects_longitudinal"),
    "transverse":   os.path.join(base_dir, "psi4_complex_defects_transverse"),
}
output_dirs = {
    "total":        os.path.join(base_dir, "psi4_defects_total"),
    "longitudinal": os.path.join(base_dir, "psi4_defects_longitudinal"),
    "transverse":   os.path.join(base_dir, "psi4_defects_transverse"),
}
for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

# Components to process
components_to_process = []
if compute_total:        components_to_process.append("total")
if compute_transverse:   components_to_process.append("transverse")
if compute_longitudinal: components_to_process.append("longitudinal")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for conf_index in range(start_index, end_index + 1):
    print(f"Processing configuration {conf_index}...")

    for component in components_to_process:
        print(f"  Component: {component}")

        input_file = os.path.join(
            input_dirs[component],
            f"psi4_complex_defects_{component}_frac_{frac_str}_conf_{conf_index:03d}.pickle",
        )

        with open(input_file, "rb") as f:
            psi4_complex_list = pickle.load(f)  # list of (n_defects, 4) arrays

        if len(psi4_complex_list) < num_modes_to_process:
            raise ValueError(
                f"psi4 file for conf {conf_index}, component {component} has "
                f"{len(psi4_complex_list)} modes, but {num_modes_to_process} requested."
            )

        avg_psi4_list = []
        for mode_data in psi4_complex_list[:num_modes_to_process]:
            # Trivial mode: stored as [[nan, nan, nan, nan]]
            if np.all(np.isnan(mode_data)):
                avg_psi4_list.append(0.0)
                continue

            re_psi4 = mode_data[:, 2]
            im_psi4 = mode_data[:, 3]
            psi4    = re_psi4 + 1j * im_psi4
            avg_psi4_list.append(float(np.abs(np.mean(psi4))))

        avg_psi4_array = np.array(avg_psi4_list)

        output_file = os.path.join(
            output_dirs[component],
            f"avg_psi4_defects_{component}_frac_{frac_str}_conf_{conf_index:03d}.npy",
        )
        np.save(output_file, avg_psi4_array)
        print(f"    Saved to {output_file}")

print("All configurations processed.")

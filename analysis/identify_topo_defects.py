"""
Automated Identification and Indexing of Topological Defects

This script processes simulation data to identify topological defects in the total, longitudinal, and
transverse components of eigenvectors from LAMMPS simulations.

Boolean flags control which components are analyzed:
- analyze_total: Analyzes the complete eigenvectors.
- analyze_longitudinal: Analyzes the longitudinal components.
- analyze_transverse: Analyzes the transverse components.

The script uses two distinct fractions:
- fraction_eigen: The fraction of eigenmodes loaded from the file (e.g., 0.5).
- fraction_td: The fraction of the total modes analyzed for defects (e.g., 0.1).
"""

# Standard library imports
import argparse
from typing import List, Tuple
import os
import pickle

# Third-party imports
import numpy as np
from scipy.ndimage import gaussian_filter

# --- Parse Command Line Arguments ---
parser = argparse.ArgumentParser(description="Identify topological defects.")
parser.add_argument("np", type=int, help="Number of particles")
parser.add_argument("start", type=int, help="Configuration start index")
parser.add_argument("end", type=int, help="Configuration end index")
args = parser.parse_args()

# --- Simulation and Analysis Parameters ---

# Simulation configuration range
configuration_start = args.start
configuration_end = args.end

# Physical parameters
number_of_particles = args.np
area_density = 0.86

# Fraction for eigensystem analysis (determines which eigenvector file to load)
fraction_eigen = 0.05

# Fraction for topological defect (TD) identification (determines how many modes to analyze)
fraction_td = 0.05

# --- MODIFIED: Boolean flags to control which components are analyzed ---
analyze_total = True
analyze_longitudinal = True
analyze_transverse = True
# --------------------------------------------------------------------

# Gaussian filter and pixel construction parameters
gaussian_standard_deviation = 1.6
pixel_multiplier = 2

# --- Helper Functions (unchanged from previous version) ---


def read_Lammps_data(num_atoms: int, path_to_data: str, return_types=False):
    """Reads LAMMPS data file."""
    INDEX, TYPE, X_COORD, Y_COORD = 0, 1, 2, 3
    coordinates = np.zeros((num_atoms, 2))
    particle_types = np.zeros(num_atoms, dtype=int) if return_types else None
    with open(path_to_data, "r") as file:
        for line in file:
            if line.startswith("Atoms"):
                next(file)
                break
        for atom_count in range(num_atoms):
            line = next(file)
            data = line.split()
            index = int(data[INDEX]) - 1
            coordinates[index] = [float(data[X_COORD]), float(data[Y_COORD])]
            if return_types:
                particle_types[index] = int(data[TYPE])
    return (coordinates, particle_types) if return_types else coordinates


def compute_orientational_field(
    coordinates_of_atoms: np.ndarray,
    displacement_vector: np.ndarray,
    num_atoms: int,
    system_size: float,
    pixel_multiplier: int,
    standard_deviation: float,
    truncate: float = 4.0,
) -> np.ndarray:
    """Computes the orientational field from displacement vectors."""
    displacement_vector = displacement_vector.reshape((num_atoms, 2))
    approx_linear_density = np.ceil(np.sqrt(num_atoms))
    grid_size = pixel_multiplier * int(approx_linear_density)
    pixel_size = system_size / grid_size
    normalized_coords = coordinates_of_atoms / pixel_size
    indices_lower = np.floor(normalized_coords).astype(int)
    frac = normalized_coords - indices_lower
    f_x, f_y = frac[:, 0], frac[:, 1]
    w00, w01, w10, w11 = (
        (1 - f_x) * (1 - f_y),
        (1 - f_x) * f_y,
        f_x * (1 - f_y),
        f_x * f_y,
    )
    i0, j0 = indices_lower[:, 0] % grid_size, indices_lower[:, 1] % grid_size
    i1, j1 = (i0 + 1) % grid_size, (j0 + 1) % grid_size
    field_x, field_y = np.zeros((grid_size, grid_size)), np.zeros(
        (grid_size, grid_size)
    )
    for i, j, w, d_vec in [
        (i0, j0, w00, displacement_vector),
        (i0, j1, w01, displacement_vector),
        (i1, j0, w10, displacement_vector),
        (i1, j1, w11, displacement_vector),
    ]:
        np.add.at(field_x, (i, j), d_vec[:, 0] * w)
        np.add.at(field_y, (i, j), d_vec[:, 1] * w)
    sigma_pixels = standard_deviation / pixel_size
    smoothed_x = gaussian_filter(
        field_x, sigma=sigma_pixels, mode="wrap", truncate=truncate
    )
    smoothed_y = gaussian_filter(
        field_y, sigma=sigma_pixels, mode="wrap", truncate=truncate
    )
    return np.arctan2(smoothed_y, smoothed_x)


def phase_difference_modulo(phase_diff: np.ndarray) -> np.ndarray:
    """Adjusts phase differences to the interval (-π, π]."""
    return phase_diff + np.where(
        phase_diff > np.pi, -2 * np.pi, np.where(phase_diff < -np.pi, 2 * np.pi, 0)
    )


def compute_topological_charges(orientational_field: np.ndarray) -> np.ndarray:
    """Calculates topological charges from the orientational field."""
    rolls = [
        np.roll(orientational_field, 1, axis=1),
        np.roll(np.roll(orientational_field, 1, axis=1), -1, axis=0),
        np.roll(orientational_field, -1, axis=0),
        orientational_field,
    ]
    phase_diffs = [
        phase_difference_modulo(rolls[1] - rolls[0]),
        phase_difference_modulo(rolls[2] - rolls[1]),
        phase_difference_modulo(rolls[3] - rolls[2]),
        phase_difference_modulo(rolls[0] - rolls[3]),
    ]
    return np.rint(np.sum(phase_diffs, axis=0) / (2.0 * np.pi)).astype(int)


def identify_topological_defects(
    orientational_field: np.ndarray,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Identifies positive and negative topological defects."""
    charges = compute_topological_charges(orientational_field)
    return [list(x) for x in zip(*np.where(charges == 1))], [
        list(x) for x in zip(*np.where(charges == -1))
    ]


def identify_defects(
    coordinates: np.ndarray,
    displacement: np.ndarray,
    num_atoms: int,
    density: float,
    std_dev: float,
    px_multiplier: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Orchestrates defect identification for a given displacement vector."""
    system_size = np.sqrt(num_atoms / density)
    orientational_field = compute_orientational_field(
        coordinates_of_atoms=coordinates,
        displacement_vector=displacement,
        num_atoms=num_atoms,
        system_size=system_size,
        pixel_multiplier=px_multiplier,
        standard_deviation=std_dev,
    )
    return identify_topological_defects(orientational_field)


def save_defects(defect_data: list, path: str):
    """Saves defect data to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(defect_data, f)


def main():
    """Main execution function."""
    if not (analyze_total or analyze_longitudinal or analyze_transverse):
        print("All analysis flags are set to False. Exiting.")
        return

    # Determine number of modes to load and analyze
    num_modes_to_load = (
        int(2 * number_of_particles * fraction_eigen)
        if fraction_eigen
        else 2 * number_of_particles
    )
    num_modes_to_analyze = (
        int(2 * number_of_particles * fraction_td) if fraction_td else num_modes_to_load
    )

    if num_modes_to_analyze > num_modes_to_load:
        raise ValueError(
            f"Analysis fraction ({fraction_td}) cannot exceed eigensystem fraction ({fraction_eigen})."
        )

    # Setup directories
    base_dir = os.getcwd()
    dirs = {
        "data": os.path.join(base_dir, "data"),
        "eigenvectors": os.path.join(
            base_dir, "eigensystems", "numerical", "eigenvectors"
        ),
        "longitudinal": os.path.join(base_dir, "longitudinal_components"),
    }
    if analyze_total:
        dirs["td_total"] = os.path.join(base_dir, "topological_defects_total")
        os.makedirs(dirs["td_total"], exist_ok=True)
    if analyze_longitudinal:
        dirs["td_long"] = os.path.join(base_dir, "topological_defects_longitudinal")
        os.makedirs(dirs["td_long"], exist_ok=True)
    if analyze_transverse:
        dirs["td_trans"] = os.path.join(base_dir, "topological_defects_transverse")
        os.makedirs(dirs["td_trans"], exist_ok=True)

    # --- Main Processing Loop ---
    for conf_idx in range(configuration_start, configuration_end + 1):
        print(f"\n--- Processing configuration: {conf_idx} ---")

        # Define file names based on fractions
        fraction_str_eigen = f"fraction_{fraction_eigen:.2f}"
        fraction_str_td = f"fraction_{fraction_td:.2f}"

        eigen_suffix = f"{fraction_str_eigen}_conf_{conf_idx:03}.npy"
        td_save_suffix = f"{fraction_str_td}_conf_{conf_idx:03}.pickle"

        # Load common data
        coords = read_Lammps_data(
            number_of_particles,
            os.path.join(dirs["data"], f"lammps_data_file_{conf_idx:03d}.dat"),
        )
        eigenvectors = np.load(
            os.path.join(dirs["eigenvectors"], f"eigenvectors_{eigen_suffix}")
        )

        # --- Analyze Total Component ---
        if analyze_total:
            print("Analyzing total components...")
            pos_defects, neg_defects = [[] for _ in range(num_modes_to_analyze)], [
                [] for _ in range(num_modes_to_analyze)
            ]
            for i in range(num_modes_to_analyze):
                pos_defects[i], neg_defects[i] = identify_defects(
                    coords,
                    eigenvectors[:, i],
                    number_of_particles,
                    area_density,
                    gaussian_standard_deviation,
                    pixel_multiplier,
                )
            save_defects(
                pos_defects,
                os.path.join(dirs["td_total"], f"pos_topo_defects_{td_save_suffix}"),
            )
            save_defects(
                neg_defects,
                os.path.join(dirs["td_total"], f"neg_topo_defects_{td_save_suffix}"),
            )

        # --- Analyze Longitudinal and Transverse Components ---
        if analyze_longitudinal or analyze_transverse:
            long_comps = np.load(
                os.path.join(dirs["longitudinal"], f"longitudinal_{eigen_suffix}")
            )

            if analyze_longitudinal:
                print("Analyzing longitudinal components...")
                pos_defects_l, neg_defects_l = [
                    [] for _ in range(num_modes_to_analyze)
                ], [[] for _ in range(num_modes_to_analyze)]
                for i in range(num_modes_to_analyze):
                    pos_defects_l[i], neg_defects_l[i] = identify_defects(
                        coords,
                        long_comps[:, i],
                        number_of_particles,
                        area_density,
                        gaussian_standard_deviation,
                        pixel_multiplier,
                    )
                save_defects(
                    pos_defects_l,
                    os.path.join(
                        dirs["td_long"], f"pos_topo_defects_long_{td_save_suffix}"
                    ),
                )
                save_defects(
                    neg_defects_l,
                    os.path.join(
                        dirs["td_long"], f"neg_topo_defects_long_{td_save_suffix}"
                    ),
                )

            if analyze_transverse:
                print("Analyzing transverse components...")
                trans_comps = eigenvectors - long_comps
                pos_defects_t, neg_defects_t = [
                    [] for _ in range(num_modes_to_analyze)
                ], [[] for _ in range(num_modes_to_analyze)]
                for i in range(num_modes_to_analyze):
                    pos_defects_t[i], neg_defects_t[i] = identify_defects(
                        coords,
                        trans_comps[:, i],
                        number_of_particles,
                        area_density,
                        gaussian_standard_deviation,
                        pixel_multiplier,
                    )
                save_defects(
                    pos_defects_t,
                    os.path.join(
                        dirs["td_trans"], f"pos_topo_defects_trans_{td_save_suffix}"
                    ),
                )
                save_defects(
                    neg_defects_t,
                    os.path.join(
                        dirs["td_trans"], f"neg_topo_defects_trans_{td_save_suffix}"
                    ),
                )


if __name__ == "__main__":
    main()

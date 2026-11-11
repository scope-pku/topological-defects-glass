import argparse
import os
import numpy as np

# --- Command Line Arguments ---
parser = argparse.ArgumentParser(description="Compute Longitudinal VDOS Weights")
parser.add_argument("num_particles", type=int, help="Number of particles in the system")
parser.add_argument("start_index", type=int, help="Starting configuration index")
parser.add_argument("end_index", type=int, help="Ending configuration index")
args = parser.parse_args()

conf_idx_start = args.start_index
conf_idx_end = args.end_index
fraction = 0.05  # Fraction of modes to process

# Main directory where data are stored
main_dir = os.getcwd()

# Directories
dir_to_read_eigenvectors = os.path.join(
    main_dir, "eigensystems", "numerical", "eigenvectors"
)
dir_to_longitudinal_components = os.path.join(main_dir, "longitudinal_components")

# Subfolder for saving longitudinal VDOS weights
vdos_weights_dir = os.path.join(dir_to_longitudinal_components, "vdos_weights")
os.makedirs(vdos_weights_dir, exist_ok=True)

# Loop over configurations
for conf_index in range(conf_idx_start, conf_idx_end + 1):
    print(f"Processing configuration index {conf_index}...")

    # Load eigenvectors
    eigenvectors_path = os.path.join(
        dir_to_read_eigenvectors,
        f"eigenvectors_fraction_{fraction:.2f}_conf_{conf_index:03}.npy",
    )
    eigenvectors = np.load(eigenvectors_path)  # Shape: (n_particles * 2, num_modes)

    # Load longitudinal components
    longitudinal_path = os.path.join(
        dir_to_longitudinal_components,
        f"longitudinal_fraction_{fraction:.2f}_conf_{conf_index:03}.npy",
    )
    longitudinal_components = np.load(
        longitudinal_path
    )  # Shape: (n_particles * 2, num_modes)

    # Calculate longitudinal VDOS weights using vectorized operations
    norm_eigenvectors = np.linalg.norm(
        eigenvectors, axis=0
    )  # Norms of all eigenvector modes
    norm_longitudinal = np.linalg.norm(
        longitudinal_components, axis=0
    )  # Norms of all longitudinal modes

    # Compute weights, handling division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        longitudinal_vdos_weights = (norm_longitudinal / norm_eigenvectors) ** 2
        longitudinal_vdos_weights = np.where(
            norm_eigenvectors != 0, longitudinal_vdos_weights, 0.0
        )

    # Save longitudinal VDOS weights
    save_path = os.path.join(
        vdos_weights_dir,
        f"longitudinal_vdos_weights_frac_{fraction:.2f}_conf_{conf_index:03}.npy",
    )
    np.save(save_path, longitudinal_vdos_weights)
    print(
        f"Saved longitudinal VDOS weights for configuration {conf_index} to {save_path}"
    )

print("All configurations processed successfully.")

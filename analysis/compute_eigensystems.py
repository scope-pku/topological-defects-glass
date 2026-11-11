import os
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

# ---------------------------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute eigensystems for LAMMPS dynamical-matrix files."
    )
    parser.add_argument(
        "num_particles", type=int, help="Number of particles in the system."
    )
    parser.add_argument(
        "start_index", type=int, help="First configuration index to process."
    )
    parser.add_argument(
        "end_index", type=int, help="Last configuration index to process."
    )
    parser.add_argument(
        "--fraction", type=float, default=0.05,
        help="Fraction of eigenmodes to save (default: 0.05)."
    )
    parser.add_argument(
        "--base-dir", type=str, default=None,
        help="Base directory (default: current working directory)."
    )
    return parser.parse_args()


def create_directory_if_not_exists(directory):
    """Create a directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def read_dynmat(num_particles: int, path_to_dynmat: str):
    """
    Read the dynamical matrix from a LAMMPS dump file efficiently.
    Returns a sparse Hessian matrix for 2D motion.

    Args:
        num_particles (int): Number of particles in the system.
        path_to_dynmat (str): Path to the dynamical matrix file.

    Returns:
        scipy.sparse.csc_matrix: Sparse Hessian matrix in 2D.
    """
    # Use pandas for MUCH faster reading compared to np.loadtxt
    df = pd.read_csv(path_to_dynmat, sep=r'\s+', header=None, dtype=float, engine='c')
    data = df.values
    
    row_idx = data[:, 0].astype(int)  # Row indices
    col_idx = data[:, 1].astype(int)  # Column indices
    values = data[:, 2]  # Matrix values

    # Filter out z-direction indices (assuming 3D indices where idx % 3 == 2 is z)
    mask = (row_idx % 3 != 2) & (col_idx % 3 != 2)
    row_idx = row_idx[mask]
    col_idx = col_idx[mask]
    values = values[mask]

    # Convert 3D indices to 2D system (x, y only)
    row_idx_2d = (row_idx // 3) * 2 + (row_idx % 3)
    col_idx_2d = (col_idx // 3) * 2 + (col_idx % 3)

    # Create sparse matrix in COO format
    sparse_matrix = coo_matrix(
        (values, (row_idx_2d, col_idx_2d)),
        shape=(2 * num_particles, 2 * num_particles),
    )

    # Return directly as a CSC sparse matrix. DO NOT use .toarray() to prevent OOM errors!
    return sparse_matrix.tocsc()


def compute_eigensystems_for_dir(base_dir, num_particles, start_index, end_index, fraction):
    """
    Compute eigenvalues and eigenvectors for dynamical matrices stored as .dynmat files.
    Saves only the first fraction of the eigensystem with updated file names.

    Args:
        base_dir (str): Base directory.
        num_particles (int): Number of particles in the system.
        start_index (int): Starting configuration index.
        end_index (int): Ending configuration index.
        fraction (float): Fraction of the eigensystem to save (e.g., 0.1 for 10%).
    """
    # Define input and output directories
    input_dir = os.path.join(base_dir, "dynmat")
    eigenvalues_dir = os.path.join(base_dir, "eigensystems", "numerical", "eigenvalues")
    eigenvectors_dir = os.path.join(
        base_dir, "eigensystems", "numerical", "eigenvectors"
    )

    # Verify paths
    print(f"Input Directory: {input_dir}")
    print(f"Eigenvalues Output Directory: {eigenvalues_dir}")
    print(f"Eigenvectors Output Directory: {eigenvectors_dir}")

    if not os.path.isdir(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    create_directory_if_not_exists(eigenvalues_dir)
    create_directory_if_not_exists(eigenvectors_dir)

    # Process each configuration
    for conf_index in range(start_index, end_index + 1):
        dynmat_file = f"dynmat_{conf_index:03}.dynmat"
        dynmat_path = os.path.join(input_dir, dynmat_file)

        if not os.path.isfile(dynmat_path):
            print(f"Dynmat file not found: {dynmat_path}")
            continue

        try:
            # Read and compute - receiving a purely sparse matrix now
            hessian_sparse = read_dynmat(num_particles, dynmat_path)
            
            if hessian_sparse.shape[0] != 2 * num_particles:
                print(
                    f"Invalid matrix shape for config {conf_index}: {hessian_sparse.shape}"
                )
                continue

            # Determine how many modes to keep
            num_modes = int(fraction * 2 * num_particles)
            if num_modes == 0:
                print(f"Fraction {fraction} too small for config {conf_index}.")
                continue

            # Shift-Invert mode: efficiently find the lowest-frequency modes.
            # sigma=-1e-5 shifts slightly below 0 to avoid singularity from
            # the two translational zero modes present in glass systems.
            # which='LM' returns eigenvalues nearest to sigma (i.e. lowest freq).
            eigenvalues, eigenvectors = eigsh(
                hessian_sparse,
                k=num_modes,
                sigma=-1e-5,
                which='LM',
                tol=1e-8,
            )

            # eigsh does not guarantee ascending order — sort explicitly
            sort_idx = np.argsort(eigenvalues)
            eigenvalues_subset = eigenvalues[sort_idx]
            eigenvectors_subset = eigenvectors[:, sort_idx]

            # Updated file names with fraction
            eigenvalues_path = os.path.join(
                eigenvalues_dir,
                f"eigenvalues_fraction_{fraction:.2f}_conf_{conf_index:03}.npy",
            )
            eigenvectors_path = os.path.join(
                eigenvectors_dir,
                f"eigenvectors_fraction_{fraction:.2f}_conf_{conf_index:03}.npy",
            )
            np.save(eigenvalues_path, eigenvalues_subset)
            np.save(eigenvectors_path, eigenvectors_subset)

            print(f"Configuration {conf_index:03} processed and subset saved.")

        except Exception as e:
            print(f"Error processing configuration {conf_index}: {e}")
            continue


def main():
    """Main function to compute eigensystems for dynmat files."""
    args = parse_args()
    base_dir = args.base_dir if args.base_dir else os.getcwd()
    print(f"Base Directory: {base_dir}")
    compute_eigensystems_for_dir(
        base_dir=base_dir,
        num_particles=args.num_particles,
        start_index=args.start_index,
        end_index=args.end_index,
        fraction=args.fraction,
    )
    print("All configurations processed successfully.")


if __name__ == "__main__":
    main()
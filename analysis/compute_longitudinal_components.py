import os
import argparse
import numpy as np
import freud
from scipy.spatial import Voronoi, ConvexHull
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.sparse import coo_matrix

# ==============================================================================
# Script: compute_longitudinal_components.py
# Description: This script computes the longitudinal components of the eigenvectors
#              for a 2D particle system using Voronoi tessellation and sparse 
#              matrix operations for high performance and low memory consumption.
# ==============================================================================

# --- Command Line Arguments ---
parser = argparse.ArgumentParser(description="Compute Longitudinal Components")
parser.add_argument("num_particles", type=int, help="Number of particles in the system")
parser.add_argument("start_index", type=int, help="Starting configuration index")
parser.add_argument("end_index", type=int, help="Ending configuration index")
args = parser.parse_args()

# --- Configuration Parameters ---
num_particles = args.num_particles
conf_idx_start = args.start_index
conf_idx_end = args.end_index

fraction = 0.05  # Fraction of modes to process

# Physical parameters
displacement_magnitude = 1e-6
buffer = 0.1
area_density = 0.86
system_size = np.sqrt(num_particles / area_density)

# --- Helper Functions (From Original Validated Script) ---


def read_data(num_particles, system_size, path_to_data):
    """
    Read LAMMPS data file and return coordinates centered at the origin.

    Args:
        num_particles (int): Number of particles in the system.
        system_size (float): Size of the simulation box.
        path_to_data (str): Path to the LAMMPS data file.

    Returns:
        np.ndarray: Array of particle coordinates.
    """
    coordinates = np.zeros((num_particles, 3))
    with open(path_to_data, "r") as file:
        while True:
            text_line = file.readline()
            data = text_line.split()
            if not data:
                continue
            elif data[0] == "Atoms":
                file.readline()
                break
        for i in range(num_particles):
            text_line = file.readline()
            data = text_line.split()
            index = int(data[0]) - 1
            coordinates[index, 0] = float(data[2])
            coordinates[index, 1] = float(data[3])
        coordinates[:, :-1] -= system_size / 2
    return coordinates


def compute_voronoi_volume_matrix(
    num_particles, system_size, buffer, volumes, neighbor_list, coordinates, box
):
    """
    Compute the sparse Voronoi volume derivative matrix (M) for the system.
    
    This function calculates the derivative of the Voronoi volume of each particle
    with respect to the displacement of its neighbors. To avoid huge memory consumption,
    it constructs the matrix directly in a sparse Coordinate (COO) format by only 
    recording the non-zero derivatives before converting it to Compressed Sparse Row (CSR) format.
    
    Args:
        num_particles (int): Number of particles.
        system_size (float): Size of the simulation box.
        buffer (float): Buffer size for periodic boundary conditions.
        volumes (np.ndarray): Array of initial Voronoi volumes.
        neighbor_list (list): List of neighbors for each particle.
        coordinates (np.ndarray): 2D array of particle positions.
        box (freud.box.Box): Simulation box object.

    Returns:
        scipy.sparse.csr_matrix: Sparse Voronoi volume matrix of shape (N, 2N).
    """
    print("  - Constructing sparse Voronoi matrix...")
    # Three lists to construct the sparse matrix
    row_idx = []
    col_idx = []
    vals = []

    for j in range(num_particles):
        if j % 10000 == 0:
            print(f"    Processing particle indexed {j}...")

        nearest_neighbors = neighbor_list[j]
        len_nearest_neighbors = len(nearest_neighbors)

        neighbors = []
        for nearestNeighbor in neighbor_list[j]:
            neighbors.extend(neighbor_list[nearestNeighbor])
        neighbors = list(dict.fromkeys(neighbors))

        for alpha in range(2):
            for sign in (+1, -1):
                finite_displacement = sign * displacement_magnitude * system_size
                coordinates_displaced = coordinates.copy()
                coordinates_displaced[j, alpha] += finite_displacement
                coordinates_displaced = coordinates_displaced[neighbors]

                pbuff = freud.locality.PeriodicBuffer()
                pbuff.compute(
                    system=(
                        box,
                        np.c_[
                            coordinates_displaced,
                            np.zeros(coordinates_displaced.shape[0]),
                        ],
                    ),
                    buffer=buffer * system_size,
                    images=False,
                    include_input_points=False,
                )
                bufferPoints = pbuff.buffer_points
                if bufferPoints.shape != (0,):
                    bufferPoints = np.delete(bufferPoints, 2, axis=1)
                    volumes_new = voronoi_volumes(
                        np.vstack((coordinates_displaced, bufferPoints))
                    )
                else:
                    volumes_new = voronoi_volumes(coordinates_displaced)
                
                # Calculate derivatives and record them into the sparse matrix lists
                derivative = (volumes_new[:len_nearest_neighbors] - volumes[nearest_neighbors]) / (volumes[nearest_neighbors] * finite_displacement)
                
                # x displacement of j corresponds to column 2j, y displacement to 2j+1
                col = 2 * j + alpha
                for idx, nn in enumerate(nearest_neighbors):
                    row_idx.append(nn)
                    col_idx.append(col)
                    # Note: Original code had /= 2 at the end calculated from both positive and negative directions. We divide by 2 here.
                    vals.append(derivative[idx] / 2.0)

    # Construct and return the sparse matrix in CSR format for fast multiplication and solving later
    sparse_M = coo_matrix(
        (vals, (row_idx, col_idx)), 
        shape=(num_particles, 2 * num_particles)
    )
    return sparse_M.tocsr()


def voronoi_volumes(points):
    """
    Compute the Voronoi volumes for a set of points.

    Args:
        points (np.ndarray): Array of points.

    Returns:
        np.ndarray: Array of Voronoi volumes.
    """
    vor = Voronoi(points)
    vol = np.zeros(vor.npoints)
    for i, reg_num in enumerate(vor.point_region):
        indices = vor.regions[reg_num]
        if -1 in indices:
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(vor.vertices[indices]).volume
    return vol


# --- Main Execution ---


def main():
    # Directory setup
    base_dir = os.getcwd()
    dir_to_data = os.path.join(base_dir, "data")
    dir_to_eigenvectors = os.path.join(
        base_dir, "eigensystems", "numerical", "eigenvectors"
    )
    dir_to_longitudinal = os.path.join(base_dir, "longitudinal_components")

    os.makedirs(dir_to_longitudinal, exist_ok=True)

    for conf_idx in range(conf_idx_start, conf_idx_end + 1):
        print(f"\n=== Processing Configuration {conf_idx} ===")

        # 1. Check Input Files
        data_filename = f"lammps_data_file_{conf_idx:03}.dat"
        path_to_data = os.path.join(dir_to_data, data_filename)

        eigen_filename = f"eigenvectors_fraction_{fraction:.2f}_conf_{conf_idx:03}.npy"
        path_to_eigen = os.path.join(dir_to_eigenvectors, eigen_filename)

        if not os.path.exists(path_to_data):
            print(f"Error: Data file not found at {path_to_data}. Skipping.")
            continue
        if not os.path.exists(path_to_eigen):
            print(f"Error: Eigenvector file not found at {path_to_eigen}. Skipping.")
            continue

        # 2. Setup System
        box = freud.box.Box.square(system_size)
        coordinates = read_data(num_particles, system_size, path_to_data)

        # 3. Initial Voronoi Computation
        print("  - Computing initial Voronoi tessellation...")
        voro = freud.locality.Voronoi()
        voro.compute((box, coordinates))

        # Neighbor list construction
        nlist = voro.nlist
        neighbor_list = [[] for _ in range(num_particles)]
        for i, j in nlist[:]:
            neighbor_list[i].append(j)
        for i in range(num_particles):
            neighbor_list[i].insert(0, i)

        # Buffer points
        pbuff = freud.locality.PeriodicBuffer()
        pbuff.compute(
            system=(box, coordinates),
            buffer=buffer * system_size,
            images=False,
            include_input_points=False,
        )
        buffer_points = pbuff.buffer_points

        # 2D cleaning
        coords_2d = np.delete(coordinates, 2, axis=1)
        buffer_2d = np.delete(buffer_points, 2, axis=1)
        volumes = voronoi_volumes(np.vstack((coords_2d, buffer_2d)))[:num_particles]

        # 4. Compute Sparse Voronoi Matrix M
        M = compute_voronoi_volume_matrix(
            num_particles, system_size, buffer, volumes, neighbor_list, coords_2d, box
        )

        # 5. Construct Gram Matrix and factorize it
        print("  - Constructing and factorizing Gram matrix A = M M^T ...")
        # A is a highly sparse N x N matrix
        A = M.dot(M.T) 
        
        # Physical constraint: total volume conservation causes A to be singular.
        # Solution: add a tiny perturbation epsilon * I, mathematically equivalent to pseudo-inverse.
        epsilon = 1e-8
        A_reg = A + sp.eye(num_particles, format='csc') * epsilon
        
        # Perform Sparse LU factorization using SuperLU (only needs to be done once!)
        lu_solver = splu(A_reg.tocsc())

        # 6. Load Eigenvectors and Project (With Memory Chunking)
        print("  - Loading eigenvectors...")
        eigenvectors = np.load(path_to_eigen)  # Shape (2N, Num_Modes)
        num_modes = eigenvectors.shape[1]
        
        # Prepare output matrix (takes about 25 GB of memory)
        longitudinal_components = np.zeros_like(eigenvectors)

        print("  - Calculating longitudinal components (in chunks to save memory)...")
        chunk_size = 1000 # Process 1000 modes at a time to prevent abnormal memory usage
        
        for start_col in range(0, num_modes, chunk_size):
            end_col = min(start_col + chunk_size, num_modes)
            E_chunk = eigenvectors[:, start_col:end_col]
            
            # Mathematical logic: L = M^T @ (M @ M^T)^(-1) @ M @ E
            # Step a: V = M @ E (Shape: N x chunk_size)
            V_chunk = M.dot(E_chunk)
            
            # Step b: solve A @ Y = V (Shape: N x chunk_size)
            Y_chunk = lu_solver.solve(V_chunk)
            
            # Step c: L = M^T @ Y (Shape: 2N x chunk_size)
            L_chunk = M.T.dot(Y_chunk)
            
            longitudinal_components[:, start_col:end_col] = L_chunk
            print(f"    Processed modes {start_col} to {end_col} / {num_modes}")

        # Free memory
        del M, A, A_reg, lu_solver, eigenvectors

        # 7. Save Result
        longitudinal_filename = f"longitudinal_fraction_{fraction:.2f}_conf_{conf_idx:03}.npy"
        path_to_output = os.path.join(dir_to_longitudinal, longitudinal_filename)

        # Highly recommended to save as float32 to save disk space
        np.save(path_to_output, longitudinal_components.astype(np.float32))
        print(f"  - Saved: {path_to_output}")


if __name__ == "__main__":
    main()

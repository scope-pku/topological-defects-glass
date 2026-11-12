# Topological defects in vibrational modes of 2D glasses

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24150526.svg)](https://doi.org/10.5281/zenodo.24150526)

Example code accompanying the manuscript

> *Bipartite lattice transition of topological defects at the Ioffe–Regel limit in amorphous solids*.

This repository contains the key analysis scripts used to produce the data in the paper, from the dynamical matrix of a quenched 2D glass to the topological defects of its vibrational modes and their bond-orientational order. Glass configurations were generated with LAMMPS (2D binary inverse-power-law glass, N = 32 000 and 128 000).

## Workflow

| Step | Script | What it does |
|---|---|---|
| 1 | `analysis/compute_eigensystems.py` | Diagonalises the dynamical (Hessian) matrix written by LAMMPS; outputs eigenfrequencies and eigenvectors. |
| 2 | `analysis/compute_longitudinal_components.py` | Helmholtz decomposition of each eigenvector into longitudinal (curl-free) and transverse (divergence-free) parts. |
| 3 | `analysis/compute_longitudinal_vdos_weights.py` | Longitudinal weight of each mode, used to split g(ω) into transverse and longitudinal parts. |
| 4 | `analysis/identify_topo_defects.py` | Locates topological defects (winding number Q = ±1) of the total, transverse and longitudinal displacement fields. |
| 5 | `analysis/compute_psi4_complex_defects.py` | Fourfold bond-orientational order parameter ψ4 of the defect pattern in each mode. |
| 6 | `analysis/compute_avg_psi4_defects.py` | Mode-resolved ⟨\|ψ4\|⟩ used in Fig. 1a. |
| 7 | `figures/fig1_compact_data.py`, `figures/plot_fig1_compact.py` | Reproduce Fig. 1 from the outputs of steps 1–6. |

## Requirements

Python ≥ 3.9 with `numpy`, `scipy`, `pandas`, `matplotlib`, `freud-analysis`; a LaTeX installation for figure text.

## Notes

Directory paths at the top of each script point to the authors' data layout and need to be adapted. The raw configurations and eigenvectors (several GB) are available from the corresponding author upon reasonable request.

## Contact

- Limei Xu, International Center for Quantum Materials, School of Physics, Peking University (limei.xu@pku.edu.cn)
- Di Zhou, Department of Theoretical Physics, School of Physics, Beijing Institute of Technology (dizhou@bit.edu.cn)

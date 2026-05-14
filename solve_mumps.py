#!/usr/bin/env python3

"""
Example script that:

1. Reads a sparse matrix in Matrix Market format (.mtx)
   from the SuiteSparse Matrix Collection.

2. Solves the linear system Ax = b using pymumps.

Requirements
------------
pip install numpy scipy pymumps

Ubuntu/Debian system packages:
sudo apt install libmumps-dev libopenmpi-dev openmpi-bin gfortran

Usage
-----
python solve_mumps.py matrix.mtx

Optional:
python solve_mumps.py matrix.mtx rhs.npy

Notes
-----
- If no RHS vector is provided, b = ones(n)
- Matrix must be square
"""

import sys
import numpy as np
from scipy.io import mmread
from scipy.sparse import csc_matrix
from mumps import DMumpsContext


def load_matrix(matrix_path):
    """
    Load Matrix Market matrix and convert to CSC format.
    """

    print(f"Reading matrix: {matrix_path}")

    A = mmread(matrix_path)

    # Convert to sparse CSC if needed
    if not hasattr(A, "tocsc"):
        A = csc_matrix(A)
    else:
        A = A.tocsc()

    print(f"Matrix shape : {A.shape}")
    print(f"Nonzeros     : {A.nnz}")
    print(f"Data type    : {A.dtype}")

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    return A


def generate_rhs(n, seed=42):
    """
    Generate random RHS vector.
    """

    print("Generating random RHS vector...")

    rng = np.random.default_rng(seed)

    # Random vector in [0, 1)
    b = rng.random(n, dtype=np.float64)

    return b

def solve_with_mumps(A, b):
    """
    Solve Ax = b using MUMPS.
    """

    print("Initializing MUMPS...")

    ctx = DMumpsContext()
    ctx.set_silent()

    try:
        # Centralized sparse matrix
        ctx.set_centralized_sparse(A)

        # RHS vector
        x = b.copy()
        ctx.set_rhs(x)

        print("Running MUMPS solve...")

        # Job 6 = analysis + factorization + solve
        ctx.run(job=6)

        print("Solve completed.")

        return x

    finally:
        ctx.destroy()


def compute_residual(A, x, b):
    """
    Compute ||Ax - b||_2
    """

    r = A @ x - b
    return np.linalg.norm(r)


def main():

    matrix_path = "./matrices/b1_ss.mtx"
    rhs_path = None

    # Load matrix
    A = load_matrix(matrix_path)

    # Generate random RHS
    b = generate_rhs(A.shape[0])

    # Solve
    x = solve_with_mumps(A, b)

    # Residual
    residual = compute_residual(A, x, b)

    print()
    print("==== RESULTS ====")
    print(f"Residual ||Ax - b||_2 = {residual:e}")

    print()
    print("First 10 entries of solution:")
    print(x[:10])


if __name__ == "__main__":
    main()
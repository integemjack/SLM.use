#!/bin/bash
# Regenerate kernels/slm.ptx. Needs nvcc; on Windows run from a shell with the MSVC
# environment loaded. CUDA 12.8 still targets sm_60, so the PTX runs on Pascal and newer.
set -e
cd "$(dirname "$0")/.."
nvcc --ptx -O3 -std=c++17 -arch=compute_60 -o kernels/slm.ptx kernels/slm.cu

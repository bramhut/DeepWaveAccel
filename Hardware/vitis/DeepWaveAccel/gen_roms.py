#!/usr/bin/env python3
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# Project constants (must match your HLS project)
# ------------------------------------------------------------------
IMG_LEN   = 2234
N_ELEM    = 48
ND        = 6
MAX_ORDER = 22

PARAM_DIR = Path(__file__).parent / "parameters"
OUTDIR    = Path(__file__).parent

# ------------------------------------------------------------------
def write_header_start(f, src_name, comment):
    f.write("#pragma once\n")
    f.write(f"// Auto-generated from {src_name}.csv\n")
    f.write(f"// {comment}\n")
    f.write('#include "types.hpp"\n\n')

# ------------------------------------------------------------------
# b_vectors.csv → b_vectors_data.hpp
# Format: pixel,elem,bre,bim
# ------------------------------------------------------------------
def gen_b_vectors():
    infile  = PARAM_DIR / "b_vectors.csv"
    outfile = OUTDIR / "b_vectors_data.hpp"
    data = np.loadtxt(infile, delimiter=",", skiprows=1)
    bre, bim = data[:, 2], data[:, 3]

    with open(outfile, "w", encoding="utf-8") as f:
        write_header_start(f, "b_vectors", "Steering vectors (complex b_t)")
        f.write(f"static const b_t b_vectors_rom[{IMG_LEN}][{N_ELEM}] = {{\n")
        for i in range(IMG_LEN):
            f.write("    { ")
            row = []
            for j in range(N_ELEM):
                idx = i * N_ELEM + j
                row.append(f"b_t(b_real_t({bre[idx]}), b_real_t({bim[idx]}))")
            f.write(", ".join(row))
            f.write(" }")
            f.write(",\n" if i < IMG_LEN - 1 else "\n")
        f.write("};\n")
    print(f"✅ Generated {outfile.name}")

# ------------------------------------------------------------------
# laplacian.csv → laplacian_data.hpp
# Layout: [main] + ND*IMG_LEN values
# ------------------------------------------------------------------
def gen_laplacian():
    infile  = PARAM_DIR / "laplacian.csv"
    outfile = OUTDIR / "laplacian_data.hpp"
    data = np.loadtxt(infile, delimiter=",", skiprows=1)

    lap_main = data[0]
    lap_rest = data[1:].reshape(ND, IMG_LEN)

    with open(outfile, "w", encoding="utf-8") as f:
        write_header_start(f, "laplacian", "Main + ND off-diagonals (lap_t)")
        f.write(f"static const lap_t lap_main_rom = lap_t({lap_main});\n")
        f.write(f"static const lap_t lap_rest_rom[{ND}][{IMG_LEN}] = {{\n")
        for d in range(ND):
            f.write("    { ")
            vals = [f"lap_t({v})" for v in lap_rest[d]]
            f.write(", ".join(vals))
            f.write(" }")
            f.write(",\n" if d < ND - 1 else "\n")
        f.write("};\n")
    print(f"✅ Generated {outfile.name}")

# ------------------------------------------------------------------
# lap_offsets.csv → lap_offsets_data.hpp
# ND comma-separated offsets
# ------------------------------------------------------------------
def gen_lap_offsets():
    infile  = PARAM_DIR / "lap_offsets.csv"
    outfile = OUTDIR / "lap_offsets_data.hpp"
    with open(infile, "r", encoding="utf-8-sig") as f:
        line = f.readline().strip()
    vals = [v.strip() for v in line.split(",") if v.strip() != ""]

    with open(outfile, "w", encoding="utf-8") as f:
        write_header_start(f, "lap_offsets", "Offsets for Laplacian diagonals")
        f.write(f"static const idx_t lap_offsets_rom[{ND}] = {{")
        f.write(", ".join(f"idx_t({v})" for v in vals))
        f.write("};\n")
    print(f"✅ Generated {outfile.name}")

# ------------------------------------------------------------------
# tau.csv → tau_data.hpp
# ------------------------------------------------------------------
def gen_tau():
    infile  = PARAM_DIR / "tau.csv"
    outfile = OUTDIR / "tau_data.hpp"
    data = np.loadtxt(infile, delimiter=",", skiprows=1)

    with open(outfile, "w", encoding="utf-8") as f:
        write_header_start(f, "tau", "Per-pixel tau compensation (tau_t)")
        f.write(f"static const tau_t tau_rom[{IMG_LEN}] = {{\n")
        for i, val in enumerate(data):
            sep = "," if i < len(data) - 1 else ""
            f.write(f"    tau_t({val}){sep}\n")
        f.write("};\n")
    print(f"✅ Generated {outfile.name}")

# ------------------------------------------------------------------
# theta.csv → theta_data.hpp
# Single line of comma-separated floats
# ------------------------------------------------------------------
def gen_theta():
    infile  = PARAM_DIR / "theta.csv"
    outfile = OUTDIR / "theta_data.hpp"

    # Read the file safely (remove BOM if present)
    with open(infile, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()
    vals = [v.strip() for v in text.split(",") if v.strip() != ""]

    if len(vals) == 0:
        raise RuntimeError(f"{infile} appears empty or malformed")

    with open(outfile, "w", encoding="utf-8") as f:
        write_header_start(f, "theta", "Chebyshev theta coefficients (theta_t)")
        f.write(f"static const theta_t theta_rom[{MAX_ORDER+1}] = {{\n")
        for i in range(MAX_ORDER + 1):
            val = vals[i] if i < len(vals) else "0.0"
            sep = "," if i < MAX_ORDER else ""
            f.write(f"    theta_t({val}){sep}\n")
        f.write("};\n")
    print(f"✅ Generated {outfile.name}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating parameter ROM headers from ./parameters ...")
    gen_b_vectors()
    gen_laplacian()
    gen_lap_offsets()
    gen_tau()
    gen_theta()
    print("✅ All parameter headers generated successfully.")

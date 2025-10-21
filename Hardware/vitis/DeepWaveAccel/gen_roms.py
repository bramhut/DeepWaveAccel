# gen_pair_rom.py
N_ELEM = 48
pairs = []
for j in range(N_ELEM):
    for k in range(j + 1, N_ELEM):
        pairs.append((j, k))

with open("pair_rom_data.hpp", "w") as f:
    f.write("#pragma once\n")
    f.write(f"// Auto-generated: {len(pairs)} (j,k) pairs for upper triangle\n\n")
    f.write(f"static const int NPAIR = {len(pairs)};\n")
    f.write(f"static const int j_rom[{len(pairs)}] = {{")
    f.write(",".join(str(j) for j, _ in pairs))
    f.write("};\n")
    f.write(f"static const int k_rom[{len(pairs)}] = {{")
    f.write(",".join(str(k) for _, k in pairs))
    f.write("};\n")
    
import numpy as np

# --- Parameters ---
N_WIN = 200  # must match your header
outfile = "hann_window_data.hpp"

# Hann window coefficients
n = np.arange(N_WIN)
w = 0.5 * (1.0 - np.cos(2 * np.pi * n / (N_WIN - 1)))

# Fixed-point scaling for win_t = ap_ufixed<12, 0> (12 fractional bits)
scale = 2**12
w_fixed = np.round(w * scale).astype(int)

with open(outfile, "w") as f:
    f.write("#pragma once\n")
    f.write(f"// Auto-generated Hann window ({N_WIN} coefficients)\n")
    f.write(f"#include \"types.hpp\"\n\n")
    f.write(f"static const win_t hann_window[{N_WIN}] = {{\n")

    for i, val in enumerate(w_fixed):
        sep = "," if i < N_WIN - 1 else ""
        # Convert back to fixed-point literal for readability
        f.write(f"    win_t({val} / (double){scale}){sep}\n")

    f.write("};\n")

print(f"✅ Generated {outfile} with {N_WIN} fixed-point Hann coefficients.")



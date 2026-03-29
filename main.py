from typing import List, Tuple
import time
from math import floor
import numba as nb
import numpy as np
import biome as bm


MASK32 = 0xffffffff
N = 624
M = 397
MATRIX_A = 0x9908b0df
UPPER_MASK = 0x80000000
LOWER_MASK = 0x7fffffff

@nb.njit(cache=True)
def mt_init(seed32):
    mt = np.empty(N, dtype=np.uint32)
    mt[0] = seed32
    for i in range(1, N):
        mt[i] = (0x6c078965 * (mt[i-1] ^ (mt[i-1] >> 30)) + i) & MASK32
    return mt

@nb.njit(cache=True)
def mt_twist(mt):
    for i in range(N):
        y = (mt[i] & UPPER_MASK) | (mt[(i+1) % N] & LOWER_MASK)
        mt[i] = mt[(i + M) % N] ^ (y >> 1)
        if y & 1:
            mt[i] ^= MATRIX_A

@nb.njit(cache=True)
def mt_extract(mt, idx):
    if idx >= N:  # needs twist
        mt_twist(mt)
        idx = 0
    y = mt[idx]
    y ^= (y >> 11)
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= (y >> 18)
    return y & MASK32, idx+1

def getpos(world_seed, rx, rz, spacing, separation, salt, linear_separation):
    spawn_range = spacing - separation
    mixed = (world_seed + rx*341873128712 + rz*132897987541 + salt) & ((1<<64)-1)
    seed32 = mixed & 0xffffffff

    mt = mt_init(seed32)
    idx = N
    r0, idx = mt_extract(mt, idx)
    r1, idx = mt_extract(mt, idx)
    if linear_separation:
        r2, idx = mt_extract(mt, idx)
        r3, idx = mt_extract(mt, idx)
        off_x = ((r0 % spawn_range) + (r1 % spawn_range)) // 2
        off_z = ((r2 % spawn_range) + (r3 % spawn_range)) // 2
    else:
        off_x = r0 % spawn_range
        off_z = r1 % spawn_range

    chunk_x = rx * spacing + off_x
    chunk_z = rz * spacing + off_z
    return (chunk_x*16, chunk_z*16)


# -------------------------
# Seed scanning
# -------------------------
print(
"""
Minecraft Bedrock Edition — brute-force structural 48-bit seed searcher
with optional Java Edition biome filtering via cubiomes.

NOTE: Structure RNG uses Bedrock constants; biome generation uses the
      Java Edition algorithm from cubiomes (same noise math, different edition).
      Use biome results as a guide or when searching Java seeds.

RNG constants  (Format: Spacing, Separation, Salt, Linear Separation)
  Bastion/Fortress:      30,  4, 30084232,  0
  Village:               34,  8, 10387312,  1
  Pillager Outpost:      80, 24, 165745296, 1
  Woodland Mansion:      80, 20, 10387319,  1
  Ocean Monument:        32,  5, 10387313,  1
  Shipwreck:             24,  4, 165745295, 1
  Ruined Portal:         40, 15, 40552231,  0
  Other Overworld:       32,  8, 14357617,  0

Search radius:  app accepts a structure hit if its position is within this
                many blocks of the origin on both axes.
Min occurrence: how many of the 4 checked regions must have a hit (max 4).
"""
)


def seedsearch():
    seedstart  = int(input("SeedStart: "))
    seedend    = int(input("SeedEnd: "))
    spacing    = int(input("Spacing: "))
    separation = int(input("Separation: "))
    salt       = int(input("Salt: "))
    linear_sep = bool(int(input("Linear separation: (0 or 1) ")))
    radius     = int(input("Search radius: "))
    occurence  = int(input("Min occurrence: "))
    output_file = input("Output file name (default: seed_results.txt): ").strip() or "seed_results.txt"

    # --- optional biome filtering ---
    print()
    biome_cfg = bm.prompt_biome_requirements()
    biome_gen = None
    if biome_cfg is not None:
        mc_version, dim, biome_reqs = biome_cfg
        biome_gen = bm.BiomeGenerator(mc_version=mc_version, dim=dim)
        print(f"\nBiome filtering ON  (MC {[k for k,v in bm.MC_VERSIONS.items() if v==mc_version][0]}, "
              f"{[k for k,v in bm.DIMENSIONS.items() if v==dim][0]})\n")
    else:
        print("\nBiome filtering OFF\n")

    times = time.time()
    seeds = range(seedstart, seedend)

    with open(output_file, 'w') as f:
        f.write(f"# Seed search results — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Range [{seedstart}, {seedend})  spacing={spacing} separation={separation} "
                f"salt={salt} linear={int(linear_sep)}\n")
        f.write(f"# Radius={radius}  min_occurrence={occurence}\n")
        if biome_gen:
            f.write(f"# Biome filter: {biome_reqs}\n")
        f.write("\n")

        for seed in seeds:
            found = 0
            i = getpos(seed, 0,  0,  spacing, separation, salt, linear_sep)
            if -radius < i[0] < radius and -radius < i[1] < radius:
                found += 1
            if found < 1 and occurence >= 4:
                continue

            j = getpos(seed, -1,  0, spacing, separation, salt, linear_sep)
            if -radius < j[0] < radius and -radius < j[1] < radius:
                found += 1

            k = getpos(seed,  0, -1, spacing, separation, salt, linear_sep)
            if -radius < k[0] < radius and -radius < k[1] < radius:
                found += 1

            l = getpos(seed, -1, -1, spacing, separation, salt, linear_sep)
            if -radius < l[0] < radius and -radius < l[1] < radius:
                found += 1

            if found < occurence:
                continue

            # --- biome check (only for seeds that pass structure filter) ---
            biome_info = ""
            if biome_gen is not None:
                passed, biome_results = biome_gen.check_seed(seed, biome_reqs)
                if not passed:
                    continue
                parts = [f"({x},{z})={name}" for x, z, name in biome_results]
                biome_info = "  biomes: " + ", ".join(parts)

            line = f"Seed {seed}: {i} {j} {k} {l}{biome_info}"
            f.write(line + "\n")

            if seed % 1_000_000 == 0 and seed != seedstart:
                elapsed = time.time() - times
                prog = f"[Progress] scanned up to {seed}  elapsed={elapsed:.1f}s"
                print(prog)
                f.write(prog + "\n")

        elapsed = time.time() - times
        f.write(f"\n# Finished scanning.  Time: {elapsed:.2f}s\n")

    print(f"Done. Results saved to '{output_file}'.  Time: {elapsed:.2f}s")


seedsearch()

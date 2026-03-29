"""
main.py — Seed search loop and all user-facing output.
"""

import time
import biome as bm
from structure import getpos


BANNER = """
Minecraft Bedrock Edition — brute-force structural 48-bit seed searcher
with biome validation via cubiomes.

NOTE on Bastion Remnants and Nether Fortresses:
  Both share RNG salt 30084232 (they cannot be distinguished by salt alone).
  These are NETHER structures — skip biome validation when searching for them.

RNG constants  (Format: Spacing, Separation, Salt, Linear Separation)
  Bastion/Fortress:      30,  4, 30084232,  0   <- Nether; skip biome
  Village:               34,  8, 10387312,  1
  Pillager Outpost:      80, 24, 165745296, 1
  Woodland Mansion:      80, 20, 10387319,  1
  Ocean Monument:        32,  5, 10387313,  1
  Shipwreck:             24,  4, 165745295, 1
  Ruined Portal:         40, 15, 40552231,  0
  Other Overworld:       32,  8, 14357617,  0

Search radius:   accepts a hit if the position is within this many blocks of
                 the origin on both axes.
Min occurrence:  how many of the 4 checked regions must have a valid hit (max 4).
                 With biome validation ON a hit only counts if the biome at the
                 computed position is in the allowed set.
"""


def seedsearch():
    print(BANNER)

    # ---- search parameters -------------------------------------------------
    seedstart  = int(input("SeedStart: "))
    seedend    = int(input("SeedEnd: "))
    spacing    = int(input("Spacing: "))
    separation = int(input("Separation: "))
    salt       = int(input("Salt: "))
    linear_sep = bool(int(input("Linear separation: (0 or 1) ")))
    radius     = int(input("Search radius: "))
    occurence  = int(input("Min occurrence: "))

    # ---- output destination ------------------------------------------------
    print()
    out_raw = input("Output to (f)ile or (c)onsole? [f]: ").strip().lower()
    to_console = out_raw in ("c", "console")
    if to_console:
        output_file = None
        print("  Results will be printed to the console.")
    else:
        output_file = input("  File name [seed_results.txt]: ").strip() or "seed_results.txt"
        print(f"  Results will be saved to '{output_file}'.")

    # ---- biome validation --------------------------------------------------
    effective_biomes = bm.prompt_biome_validation()  # frozenset[int] | None

    biome_gen = None
    if effective_biomes is not None:
        biome_gen = bm.BiomeGenerator(mc_version=bm.MC_1_21, dim=bm.DIM_OVERWORLD)
        print("  Biome generator ready (MC 1.21, overworld).")
    print()

    # ---- header ------------------------------------------------------------
    header = (
        f"# Seed search results — {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Range [{seedstart}, {seedend})  spacing={spacing} "
        f"separation={separation} salt={salt} linear={int(linear_sep)}\n"
        f"# Radius={radius}  min_occurrence={occurence}"
    )
    if effective_biomes is not None:
        labels = ", ".join(bm.BIOME_NAMES.get(b, str(b)) for b in sorted(effective_biomes))
        header += f"\n# Biome filter: [{labels}]"

    # ---- helpers -----------------------------------------------------------
    def emit(line, f=None):
        """Write a line to the file, or print it if in console mode."""
        if to_console:
            print(line)
        else:
            f.write(line + "\n")

    # ---- main scan loop ----------------------------------------------------
    def run(f=None):
        emit(header, f)
        emit("", f)

        times = time.time()

        for seed in range(seedstart, seedend):

            # compute the 4 candidate positions
            i = getpos(seed,  0,  0, spacing, separation, salt, linear_sep)
            j = getpos(seed, -1,  0, spacing, separation, salt, linear_sep)
            k = getpos(seed,  0, -1, spacing, separation, salt, linear_sep)
            l = getpos(seed, -1, -1, spacing, separation, salt, linear_sep)

            # fast radius filter — no biome lookup yet
            i_in = -radius < i[0] < radius and -radius < i[1] < radius
            j_in = -radius < j[0] < radius and -radius < j[1] < radius
            k_in = -radius < k[0] < radius and -radius < k[1] < radius
            l_in = -radius < l[0] < radius and -radius < l[1] < radius

            if i_in + j_in + k_in + l_in < occurence:
                continue

            # biome check — only for positions that passed the radius filter
            pos_biome: dict[tuple, str] = {}

            if biome_gen is not None:
                biome_gen.apply_seed(seed)
                found = 0
                for pos, in_rad in [(i, i_in), (j, j_in), (k, k_in), (l, l_in)]:
                    if not in_rad:
                        continue
                    bid  = biome_gen.biome_at_block(pos[0], pos[1])
                    name = biome_gen.biome_name(bid)
                    pos_biome[pos] = name
                    if bid in effective_biomes:
                        found += 1
            else:
                found = i_in + j_in + k_in + l_in

            if found < occurence:
                continue

            # build and emit result line
            parts = []
            for pos, in_rad in [(i, i_in), (j, j_in), (k, k_in), (l, l_in)]:
                if in_rad and pos in pos_biome:
                    parts.append(f"{pos}[{pos_biome[pos]}]")
                else:
                    parts.append(str(pos))
            emit(f"Seed {seed}: {' '.join(parts)}", f)

            # periodic progress to stdout
            if seed % 1_000_000 == 0 and seed != seedstart:
                elapsed = time.time() - times
                prog = f"[Progress] scanned up to {seed}  elapsed={elapsed:.1f}s"
                print(prog)
                if not to_console and f:
                    f.write(prog + "\n")

        elapsed = time.time() - times
        emit(f"\n# Finished scanning.  Time: {elapsed:.2f}s", f)
        if not to_console:
            print(f"Done. Results saved to '{output_file}'.  Time: {elapsed:.2f}s")

    # ---- dispatch ----------------------------------------------------------
    if to_console:
        run()
    else:
        with open(output_file, "w") as f:
            run(f)


seedsearch()

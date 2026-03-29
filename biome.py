"""
biome.py — Python bindings for cubiomes (https://github.com/Cubitect/cubiomes)

Provides fast Minecraft Java Edition biome lookup via the compiled
cubiomes shared library.  Use BiomeGenerator to apply a seed and query
the biome at any block or chunk coordinate.
"""

import ctypes
import os

# ---------------------------------------------------------------------------
# Load the shared library
# ---------------------------------------------------------------------------
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cubiomes", "libcubiomes.so")
_lib = ctypes.CDLL(_LIB_PATH)

# Opaque buffer size — must match sizeof(Generator) from cubiomes
GENERATOR_SIZE = 27592

# ---------------------------------------------------------------------------
# MC version constants  (from cubiomes/biomes.h enum MCVersion)
# ---------------------------------------------------------------------------
MC_1_7  = 8
MC_1_8  = 9
MC_1_9  = 10
MC_1_10 = 11
MC_1_11 = 12
MC_1_12 = 13
MC_1_13 = 16
MC_1_14 = 17
MC_1_15 = 18
MC_1_16 = 20
MC_1_17 = 21
MC_1_18 = 22
MC_1_19 = 24
MC_1_20 = 25
MC_1_21 = 28

MC_VERSIONS = {
    "1.7":  MC_1_7,  "1.8":  MC_1_8,  "1.9":  MC_1_9,
    "1.10": MC_1_10, "1.11": MC_1_11, "1.12": MC_1_12,
    "1.13": MC_1_13, "1.14": MC_1_14, "1.15": MC_1_15,
    "1.16": MC_1_16, "1.17": MC_1_17, "1.18": MC_1_18,
    "1.19": MC_1_19, "1.20": MC_1_20, "1.21": MC_1_21,
}

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------
DIM_OVERWORLD = 0
DIM_NETHER    = -1
DIM_END       = 1

DIMENSIONS = {
    "overworld": DIM_OVERWORLD,
    "nether":    DIM_NETHER,
    "end":       DIM_END,
}

# ---------------------------------------------------------------------------
# Biome ID <-> name maps  (cubiomes/biomes.h enum BiomeID)
# ---------------------------------------------------------------------------
BIOME_IDS: dict[str, int] = {
    # Classic biomes
    "ocean":                           0,
    "plains":                          1,
    "desert":                          2,
    "mountains":                       3,
    "forest":                          4,
    "taiga":                           5,
    "swamp":                           6,
    "river":                           7,
    "nether_wastes":                   8,
    "the_end":                         9,
    "frozen_ocean":                    10,
    "frozen_river":                    11,
    "snowy_tundra":                    12,
    "snowy_mountains":                 13,
    "mushroom_fields":                 14,
    "mushroom_field_shore":            15,
    "beach":                           16,
    "desert_hills":                    17,
    "wooded_hills":                    18,
    "taiga_hills":                     19,
    "mountain_edge":                   20,
    "jungle":                          21,
    "jungle_hills":                    22,
    "jungle_edge":                     23,
    "deep_ocean":                      24,
    "stone_shore":                     25,
    "snowy_beach":                     26,
    "birch_forest":                    27,
    "birch_forest_hills":              28,
    "dark_forest":                     29,
    "snowy_taiga":                     30,
    "snowy_taiga_hills":               31,
    "giant_tree_taiga":                32,
    "giant_tree_taiga_hills":          33,
    "wooded_mountains":                34,
    "savanna":                         35,
    "savanna_plateau":                 36,
    "badlands":                        37,
    "wooded_badlands_plateau":         38,
    "badlands_plateau":                39,
    # 1.13+
    "small_end_islands":               40,
    "end_midlands":                    41,
    "end_highlands":                   42,
    "end_barrens":                     43,
    "warm_ocean":                      44,
    "lukewarm_ocean":                  45,
    "cold_ocean":                      46,
    "deep_warm_ocean":                 47,
    "deep_lukewarm_ocean":             48,
    "deep_cold_ocean":                 49,
    "deep_frozen_ocean":               50,
    # Mutated variants (IDs 128+)
    "sunflower_plains":                129,
    "desert_lakes":                    130,
    "gravelly_mountains":              131,
    "flower_forest":                   132,
    "taiga_mountains":                 133,
    "swamp_hills":                     134,
    "ice_spikes":                      140,
    "modified_jungle":                 149,
    "modified_jungle_edge":            151,
    "tall_birch_forest":               155,
    "tall_birch_hills":                156,
    "dark_forest_hills":               157,
    "snowy_taiga_mountains":           158,
    "giant_spruce_taiga":              160,
    "giant_spruce_taiga_hills":        161,
    "modified_gravelly_mountains":     162,
    "shattered_savanna":               163,
    "shattered_savanna_plateau":       164,
    "eroded_badlands":                 165,
    "modified_wooded_badlands_plateau": 166,
    "modified_badlands_plateau":       167,
    # 1.16 Nether
    "soul_sand_valley":                170,
    "crimson_forest":                  171,
    "warped_forest":                   172,
    "basalt_deltas":                   173,
    # 1.18+ Overworld
    "meadow":                          177,
    "grove":                           178,
    "snowy_slopes":                    179,
    "jagged_peaks":                    180,
    "frozen_peaks":                    181,
    "stony_peaks":                     182,
    "deep_dark":                       183,
    "mangrove_swamp":                  184,
    "cherry_grove":                    185,
}

# Aliases — common alternative names users might type
_ALIASES: dict[str, str] = {
    "extreme_hills":       "mountains",
    "mega_taiga":          "giant_tree_taiga",
    "mega_taiga_hills":    "giant_tree_taiga_hills",
    "roofed_forest":       "dark_forest",
    "cold_taiga":          "snowy_taiga",
    "cold_taiga_hills":    "snowy_taiga_hills",
    "ice_plains":          "snowy_tundra",
    "ice_mountains":       "snowy_mountains",
    "mushroom_island":     "mushroom_fields",
    "stone_beach":         "stone_shore",
    "cold_beach":          "snowy_beach",
    "mesa":                "badlands",
    "mesa_plateau_f":      "wooded_badlands_plateau",
    "mesa_plateau":        "badlands_plateau",
    "hell":                "nether_wastes",
    "sky":                 "the_end",
    "nether":              "nether_wastes",
    "deep_warm":           "deep_warm_ocean",
    "deep_lukewarm":       "deep_lukewarm_ocean",
    "deep_cold":           "deep_cold_ocean",
    "deep_frozen":         "deep_frozen_ocean",
}

BIOME_NAMES: dict[int, str] = {v: k for k, v in BIOME_IDS.items()}


def resolve_biome_name(name: str) -> int | None:
    """Return biome ID for a name string, or None if not found.
    Accepts canonical names, aliases, and is case/space-insensitive."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    # Try alias first
    key = _ALIASES.get(key, key)
    return BIOME_IDS.get(key)


def list_biomes() -> str:
    """Return a human-readable list of all known biome names."""
    return "\n".join(sorted(BIOME_IDS.keys()))


# ---------------------------------------------------------------------------
# ctypes function signatures
# ---------------------------------------------------------------------------
_lib.setupGenerator.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
_lib.setupGenerator.restype  = None

_lib.applySeed.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
_lib.applySeed.restype  = None

_lib.getBiomeAt.argtypes = [ctypes.c_void_p, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.getBiomeAt.restype  = ctypes.c_int


# ---------------------------------------------------------------------------
# BiomeGenerator — main public class
# ---------------------------------------------------------------------------
class BiomeGenerator:
    """
    Wraps a cubiomes Generator struct.  One instance per thread is sufficient
    since apply_seed() is cheap; however the object is NOT thread-safe.

    Usage::

        gen = BiomeGenerator(mc_version=MC_1_21, dim=DIM_OVERWORLD)
        gen.apply_seed(12345)
        biome_id   = gen.get_biome(0, 64, 0)   # block coords x=0, y=64, z=0
        biome_name = gen.biome_name(biome_id)
    """

    def __init__(self, mc_version: int = MC_1_21, flags: int = 0,
                 dim: int = DIM_OVERWORLD):
        self._buf = (ctypes.c_uint8 * GENERATOR_SIZE)()
        self._ptr = ctypes.cast(self._buf, ctypes.c_void_p)
        self.mc_version = mc_version
        self.dim = dim
        _lib.setupGenerator(self._ptr, mc_version, flags)

    def apply_seed(self, seed: int) -> None:
        """Apply a world seed (up to 64-bit)."""
        _lib.applySeed(self._ptr, self.dim, ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF))

    def get_biome(self, x: int, y: int, z: int, scale: int = 1) -> int:
        """
        Return the biome ID at position (x, y, z).

        scale=1  → block coordinates (slower, exact)
        scale=4  → biome/chunk coordinates (faster, ~4-block granularity)

        For pre-1.18 (layer-based), y is ignored.
        For 1.18+ (noise-based), y matters; use 64 for sea-level surface.
        """
        return _lib.getBiomeAt(self._ptr, scale, x, y, z)

    def biome_name(self, biome_id: int) -> str:
        """Return a human-readable name for a biome ID."""
        return BIOME_NAMES.get(biome_id, f"unknown({biome_id})")

    def check_seed(self, seed: int,
                   requirements: list[tuple[int, int, int, set[int]]]
                   ) -> tuple[bool, list[tuple[int, int, str]]]:
        """
        Apply *seed* and verify every biome requirement.

        requirements — list of (x, z, y, allowed_biome_id_set)
        Returns (all_passed, [(x, z, biome_name), ...])
        """
        self.apply_seed(seed)
        results: list[tuple[int, int, str]] = []
        passed = True
        for x, z, y, allowed in requirements:
            bid = self.get_biome(x, y, z)
            name = self.biome_name(bid)
            if bid not in allowed:
                passed = False
            results.append((x, z, name))
        return passed, results


# ---------------------------------------------------------------------------
# Interactive helper — collect biome requirements from the user
# ---------------------------------------------------------------------------
def prompt_biome_requirements() -> tuple[int, int, list[tuple[int, int, int, set[int]]]] | None:
    """
    Ask the user whether they want biome filtering, and if so, collect:
      - MC version
      - Dimension
      - A list of (x, z, y, {allowed biome IDs})

    Returns (mc_version, dim, requirements) or None if skipped.
    """
    ans = input("Enable biome filtering? (y/n) [n]: ").strip().lower()
    if ans not in ("y", "yes"):
        return None

    print("\nAvailable MC versions:", ", ".join(MC_VERSIONS.keys()))
    ver_str = input("Minecraft version [1.21]: ").strip() or "1.21"
    mc_version = MC_VERSIONS.get(ver_str)
    if mc_version is None:
        print(f"  Unknown version '{ver_str}', defaulting to 1.21")
        mc_version = MC_1_21

    print("Dimension: overworld / nether / end")
    dim_str = input("Dimension [overworld]: ").strip().lower() or "overworld"
    dim = DIMENSIONS.get(dim_str, DIM_OVERWORLD)

    print()
    print("Enter biome requirements one per line.")
    print("  Format:  <x> <z> <biome>[,<biome>...]")
    print("  y-level defaults to 64 (sea level). Append it as 4th token to override.")
    print("  Example: 0 0 mushroom_fields")
    print("  Example: 100 -200 plains,forest,meadow")
    print("  Example: 0 0 64 ocean          (explicit y)")
    print("  Type 'list' to see all biome names.  Empty line to finish.")
    print()

    requirements: list[tuple[int, int, int, set[int]]] = []
    idx = 1
    while True:
        raw = input(f"Requirement {idx}: ").strip()
        if not raw:
            break
        if raw.lower() == "list":
            print(list_biomes())
            continue
        parts = raw.split()
        if len(parts) < 3:
            print("  Need at least: x z biome_name")
            continue
        try:
            x = int(parts[0])
            z = int(parts[1])
            # Optional explicit y
            if len(parts) >= 4:
                y = int(parts[2])
                biome_str = parts[3]
            else:
                y = 64
                biome_str = parts[2]
        except ValueError:
            print("  x, z (and optional y) must be integers.")
            continue

        biome_names_raw = [b.strip() for b in biome_str.split(",")]
        allowed: set[int] = set()
        bad = []
        for bn in biome_names_raw:
            bid = resolve_biome_name(bn)
            if bid is None:
                bad.append(bn)
            else:
                allowed.add(bid)
        if bad:
            print(f"  Unknown biome(s): {', '.join(bad)}.  Type 'list' to see valid names.")
            continue
        requirements.append((x, z, y, allowed))
        biome_labels = ", ".join(BIOME_NAMES.get(b, str(b)) for b in allowed)
        print(f"  Added: ({x}, {z}, y={y}) must be one of [{biome_labels}]")
        idx += 1

    if not requirements:
        print("  No requirements entered — biome filtering disabled.")
        return None

    return mc_version, dim, requirements

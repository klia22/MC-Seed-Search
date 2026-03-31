"""
structure.py — Bedrock Edition structural position calculations.

Implements the Mersenne Twister RNG used by Minecraft Bedrock to determine
where structures are placed within each region, and the getpos() function
that combines the world seed, region coordinates, and RNG constants into a
block-level structure position.
"""

import numba as nb
import numpy as np


# ---------------------------------------------------------------------------
# Mersenne Twister constants
# ---------------------------------------------------------------------------
MASK32     = 0xffffffff
N          = 624
M          = 397
MATRIX_A   = 0x9908b0df
UPPER_MASK = 0x80000000
LOWER_MASK = 0x7fffffff


@nb.njit(cache=True)
def mt_init(seed32):
    """Initialise a 624-element MT state from a 32-bit seed."""
    mt = np.empty(N, dtype=np.uint32)
    mt[0] = seed32
    for i in range(1, N):
        mt[i] = (0x6c078965 * (mt[i-1] ^ (mt[i-1] >> 30)) + i) & MASK32
    return mt


@nb.njit(cache=True)
def mt_twist(mt):
    """Apply one full twist to the MT state array."""
    for i in range(N):
        y = (mt[i] & UPPER_MASK) | (mt[(i+1) % N] & LOWER_MASK)
        mt[i] = mt[(i + M) % N] ^ (y >> 1)
        if y & 1:
            mt[i] ^= MATRIX_A


@nb.njit(cache=True)
def mt_extract(mt, idx):
    """Extract and temper one value from the MT state, twisting when needed."""
    if idx >= N:
        mt_twist(mt)
        idx = 0
    y = mt[idx]
    y ^= (y >> 11)
    y ^= (y << 7)  & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= (y >> 18)
    return y & MASK32, idx + 1


def getpos(world_seed, rx, rz, spacing, separation, salt, linear_separation):
    """
    Return the block-level (x, z) position of a structure candidate in
    region (rx, rz) for the given world seed and structure RNG constants.

    Parameters
    ----------
    world_seed       : 48-bit world seed
    rx, rz           : region coordinates (integers)
    spacing          : region size in chunks
    separation       : minimum separation in chunks
    salt             : structure-specific RNG salt
    linear_separation: if True uses the averaged two-draw algorithm
    """
    spawn_range = spacing - separation
    mixed = (world_seed + rx * 341873128712 + rz * 132897987541 + salt) & ((1 << 64) - 1)
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
    return (chunk_x * 16 + 8, chunk_z * 16 + 8)

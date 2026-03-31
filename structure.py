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


@nb.njit(cache=True)
def _scan_batch(seeds_start, seeds_end, spacing, separation, salt,
                linear_sep, radius, occurence):
    """
    Numba-compiled inner loop.  Scans seeds in [seeds_start, seeds_end) and
    returns a numpy int64 array of seeds whose structure positions satisfy the
    radius / occurrence requirements.

    Regions checked (rx, rz): (0,0), (-1,0), (0,-1), (-1,-1).
    Early exit: once it is impossible for the remaining regions to push the
    hit count up to `occurence`, the seed is discarded immediately.
    """
    spawn_range = spacing - separation
    R_X = np.int64(341873128712)
    R_Z = np.int64(132897987541)

    buf   = np.empty(seeds_end - seeds_start, dtype=np.int64)
    count = np.int64(0)

    for world_seed in range(seeds_start, seeds_end):
        found = np.int32(0)

        for i in range(4):
            # region order: (0,0) → (-1,0) → (0,-1) → (-1,-1)
            rx = np.int64(-(i & 1))
            rz = np.int64(-((i >> 1) & 1))

            # lower 32 bits of the mixed seed (overflow wraps naturally)
            seed32 = np.uint32(world_seed + rx * R_X + rz * R_Z + salt)

            mt  = mt_init(seed32)
            idx = N
            r0, idx = mt_extract(mt, idx)
            r1, idx = mt_extract(mt, idx)

            sr = np.int64(spawn_range)
            if linear_sep:
                r2, idx = mt_extract(mt, idx)
                r3, idx = mt_extract(mt, idx)
                off_x = (np.int64(r0) % sr + np.int64(r1) % sr) // np.int64(2)
                off_z = (np.int64(r2) % sr + np.int64(r3) % sr) // np.int64(2)
            else:
                off_x = np.int64(r0) % sr
                off_z = np.int64(r1) % sr

            bx = (rx * np.int64(spacing) + off_x) * np.int64(16) + np.int64(8)
            bz = (rz * np.int64(spacing) + off_z) * np.int64(16) + np.int64(8)

            if -radius < bx < radius and -radius < bz < radius:
                found += np.int32(1)

            # early exit: remaining regions cannot bring found to occurence
            if found + np.int32(3 - i) < np.int32(occurence):
                break

        if found >= np.int32(occurence):
            buf[count] = world_seed
            count += np.int64(1)

    return buf[:count]


def scan_batch(seeds_start, seeds_end, spacing, separation, salt,
               linear_sep, radius, occurence):
    """Python wrapper — triggers numba JIT on first call, cached afterwards."""
    return _scan_batch(int(seeds_start), int(seeds_end),
                       int(spacing), int(separation), int(salt),
                       bool(linear_sep), int(radius), int(occurence))


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

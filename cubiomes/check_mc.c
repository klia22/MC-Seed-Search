#include <stdio.h>
#include "biomes.h"
int main() {
    printf("MC_1_13=%d MC_1_14=%d MC_1_15=%d MC_1_16=%d MC_1_17=%d MC_1_18=%d MC_1_19=%d MC_1_20=%d MC_1_21=%d\n",
        MC_1_13, MC_1_14, MC_1_15, MC_1_16, MC_1_17, MC_1_18, MC_1_19, MC_1_20, MC_1_21);
    // Check some biome IDs that 1.18+ added
    printf("meadow=%d grove=%d snowy_slopes=%d jagged_peaks=%d frozen_peaks=%d\n",
        meadow, grove, snowy_slopes, jagged_peaks, frozen_peaks);
    printf("mangrove_swamp=%d cherry_grove=%d\n", mangrove_swamp, cherry_grove);
    return 0;
}

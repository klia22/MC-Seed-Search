#include <stdio.h>
#include "generator.h"
int main() {
    printf("sizeof(Generator)=%zu\n", sizeof(Generator));
    printf("sizeof(int)=%zu\n", sizeof(int));
    printf("MC_1_21=%d\n", MC_1_21);
    printf("MC_1_18=%d\n", MC_1_18);
    printf("MC_1_16=%d\n", MC_1_16);
    printf("ocean=%d plains=%d desert=%d forest=%d taiga=%d\n", ocean, plains, desert, forest, taiga);
    printf("jungle=%d mushroom_fields=%d swamp=%d badlands=%d savanna=%d\n", jungle, mushroom_fields, swamp, badlands, savanna);
    return 0;
}

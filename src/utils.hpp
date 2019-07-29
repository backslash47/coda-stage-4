#include <cstdint>
#include "constants.hpp"

void print_array(uint8_t *a);
void print_array_dec(uint8_t *a, int size);
void printG1(uint8_t *src);

void print_mnt4_G2(uint8_t *src);
void print_mnt6_G2(uint8_t *src);

int getDimension(GEnum e);
int getKnaryK(GEnum Gx);
int getKnarySize(GEnum Gx);
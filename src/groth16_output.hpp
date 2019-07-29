#pragma once

#include <cstdint>
#include <cstdio>
#include "constants.hpp"

template <ECurve ecurve>
class groth16_output
{
public:
  void write(uint8_t *evaluation_At, uint8_t *evaluation_Bt2, uint8_t *C, FILE *output);
};

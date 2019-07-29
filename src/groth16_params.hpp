#pragma once

#include <cstdio>
#include <cstdint>
#include "constants.hpp"

template <ECurve curve>
class Groth16Params
{
public:
  size_t d;
  size_t m;
  uint8_t *A, *B1, *L, *H; // G1
  uint8_t *B2;             // G2

  void load(FILE *file);
};
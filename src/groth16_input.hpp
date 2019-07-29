#pragma once

#include <cstdio>
#include <cstdint>
#include "constants.hpp"

template <ECurve curve>
class Groth16Input
{
public:
  uint8_t *w, *ca, *cb, *cc;
  uint8_t *r;

  void load(FILE *file, size_t d, size_t m);
};
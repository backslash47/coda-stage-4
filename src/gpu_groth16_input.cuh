#pragma once

#include <cstdio>
#include <cstdint>
#include "constants.hpp"
#include "gpu_constants.cuh"
#include "groth16_input.hpp"

template <ECurve curve>
class GpuGroth16Input
{
public:
  my_fixnum_array *w;

  void load(Groth16Input<curve>* input, long m);
  void destroy();
  my_fixnum_array* offseted(int offset);
};
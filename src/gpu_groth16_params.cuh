#pragma once

#include "gpu_constants.cuh"
#include "groth16_params.hpp"

template <ECurve curve>
class GpuGroth16Params
{
public:
  my_fixnum_array *A, *B1, *L, *H; // G1
  my_fixnum_array *B2;             // G2

  void load(Groth16Params<curve> *params);
  void destroy();
};
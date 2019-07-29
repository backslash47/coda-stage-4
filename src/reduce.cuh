#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "constants.hpp"
#include "gpu_constants.cuh"

template <GEnum Gx>
void reduce_g(cudaStream_t stream, my_fixnum_array *a, long nelts);
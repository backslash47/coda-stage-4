#pragma once

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "constants.hpp"
#include "gpu_constants.cuh"

template <GEnum Gx, int knary_k>
void multiexp(bool debug, cudaStream_t stream, my_fixnum_array *gpu_w, my_fixnum_array *gpu_a, my_fixnum_array *gpu_o, size_t n);

template <GEnum Gx, int knary_k>
void multiexp_prepared(bool debug, cudaStream_t stream, my_fixnum_array *gpu_w, my_fixnum_array *gpu_a, my_fixnum_array *gpu_o, my_fixnum_array *gpu_p, size_t n);
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "constants.hpp"
#include "gpu_constants.cuh"

template <GEnum Gx, int K>
void scale_batch_knary(cudaStream_t stream, my_fixnum_array *s, my_fixnum_array *a, my_fixnum_array *out, long nelts);

template <GEnum Gx>
void scale_knary(cudaStream_t stream, my_fixnum_array *gpu_s, my_fixnum_array *gpu_a, my_fixnum_array *gpu_o, long nelts);

template <GEnum Gx>
void scale_prepared(cudaStream_t stream, my_fixnum_array *s, my_fixnum_array *out, my_fixnum_array *p, long nelts);

template <GEnum Gx>
void scale_prepared_batched(cudaStream_t stream, my_fixnum_array *gpu_s, my_fixnum_array *gpu_o, my_fixnum_array *gpu_p, long nelts);

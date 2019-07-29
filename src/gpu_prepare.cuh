#include "gpu_constants.cuh"

template <typename GpuGx>
__device__ void multiexp_prepare(fixnum *input, GpuGx *combinations, long knary_chunk_size, long num);

template <typename GpuGx, int knary_k, int knary_size>
__device__ void multiexp_prepare_combinations(fixnum *input, GpuGx *combinations, long knary_chunk_size, long num);
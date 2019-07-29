#include "gpu_utils.cuh"

__device__ long get_thread_idx() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ long get_element_idx() {
  return get_thread_idx()/libms_per_elem;
}

__device__ fixnum* make_pointer(fixnum* address) {
  const long thread_idx = get_thread_idx();
  return address - thread_idx;
}
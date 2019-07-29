#pragma once 

#include "gpu_constants.cuh"

__device__ long get_thread_idx();

__device__ long get_element_idx();

__device__ fixnum* make_pointer(fixnum* address);

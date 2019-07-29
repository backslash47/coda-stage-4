#include "gpu_constants.cuh"
#include "gpu_g1.cuh"
#include "gpu_g2.cuh"

__device__ GpuG1 knaryG1(fixnum *scale_input, fixnum *g_input, long num);
__device__ GpuMnt4G2 knaryMnt4G2(fixnum *scale_input, fixnum *g_input, long num);
__device__ GpuMnt6G2 knaryMnt6G2(fixnum *scale_input, fixnum *g_input, long num);

__device__ GpuG1 knaryG1_prepared(fixnum *scale_input, GpuG1 *combinations, long num);
__device__ GpuMnt4G2 knaryMnt4G2_prepared(fixnum *scale_input, GpuMnt4G2 *combinations, long num);
__device__ GpuMnt6G2 knaryMnt6G2_prepared(fixnum *scale_input, GpuMnt6G2 *combinations, long num);

__device__ GpuG1 knaryG1Processed(fixnum *scale_input, fixnum *g_input, fixnum *p_input, long num);
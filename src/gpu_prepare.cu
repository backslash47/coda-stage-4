#include "gpu_prepare.cuh"
#include "gpu_g1.cuh"
#include "gpu_g2.cuh"
#include "gpu_utils.cuh"

// template <typename GpuGx>
// __device__ void multiexp_prepare(fixnum *input, GpuGx *combinations, long knary_chunk_size, long num)
// {
//   long el_idx = get_element_idx();

//   GpuGx single_terms[knary_k];
//   for (int i = 0; i < knary_k; i++)
//   {
//     bool overflow = el_idx + knary_chunk_size * i >= num;
//     single_terms[i] = overflow ? GpuGx::zero() : GpuGx::load_from_array(input, el_idx + knary_chunk_size * i);
//   }

//   combinations[0] = GpuGx::zero();

//   for (int j = 0; j < knary_k; j++)
//   {
//     const int offset = 1 << j;
//     combinations[offset] = single_terms[j];

//     for (int i = 1; i < offset; i++)
//     {
//       combinations[offset + i] = combinations[i].mixed_add(single_terms[j]);
//     }
//   }
// }

// template __device__ void multiexp_prepare<GpuG1>(fixnum *input, GpuG1 *combinations, long knary_chunk_size, long num);
// template __device__ void multiexp_prepare<GpuMnt4G2>(fixnum *input, GpuMnt4G2 *combinations, long knary_chunk_size, long num);
// template __device__ void multiexp_prepare<GpuMnt6G2>(fixnum *input, GpuMnt6G2 *combinations, long knary_chunk_size, long num);

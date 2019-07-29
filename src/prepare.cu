#include <cstring>
#include <libff/common/profiling.hpp>

#include "prepare.cuh"
#include "retrieve_utils.cuh"
#include "gpu_constants.cuh"
#include "constants.hpp"
#include "gpu_fq.cuh"
#include "gpu_g1.cuh"
#include "gpu_g2.cuh"
#include "knary.cuh"
#include "gpu_utils.cuh"
#include "utils.hpp"
#include "gpu_prepare.cuh"

using namespace libff;

template <typename fixnum>
struct prepare_g1_gpu
{
  __device__ void operator()(int n, fixnum *a, fixnum *o) const
  {
    long offset = get_thread_idx() * 3 * g1_knary_size;

    long knary_chunk_size = (n + g1_knary_k - 1) / g1_knary_k;
    multiexp_prepare_combinations<GpuG1, g1_knary_k, g1_knary_size>(a, (GpuG1 *)(o + offset), knary_chunk_size, n);
  }
};

template <typename fixnum>
struct prepare_mnt4_g2_gpu
{
  __device__ void operator()(int n, fixnum *a, fixnum *o) const
  {
    long offset = get_thread_idx() * 6 * g2_mnt4_knary_size;

    long knary_chunk_size = (n + g2_mnt4_knary_k - 1) / g2_mnt4_knary_k;
    multiexp_prepare_combinations<GpuMnt4G2, g2_mnt4_knary_k, g2_mnt4_knary_size>(a, (GpuMnt4G2 *)(o + offset), knary_chunk_size, n);
  }
};

template <typename fixnum>
struct prepare_mnt6_g2_gpu
{
  __device__ void operator()(int n, fixnum *a, fixnum *o) const
  {
    long offset = get_thread_idx() * 9 * g2_mnt6_knary_size;

    long knary_chunk_size = (n + g2_mnt6_knary_k - 1) / g2_mnt6_knary_k;
    multiexp_prepare_combinations<GpuMnt6G2, g2_mnt6_knary_k, g2_mnt6_knary_size>(a, (GpuMnt6G2 *)(o + offset), knary_chunk_size, n);
  }
};

template <GEnum Gx>
void prepare(cudaStream_t stream, my_fixnum_array *gpu_a, my_fixnum_array *gpu_o, long nelts)
{
  bool is_g1 = Gx == G1_MNT;
  bool is_mnt4_g2 = Gx == G2_MNT4;
  int dimension = getDimension(Gx);
  int knary_k = getKnaryK(Gx);

  long knary_chunk_size = (nelts + knary_k - 1) / knary_k;
  my_fixnum_array *gpu_a_chunk = my_fixnum_array::wrap(gpu_a->get_ptr(), knary_chunk_size);
  my_fixnum_array *gpu_o_chunk = my_fixnum_array::wrap(gpu_o->get_ptr(), knary_chunk_size);

  if (is_g1)
  {
    my_fixnum_array::template mapNoSync<prepare_g1_gpu>(stream, nelts, gpu_a_chunk, gpu_o_chunk);
  }
  else if (is_mnt4_g2)
  {
    my_fixnum_array::template mapNoSync<prepare_mnt4_g2_gpu>(stream, nelts, gpu_a_chunk, gpu_o_chunk);
  }
  else
  {
    my_fixnum_array::template mapNoSync<prepare_mnt6_g2_gpu>(stream, nelts, gpu_a_chunk, gpu_o_chunk);
  }
}

template void prepare<G1_MNT>(cudaStream_t stream, my_fixnum_array *a, my_fixnum_array *o, long nelts);
template void prepare<G2_MNT4>(cudaStream_t stream, my_fixnum_array *a, my_fixnum_array *o, long nelts);
template void prepare<G2_MNT6>(cudaStream_t stream, my_fixnum_array *a, my_fixnum_array *o, long nelts);

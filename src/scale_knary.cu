#include <cstring>
#include <libff/common/profiling.hpp>

#include "scale.cuh"
#include "gpu_preprocess.cuh"
#include "retrieve_utils.cuh"
#include "gpu_constants.cuh"
#include "constants.hpp"
#include "gpu_fq.cuh"
#include "gpu_g1.cuh"
#include "gpu_g2.cuh"
#include "knary.cuh"
#include "gpu_utils.cuh"
#include "utils.hpp"

using namespace libff;

template <typename fixnum>
struct scale_g1_gpu_knary
{
  __device__ void operator()(int n, fixnum *s, fixnum *a, fixnum *o) const
  {
    GpuG1 fqS = knaryG1(s, a, n);

    long el_idx = get_element_idx();
    fqS.save_to_array(o, el_idx);
  }
};

template <typename fixnum>
struct scale_mnt4_g2_gpu_knary
{
  __device__ void operator()(int n, fixnum *s, fixnum *a, fixnum *o) const
  {
    GpuMnt4G2 fqS = knaryMnt4G2(s, a, n);

    long el_idx = get_element_idx();
    fqS.save_to_array(o, el_idx);
  }
};

template <typename fixnum>
struct scale_mnt6_g2_gpu_knary
{
  __device__ void operator()(int n, fixnum *s, fixnum *a, fixnum *o) const
  {
    GpuMnt6G2 fqS = knaryMnt6G2(s, a, n);

    long el_idx = get_element_idx();
    fqS.save_to_array(o, el_idx);
  }
};

template <GEnum Gx>
void scale_knary(cudaStream_t stream, my_fixnum_array *gpu_s, my_fixnum_array *gpu_a, my_fixnum_array *gpu_o, long nelts)
{
  // const int batch_count = 2;
  // long batch_size = (nelts + batch_count - 1) / batch_count;

  // for (int i = 0; i < batch_count; i++)
  // {
  //   long batch_nelts = i + 1 == batch_count ? (batch_size - ((batch_count * batch_size) - nelts)) : batch_size;

  //   long s_offset = i * batch_size * bytes_per_elem;
  //   long a_offset = i * batch_size * bytes_per_elem * getDimension(Gx);
  //   long o_offset = i * batch_size * bytes_per_elem * getDimension(Gx);

  //   my_fixnum_array *gpu_s_batch = my_fixnum_array::wrap(gpu_s->get_ptr() + s_offset, gpu_s->length());
  //   my_fixnum_array *gpu_a_batch = my_fixnum_array::wrap(gpu_a->get_ptr() + a_offset, gpu_a->length());
  //   my_fixnum_array *gpu_o_batch = my_fixnum_array::wrap(gpu_o->get_ptr() + o_offset, gpu_o->length());

  //   scale_batch_knary<Gx>(stream, gpu_s_batch, gpu_a_batch, gpu_o_batch, batch_nelts);
  // }
}

//TODO: K is ignored
template <GEnum Gx, int K>
void scale_batch_knary(cudaStream_t stream, my_fixnum_array *gpu_s, my_fixnum_array *gpu_a, my_fixnum_array *gpu_o, long nelts)
{
  bool is_g1 = Gx == G1_MNT;
  bool is_mnt4_g2 = Gx == G2_MNT4;
  int dimension = getDimension(Gx);

  if (is_g1)
  {
    long knary_chunk_size = (nelts + g1_knary_k - 1) / g1_knary_k;
    my_fixnum_array *gpu_s_chunk = my_fixnum_array::wrap(gpu_s->get_ptr(), knary_chunk_size);
    my_fixnum_array *gpu_a_chunk = my_fixnum_array::wrap(gpu_a->get_ptr(), knary_chunk_size);
    my_fixnum_array *gpu_o_chunk = my_fixnum_array::wrap(gpu_o->get_ptr(), knary_chunk_size);

    my_fixnum_array::template mapNoSync<scale_g1_gpu_knary>(stream, nelts, gpu_s_chunk, gpu_a_chunk, gpu_o_chunk);
  }
  else if (is_mnt4_g2)
  {
    long knary_chunk_size = (nelts + g2_mnt4_knary_k - 1) / g2_mnt4_knary_k;
    my_fixnum_array *gpu_s_chunk = my_fixnum_array::wrap(gpu_s->get_ptr(), knary_chunk_size);
    my_fixnum_array *gpu_a_chunk = my_fixnum_array::wrap(gpu_a->get_ptr(), knary_chunk_size);
    my_fixnum_array *gpu_o_chunk = my_fixnum_array::wrap(gpu_o->get_ptr(), knary_chunk_size);

    my_fixnum_array::template mapNoSync<scale_mnt4_g2_gpu_knary>(stream, nelts, gpu_s_chunk, gpu_a_chunk, gpu_o_chunk);
  }
  else
  {
    long knary_chunk_size = (nelts + g2_mnt6_knary_k - 1) / g2_mnt6_knary_k;
    my_fixnum_array *gpu_s_chunk = my_fixnum_array::wrap(gpu_s->get_ptr(), knary_chunk_size);
    my_fixnum_array *gpu_a_chunk = my_fixnum_array::wrap(gpu_a->get_ptr(), knary_chunk_size);
    my_fixnum_array *gpu_o_chunk = my_fixnum_array::wrap(gpu_o->get_ptr(), knary_chunk_size);

    my_fixnum_array::template mapNoSync<scale_mnt6_g2_gpu_knary>(stream, nelts, gpu_s_chunk, gpu_a_chunk, gpu_o_chunk);
  }
}

template void scale_batch_knary<G1_MNT, g1_knary_k>(cudaStream_t stream, my_fixnum_array *s, my_fixnum_array *a, my_fixnum_array *o, long nelts);
template void scale_batch_knary<G2_MNT4, g2_mnt4_knary_k>(cudaStream_t stream, my_fixnum_array *s, my_fixnum_array *a, my_fixnum_array *o, long nelts);
template void scale_batch_knary<G2_MNT6, g2_mnt6_knary_k>(cudaStream_t stream, my_fixnum_array *s, my_fixnum_array *a, my_fixnum_array *o, long nelts);

template void scale_knary<G1_MNT>(cudaStream_t stream, my_fixnum_array *s, my_fixnum_array *a, my_fixnum_array *o, long nelts);
template void scale_knary<G2_MNT4>(cudaStream_t stream, my_fixnum_array *s, my_fixnum_array *a, my_fixnum_array *o, long nelts);
template void scale_knary<G2_MNT6>(cudaStream_t stream, my_fixnum_array *s, my_fixnum_array *a, my_fixnum_array *o, long nelts);

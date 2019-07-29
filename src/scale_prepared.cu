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
struct scale_prepared_g1_gpu
{
  __device__ void operator()(int n, fixnum *s, fixnum *o, fixnum *p) const
  {
    long offset = get_thread_idx() * 3 * g1_knary_size;

    GpuG1 fqS = knaryG1_prepared(s, (GpuG1 *)(p + offset), n);

    long el_idx = get_element_idx();
    fqS.save_to_array(o, el_idx);
  }
};

template <typename fixnum>
struct scale_prepared_mnt4_g2_gpu
{
  __device__ void operator()(int n, fixnum *s, fixnum *o, fixnum *p) const
  {
    long offset = get_thread_idx() * 6 * g2_mnt4_knary_size;

    GpuMnt4G2 fqS = knaryMnt4G2_prepared(s, (GpuMnt4G2 *)(p + offset), n);

    long el_idx = get_element_idx();
    fqS.save_to_array(o, el_idx);
  }
};

template <typename fixnum>
struct scale_prepared_mnt6_g2_gpu
{
  __device__ void operator()(int n, fixnum *s, fixnum *o, fixnum *p) const
  {
    long offset = get_thread_idx() * 9 * g2_mnt6_knary_size;

    // const long lane_idx = fixnum::layout::laneIdx();
    // if (lane_idx == 0)
    // {
    //   printf("offset: %ld\n", offset * sizeof(fixnum));
    // }

    GpuMnt6G2 fqS = knaryMnt6G2_prepared(s, (GpuMnt6G2 *)(p + offset), n);

    long el_idx = get_element_idx();
    fqS.save_to_array(o, el_idx);
  }
};

template <GEnum Gx>
void scale_prepared_batched(cudaStream_t stream, my_fixnum_array *gpu_s, my_fixnum_array *gpu_o, my_fixnum_array *gpu_p, long nelts)
{
  const int batch_count = 1;
  long batch_size = (nelts + batch_count - 1) / batch_count;

  for (int i = 0; i < batch_count; i++)
  {

    long real_batch_size = i + 1 == batch_count ? (batch_size - ((batch_count * batch_size) - nelts)) : batch_size;
    long element_offset = i * batch_size;
    long s_offset = element_offset * bytes_per_elem;
    long o_offset = element_offset * bytes_per_elem * getDimension(Gx);
    long p_offset = element_offset * bytes_per_elem * getDimension(Gx) * getKnarySize(Gx);

    my_fixnum_array *gpu_s_batch = my_fixnum_array::wrap(gpu_s->get_ptr() + s_offset, gpu_s->length());
    my_fixnum_array *gpu_o_batch = my_fixnum_array::wrap(gpu_o->get_ptr() + o_offset, gpu_o->length());
    my_fixnum_array *gpu_p_batch = my_fixnum_array::wrap(gpu_p->get_ptr() + p_offset, gpu_p->length());

    printf("real batch size: %ld\n", real_batch_size);
    printf("s_offset: %ld\n", s_offset);
    scale_prepared<Gx>(stream, gpu_s_batch, gpu_o_batch, gpu_p_batch, real_batch_size);
    cudaStreamSynchronize(stream);
  }
}

template <GEnum Gx>
void scale_prepared(cudaStream_t stream, my_fixnum_array *gpu_s, my_fixnum_array *gpu_o, my_fixnum_array *gpu_p, long nelts)
{
  bool is_g1 = Gx == G1_MNT;
  bool is_mnt4_g2 = Gx == G2_MNT4;
  int dimension = getDimension(Gx);

  if (is_g1)
  {
    long knary_chunk_size = (nelts + g1_knary_k - 1) / g1_knary_k;
    my_fixnum_array *gpu_s_chunk = my_fixnum_array::wrap(gpu_s->get_ptr(), knary_chunk_size);
    my_fixnum_array *gpu_o_chunk = my_fixnum_array::wrap(gpu_o->get_ptr(), knary_chunk_size);

    my_fixnum_array::template mapNoSync<scale_prepared_g1_gpu>(stream, nelts, gpu_s_chunk, gpu_o_chunk, gpu_p);
  }
  else if (is_mnt4_g2)
  {
    long knary_chunk_size = (nelts + g2_mnt4_knary_k - 1) / g2_mnt4_knary_k;
    my_fixnum_array *gpu_s_chunk = my_fixnum_array::wrap(gpu_s->get_ptr(), knary_chunk_size);
    my_fixnum_array *gpu_o_chunk = my_fixnum_array::wrap(gpu_o->get_ptr(), knary_chunk_size);

    my_fixnum_array::template mapNoSync<scale_prepared_mnt4_g2_gpu>(stream, nelts, gpu_s_chunk, gpu_o_chunk, gpu_p);
  }
  else
  {
    long knary_chunk_size = (nelts + g2_mnt6_knary_k - 1) / g2_mnt6_knary_k;
    my_fixnum_array *gpu_s_chunk = my_fixnum_array::wrap(gpu_s->get_ptr(), knary_chunk_size);
    my_fixnum_array *gpu_o_chunk = my_fixnum_array::wrap(gpu_o->get_ptr(), knary_chunk_size);

    my_fixnum_array::template mapNoSync<scale_prepared_mnt6_g2_gpu>(stream, nelts, gpu_s_chunk, gpu_o_chunk, gpu_p);
  }
}

template void scale_prepared<G1_MNT>(cudaStream_t stream, my_fixnum_array *s, my_fixnum_array *o, my_fixnum_array *p, long nelts);
template void scale_prepared<G2_MNT4>(cudaStream_t stream, my_fixnum_array *s, my_fixnum_array *o, my_fixnum_array *p, long nelts);
template void scale_prepared<G2_MNT6>(cudaStream_t stream, my_fixnum_array *s, my_fixnum_array *o, my_fixnum_array *p, long nelts);

template void scale_prepared_batched<G1_MNT>(cudaStream_t stream, my_fixnum_array *gpu_s, my_fixnum_array *gpu_o, my_fixnum_array *gpu_p, long nelts);
template void scale_prepared_batched<G2_MNT4>(cudaStream_t stream, my_fixnum_array *gpu_s, my_fixnum_array *gpu_o, my_fixnum_array *gpu_p, long nelts);
template void scale_prepared_batched<G2_MNT6>(cudaStream_t stream, my_fixnum_array *gpu_s, my_fixnum_array *gpu_o, my_fixnum_array *gpu_p, long nelts);

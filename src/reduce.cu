#include <cstring>

#include "reduce.cuh"
#include "constants.hpp"

#include "retrieve_utils.cuh"
#include "gpu_constants.cuh"
#include "gpu_fq.cuh"
#include "gpu_g1.cuh"
#include "gpu_g2.cuh"
#include "gpu_utils.cuh"

template <typename fixnum>
struct reduce_g1_gpu
{
  __device__ void operator()(int n, fixnum *a, fixnum *b) const
  {
    long el_idx = get_element_idx();

    GpuG1 gA = GpuG1::load_from_array(a, el_idx);
    GpuG1 gB = GpuG1::load_from_array(b, el_idx);

    GpuG1 fqS = gA + gB;
    fqS.save_to_array(a, el_idx);
  }
};

template <typename fixnum>
struct reduce_mnt4_g2_gpu
{
  __device__ void operator()(int n, fixnum *a, fixnum *b) const
  {
    long el_idx = get_element_idx();

    GpuMnt4G2 gA = GpuMnt4G2::load_from_array(a, el_idx);
    GpuMnt4G2 gB = GpuMnt4G2::load_from_array(b, el_idx);

    GpuMnt4G2 fqS = gA + gB;
    fqS.save_to_array(a, el_idx);
  }
};

template <typename fixnum>
struct reduce_mnt6_g2_gpu
{
  __device__ void operator()(int n, fixnum *a, fixnum *b) const
  {
    long el_idx = get_element_idx();

    GpuMnt6G2 gA = GpuMnt6G2::load_from_array(a, el_idx);
    GpuMnt6G2 gB = GpuMnt6G2::load_from_array(b, el_idx);

    GpuMnt6G2 fqS = gA + gB;
    fqS.save_to_array(a, el_idx);
  }
};

template <GEnum Gx>
void reduce_g_internal(cudaStream_t stream, my_fixnum_array *in_a, int nelts)
{
  bool is_g1 = Gx == G1_MNT;
  bool is_mnt4_g2 = Gx == G2_MNT4;

  int dimension = is_g1 ? 3 : (is_mnt4_g2 ? 6 : 9);

  if (nelts == 1)
  {
    return;
  }
  else
  {
    int m = nelts / 2;
    my_fixnum_array *new_in_a = my_fixnum_array::wrap(in_a->get_ptr(), m);
    my_fixnum_array *new_in_b = my_fixnum_array::wrap(in_a->get_ptr() + dimension * bytes_per_elem * m, m);

    if (is_g1)
    {
      my_fixnum_array::template mapNoSync<reduce_g1_gpu>(stream, 0, new_in_a, new_in_b);
    }
    else if (is_mnt4_g2)
    {
      my_fixnum_array::template mapNoSync<reduce_mnt4_g2_gpu>(stream, 0, new_in_a, new_in_b);
    }
    else
    {
      my_fixnum_array::template mapNoSync<reduce_mnt6_g2_gpu>(stream, 0, new_in_a, new_in_b);
    }

    if (nelts % 2 == 1)
    {
      my_fixnum_array *first = my_fixnum_array::wrap(in_a->get_ptr(), 1);
      my_fixnum_array *last = my_fixnum_array::wrap(in_a->get_ptr() + dimension * bytes_per_elem * m * 2, 1);

      if (is_g1)
      {
        my_fixnum_array::template mapNoSync<reduce_g1_gpu>(stream, 0, first, last);
      }
      else if (is_mnt4_g2)
      {
        my_fixnum_array::template mapNoSync<reduce_mnt4_g2_gpu>(stream, 0, first, last);
      }
      else
      {
        my_fixnum_array::template mapNoSync<reduce_mnt6_g2_gpu>(stream, 0, first, last);
      }
    }

    // next round
    // input is previous result divided into two halves
    reduce_g_internal<Gx>(stream, new_in_a, m);
  }
}

template <GEnum Gx>
void reduce_g(cudaStream_t stream, my_fixnum_array *gpu_a, long nelts)
{
  reduce_g_internal<Gx>(stream, gpu_a, nelts);
}

template void reduce_g<G1_MNT>(cudaStream_t stream, my_fixnum_array *a, long nelts);
template void reduce_g<G2_MNT4>(cudaStream_t stream, my_fixnum_array *a, long nelts);
template void reduce_g<G2_MNT6>(cudaStream_t stream, my_fixnum_array *a, long nelts);
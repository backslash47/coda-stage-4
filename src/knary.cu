#include "knary.cuh"
#include "constants.hpp"
#include "gpu_utils.cuh"
#include "gpu_prepare.cuh"

__device__ bool test_bit(const fixnum &base, int bitno)
{
  long bit = bitno % fixnum::digit::BITS;
  long part = bitno / fixnum::digit::BITS;

  typedef typename fixnum::digit digit;

  const digit partDigit = fixnum::get(base, part);
  digit one = fixnum::digit::one();
  digit shifted, result;

  fixnum::digit::lshift(shifted, one, bit);

  fixnum::digit::andOperator(result, partDigit, shifted);
  return !fixnum::digit::is_zero(result);
}

__device__ fixnum loadScalar(fixnum *array, int i)
{
  const long lane_idx = fixnum::layout::laneIdx();
  return array[i * libms_per_elem + lane_idx];
}

template <int knary_k>
__device__ int maxMsb(fixnum *scale_input, long knary_chunk_size, long num)
{
  long el_idx = get_element_idx();

  int max = 0;
  for (int j = 0; j < knary_k; j++)
  {
    long scalar_index = el_idx * knary_k + j;
    bool overflow = scalar_index >= num;
    if (overflow)
    {
      //printf("overflow: %ld\n", el_idx + knary_chunk_size * j);
      break;
    }

    fixnum scalar = loadScalar(scale_input, scalar_index);

    int msb = fixnum::msb(scalar);
    max = msb > max ? msb : max;
  }

  return max;
}

template <int knary_k>
__device__ int constructBitIndex(fixnum *scale_input, long knary_chunk_size, long num, int i)
{
  long el_idx = get_element_idx();
  int index = 0;

  for (int j = 0; j < knary_k; j++)
  {
    long scalar_index = el_idx * knary_k + j;

    bool overflow = scalar_index >= num;
    if (overflow)
    {
      break;
    }

    fixnum scalar = loadScalar(scale_input, scalar_index);
    index |= test_bit(scalar, i) << j;
  }

  return index;
}

template <typename GpuGx, int knary_k>
__device__ GpuGx knary(fixnum *scale_input, GpuGx *combinations, long knary_chunk_size, long num)
{
  const int msb = maxMsb<knary_k>(scale_input, knary_chunk_size, num);

  GpuGx result = GpuGx::zero();
  bool found_one = false;

  for (int i = msb; i >= 0; --i)
  {
    if (found_one)
    {
      result = result.dbl();
    }

    const int index = constructBitIndex<knary_k>(scale_input, knary_chunk_size, num, i);

    if (index > 0)
    {
      found_one = true;

      result = result + combinations[index];
    }
  }

  return result;
}

__device__ GpuG1 knaryG1(fixnum *scale_input, fixnum *g_input, long num)
{
  GpuG1 combinations[g1_knary_size];

  long knary_chunk_size = (num + g1_knary_k - 1) / g1_knary_k;
  multiexp_prepare_combinations<GpuG1, g1_knary_k, g1_knary_size>(g_input, combinations, knary_chunk_size, num);

  return knary<GpuG1, g1_knary_k>(scale_input, combinations, knary_chunk_size, num);
}

__device__ GpuMnt4G2 knaryMnt4G2(fixnum *scale_input, fixnum *g_input, long num)
{
  GpuMnt4G2 combinations[g2_mnt4_knary_size];

  long knary_chunk_size = (num + g2_mnt4_knary_k - 1) / g2_mnt4_knary_k;
  multiexp_prepare_combinations<GpuMnt4G2, g2_mnt4_knary_k, g2_mnt4_knary_size>(g_input, combinations, knary_chunk_size, num);

  return knary<GpuMnt4G2, g2_mnt4_knary_k>(scale_input, combinations, knary_chunk_size, num);
}

__device__ GpuMnt6G2 knaryMnt6G2(fixnum *scale_input, fixnum *g_input, long num)
{
  GpuMnt6G2 combinations[g2_mnt6_knary_size];

  long knary_chunk_size = (num + g2_mnt6_knary_k - 1) / g2_mnt6_knary_k;
  multiexp_prepare_combinations<GpuMnt6G2, g2_mnt6_knary_k, g2_mnt6_knary_size>(g_input, combinations, knary_chunk_size, num);

  return knary<GpuMnt6G2, 3>(scale_input, combinations, knary_chunk_size, num);
}

__device__ GpuG1 knaryG1_prepared(fixnum *scale_input, GpuG1 *combinations, long num)
{
  long knary_chunk_size = (num + g1_knary_k - 1) / g1_knary_k;
  return knary<GpuG1, g1_knary_k>(scale_input, combinations, knary_chunk_size, num);
}

__device__ GpuMnt4G2 knaryMnt4G2_prepared(fixnum *scale_input, GpuMnt4G2 *combinations, long num)
{
  long knary_chunk_size = (num + g2_mnt4_knary_k - 1) / g2_mnt4_knary_k;
  return knary<GpuMnt4G2, g2_mnt4_knary_k>(scale_input, combinations, knary_chunk_size, num);
}

__device__ GpuMnt6G2 knaryMnt6G2_prepared(fixnum *scale_input, GpuMnt6G2 *combinations, long num)
{
  long knary_chunk_size = (num + g2_mnt6_knary_k - 1) / g2_mnt6_knary_k;
  return knary<GpuMnt6G2, g2_mnt6_knary_k>(scale_input, combinations, knary_chunk_size, num);
}

template <typename GpuGx, int knary_k, int knary_size>
__device__ void multiexp_prepare_combinations(fixnum *input, GpuGx *combinations, long knary_chunk_size, long num)
{
  long el_idx = get_element_idx();

  GpuGx single_terms[knary_k];
  for (int i = 0; i < knary_k; i++)
  {
    long g_index = el_idx * knary_k + i;
    bool overflow = g_index >= num;
    single_terms[i] = overflow ? GpuGx::zero() : GpuGx::load_from_array(input, g_index);
  }

  for (int i = 0; i < knary_size; i++)
  {
    bool found_one = false;
    GpuGx acc = GpuGx::zero();

    for (int j = 0; j < knary_k; j++)
    {
      int b = (i >> j) & 1;

      if (b == 1)
      {
        long g_index = el_idx * knary_k + j;
        bool overflow = g_index >= num;
        // GpuGx single_term = overflow ? GpuGx::zero() : GpuGx::load_from_array(input, g_index);
        //acc = acc.mixed_add(single_term);

        acc = acc.mixed_add(single_terms[j]);
      }
    }

    combinations[i] = acc;
  }
}

template __device__ void multiexp_prepare_combinations<GpuG1, g1_knary_k, g1_knary_size>(fixnum *input, GpuG1 *combinations, long knary_chunk_size, long num);
template __device__ void multiexp_prepare_combinations<GpuMnt4G2, g2_mnt4_knary_k, g2_mnt4_knary_size>(fixnum *input, GpuMnt4G2 *combinations, long knary_chunk_size, long num);
template __device__ void multiexp_prepare_combinations<GpuMnt6G2, g2_mnt6_knary_k, g2_mnt6_knary_size>(fixnum *input, GpuMnt6G2 *combinations, long knary_chunk_size, long num);

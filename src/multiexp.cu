#include <cstdio>
#include <cstdint>
#include <libff/common/profiling.hpp>

#include "constants.hpp"
#include "multiexp.cuh"
#include "reduce.cuh"
#include "scale.cuh"
#include "prepare.cuh"
#include "utils.hpp"
#include "retrieve_utils.cuh"
#include "gpu_utils.cuh"

template <GEnum Gx, int knary_k>
void multiexp(bool debug, cudaStream_t stream, my_fixnum_array *gpu_s, my_fixnum_array *gpu_a, my_fixnum_array *gpu_o, size_t n)
{
  long knary_chunk_size = (n + knary_k - 1) / knary_k;

  if (debug)
  {
    cudaStreamSynchronize(stream);
  }

  libff::enter_block("Scaling");
  scale_batch_knary<Gx, knary_k>(stream, gpu_s, gpu_a, gpu_o, n);
  if (debug)
  {
    cudaStreamSynchronize(stream);
  }
  libff::leave_block("Scaling");

  libff::enter_block("Reduce");
  reduce_g<Gx>(stream, gpu_o, knary_chunk_size);
  if (debug)
  {
    cudaStreamSynchronize(stream);
  }
  libff::leave_block("Reduce");
}

template <GEnum Gx, int knary_k>
void multiexp_prepared(bool debug, cudaStream_t stream, my_fixnum_array *gpu_s, my_fixnum_array *gpu_a, my_fixnum_array *gpu_o, my_fixnum_array *gpu_p, size_t n)
{
  long knary_chunk_size = (n + knary_k - 1) / knary_k;

  if (debug)
  {
    cudaStreamSynchronize(stream);
  }
  libff::enter_block("Preparing");
  prepare<Gx>(stream, gpu_a, gpu_p, n);
  if (debug)
  {
    cudaStreamSynchronize(stream);
    cuda_check(cudaPeekAtLastError(), "Prepare");
  }
  libff::leave_block("Preparing");

  libff::enter_block("Scaling");
  scale_prepared_batched<Gx>(stream, gpu_s, gpu_o, gpu_p, n);
  if (debug)
  {
    cudaStreamSynchronize(stream);
    cuda_check(cudaPeekAtLastError(), "Scale");
  }
  libff::leave_block("Scaling");

  libff::enter_block("Reduce");
  reduce_g<Gx>(stream, gpu_o, knary_chunk_size);
  if (debug)
  {
    cudaStreamSynchronize(stream);
  }
  libff::leave_block("Reduce");
}

template void multiexp<G1_MNT, g1_knary_k>(bool debug, cudaStream_t stream, my_fixnum_array *w, my_fixnum_array *A, my_fixnum_array *gpu_o, size_t n);
template void multiexp<G2_MNT4, g2_mnt4_knary_k>(bool debug, cudaStream_t stream, my_fixnum_array *w, my_fixnum_array *A, my_fixnum_array *gpu_o, size_t n);
template void multiexp<G2_MNT6, g2_mnt6_knary_k>(bool debug, cudaStream_t stream, my_fixnum_array *w, my_fixnum_array *A, my_fixnum_array *gpu_o, size_t n);

template void multiexp_prepared<G1_MNT, g1_knary_k>(bool debug, cudaStream_t stream, my_fixnum_array *w, my_fixnum_array *A, my_fixnum_array *gpu_o, my_fixnum_array *gpu_p, size_t n);
template void multiexp_prepared<G2_MNT4, g2_mnt4_knary_k>(bool debug, cudaStream_t stream, my_fixnum_array *w, my_fixnum_array *A, my_fixnum_array *gpu_o, my_fixnum_array *gpu_p, size_t n);
template void multiexp_prepared<G2_MNT6, g2_mnt6_knary_k>(bool debug, cudaStream_t stream, my_fixnum_array *w, my_fixnum_array *A, my_fixnum_array *gpu_o, my_fixnum_array *gpu_p, size_t n);

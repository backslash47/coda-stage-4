#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <chrono>

#include <libff/common/profiling.hpp>

#include "groth16_params.hpp"
#include "gpu_groth16_params.cuh"
#include "groth16_input.hpp"
#include "gpu_groth16_input.cuh"
#include "groth16_output.hpp"
#include "multiexp.cuh"
#include "utils.hpp"
#include "io.hpp"
#include "params.hpp"
#include "constants.hpp"
#include "cpu_compute.hpp"
//#include "preprocess2.cuh"
#include "gpu_constants.cuh"
#include "retrieve_utils.cuh"

using namespace libff;

template <ECurve ecurve>
void preprocessStage4(FILE *input_params)
{
  Groth16Params<ecurve> params;
  params.load(input_params);
  printf("m: %ld, d: %ld\n", params.m, params.d);

  GpuGroth16Params<ecurve> gpuParams;
  gpuParams.load(&params);

  libff::enter_block("Init params");
  init_params<ecurve>();
  libff::leave_block("Init params");

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Preprocessor<ecurve> preprocessor;
  // preprocessor.init(stream, params, gpuParams);

  cudaPeekAtLastError();
  libff::enter_block("Waiting for GPU");
  cudaStreamSynchronize(stream);
  libff::leave_block("Waiting for GPU");
  cudaStreamDestroy(stream);
}

static inline auto now() -> decltype(std::chrono::high_resolution_clock::now())
{
  return std::chrono::high_resolution_clock::now();
}

template <typename T>
void print_time(T &t1, const char *str)
{
  auto t2 = std::chrono::high_resolution_clock::now();
  auto tim = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  printf("%s: %ld ms\n", str, tim);
  t1 = t2;
}

template <ECurve ecurve, GEnum G2x = ecurve == CURVE_MNT4 ? G2_MNT4 : G2_MNT6>
void calculateStage4(bool debug, FILE *input_params, FILE *inputs, FILE *outputs)
{
  bool is_mnt4 = ecurve == CURVE_MNT4;
  auto print_G2 = is_mnt4 ? print_mnt4_G2 : print_mnt6_G2;
  int g2_dimension = (is_mnt4 ? 6 : 9);

  size_t primary_input_size = 1;

  uint8_t *evaluation_At = new uint8_t[3 * bytes_per_elem];
  uint8_t *evaluation_Bt1 = new uint8_t[3 * bytes_per_elem];
  uint8_t *evaluation_Bt2 = new uint8_t[g2_dimension * bytes_per_elem];
  uint8_t *evaluation_Ht = new uint8_t[3 * bytes_per_elem];
  uint8_t *evaluation_Lt = new uint8_t[3 * bytes_per_elem];

  auto beginning = now();
  auto t = beginning;

  libff::enter_block("Load params");
  Groth16Params<ecurve> params;
  params.load(input_params);
  libff::leave_block("Load params");
  print_time(t, "load groth16 params to RAM");

  GpuGroth16Params<ecurve> gpuParams;
  gpuParams.load(&params);

  print_time(t, "load groth16 params to GPU");

  cudaStream_t streamB1, streamB2, streamL; //streamA
  //cudaStreamCreate(&streamA);
  cudaStreamCreate(&streamB1);
  cudaStreamCreate(&streamB2);
  cudaStreamCreate(&streamL);

  libff::enter_block("load preprocessed data to GPU");
  //Preprocessor<ecurve> preprocessor;
  //preprocessor.load(params);
  cudaDeviceSynchronize();
  libff::leave_block("load preprocessed data to GPU");

  libff::enter_block("Init params");
  init_params<ecurve>();
  libff::leave_block("Init params");

  print_time(t, "load curve params to GPU");

  auto t_main = t;

  libff::enter_block("Prepare output buffers");
  long output_size = (3 * (params.m + 1) + 3 * (params.m + 1) + g2_dimension * (params.m + 1) + 3 * (params.m - 1)) * bytes_per_elem;
  printf("Output buffers allocated: %ld MB\n", output_size / 1024 / 1024);

  //my_fixnum_array *outA = my_fixnum_array::create(3 * (params.m + 1));
  my_fixnum_array *outB1 = my_fixnum_array::create(3 * (params.m + 1));
  my_fixnum_array *outB2 = my_fixnum_array::create(g2_dimension * (params.m + 1));
  my_fixnum_array *outL = my_fixnum_array::create(3 * (params.m - 1));
  libff::leave_block("Prepare output buffers");

  // libff::enter_block("Prepare preprocess buffers");
  // long prepared_size = (3 * (params.m + 1) + 3 * (params.m + 1) + g2_dimension * (params.m + 1) + 3 * (params.m - 1)) * bytes_per_elem * knary_size;
  // printf("Preprocess buffers allocated: %ld MB\n", prepared_size / 1024 / 1024);
  // printf("Preprocess buffers allocated for B2: %ld MB\n", g2_dimension * (params.m + 1) * bytes_per_elem * knary_size / 1024 / 1024);

  // //my_fixnum_array *pA = my_fixnum_array::create(3 * (params.m + 1) * knary_size);
  // //my_fixnum_array *pB1 = my_fixnum_array::create(3 * (params.m + 1) * knary_size);
  // my_fixnum_array *pB2 = my_fixnum_array::create(g2_dimension * (params.m + 1) * knary_size);
  // //my_fixnum_array *pL = my_fixnum_array::create(3 * (params.m - 1) * knary_size);
  // libff::leave_block("Prepare preprocess buffers");

  libff::enter_block("Load input");
  Groth16Input<ecurve> input;
  input.load(inputs, params.d, params.m);
  libff::leave_block("Load input");

  GpuGroth16Input<ecurve> gpuInput;
  gpuInput.load(&input, params.m);

  CpuCompute<ecurve> compute;

  libff::enter_block("G2 Multiexp B2");
  if (ecurve == CURVE_MNT4)
  {
    multiexp<G2x, g2_mnt4_knary_k>(debug, streamB2, gpuInput.w, gpuParams.B2, outB2, params.m + 1);
  }
  else
  {
    multiexp<G2x, g2_mnt6_knary_k>(debug, streamB2, gpuInput.w, gpuParams.B2, outB2, params.m + 1);
  }
  //compute.computeMultiExpG2(evaluation_Bt2, input.w, params.B2, params.m + 1);
  // multiexp_prepared<G2x>(debug, streamB2, gpuInput.w, gpuParams.B2, outB2, pB2, params.m + 1);
  libff::leave_block("G2 Multiexp B2");

  libff::enter_block("G1 Multiexp B1");
  multiexp<G1_MNT, g1_knary_k>(debug, streamB1, gpuInput.w, gpuParams.B1, outB1, params.m + 1);
  //multiexp_prepared<G1_MNT>(debug, streamB1, gpuInput.w, gpuParams.B1, outB1, pB1, params.m + 1);
  libff::leave_block("G1 Multiexp B1");

  libff::enter_block("G1 Multiexp L");
  multiexp<G1_MNT, g1_knary_k>(debug, streamL, gpuInput.offseted(primary_input_size + 1), gpuParams.L, outL, params.m - 1);
  //multiexp_prepared<G1_MNT>(debug, streamL, gpuInput.offseted(primary_input_size + 1), gpuParams.L, outL, pL, params.m - 1);
  // compute.computeMultiExp(evaluation_Lt, input.w + (primary_input_size + 1) * bytes_per_elem, params.L, params.m - 1);
  libff::leave_block("G1 Multiexp L");

  libff::enter_block("G1 Multiexp A");
  //multiexp<G1_MNT, g1_knary_k>(debug, streamA, gpuInput.w, gpuParams.A, outA, params.m + 1);
  //multiexp_prepared<G1_MNT>(debug, streamA, gpuInput.w, gpuParams.A, outA, pA, params.m + 1);
  compute.computeMultiExp(evaluation_At, input.w, params.A, params.m + 1);
  libff::leave_block("G1 Multiexp A");

  libff::enter_block("Compute the polynomial H");
  uint8_t *coefficients_for_H = compute.computeH(params.d, input.ca, input.cb, input.cc);
  libff::leave_block("Compute the polynomial H");

  libff::enter_block("G1 Multiexp H");
  // my_fixnum_array *gpu_coefficients_for_H = my_fixnum_array::create(coefficients_for_H, bytes_per_elem * params.d, bytes_per_elem);
  //multiexp<G1_MNT>(stream, gpu_coefficients_for_H, gpuParams.H, params.d);
  compute.computeMultiExp(evaluation_Ht, coefficients_for_H, params.H, params.d);
  libff::leave_block("G1 Multiexp H");

  cudaPeekAtLastError();
  libff::enter_block("Waiting for GPU");
  //cudaStreamSynchronize(streamA);
  cudaStreamSynchronize(streamB1);
  cudaStreamSynchronize(streamL);
  cudaStreamSynchronize(streamB2);
  libff::leave_block("Waiting for GPU");
  //cudaStreamDestroy(streamA);
  cudaStreamDestroy(streamB1);
  cudaStreamDestroy(streamB2);
  cudaStreamDestroy(streamL);

  //get_1D_fixnum_array(evaluation_At, outA, 3);
  g_to_affine_inplace<ecurve, G1_MNT>(evaluation_At);

  get_1D_fixnum_array(evaluation_Bt1, outB1, 3);
  g_to_affine_inplace<ecurve, G1_MNT>(evaluation_Bt1);

  get_1D_fixnum_array(evaluation_Bt2, outB2, g2_dimension);
  g_to_affine_inplace<ecurve, G2_MNT>(evaluation_Bt2);

  // get_1D_fixnum_array(evaluation_Ht, gpuParams.H, 3);
  g_to_affine_inplace<ecurve, G1_MNT>(evaluation_Ht);

  get_1D_fixnum_array(evaluation_Lt, outL, 3);
  g_to_affine_inplace<ecurve, G1_MNT>(evaluation_Lt);

  printf("evaluation_At: \n");
  printG1(evaluation_At);
  printf("----------------------\n");
  printf("evaluation_Bt1: \n");
  printG1(evaluation_Bt1);
  printf("----------------------\n");
  printf("evaluation_Bt2: \n");
  print_G2(evaluation_Bt2);
  printf("----------------------\n");
  printf("evaluation_Ht: \n");
  printG1(evaluation_Ht);
  printf("----------------------\n");
  printf("evaluation_Lt: \n");
  printG1(evaluation_Lt);
  printf("----------------------\n");

  gpuParams.destroy();
  gpuInput.destroy();
  delete[] coefficients_for_H;
  // delete gpu_coefficients_for_H;

  //delete outA;
  delete outB1;
  delete outB2;
  delete outL;

  //delete pA;
  //delete pB1;
  //delete pB2;
  //delete pL;

  // preprocessor.close();

  libff::enter_block("Computing C");
  uint8_t *C = compute.computeC(evaluation_Ht, evaluation_Bt1, evaluation_Lt, input.r);
  libff::leave_block("Computing C");

  print_time(t, "calculation cpu + gpu");

  groth16_output<ecurve> output;
  output.write(evaluation_At, evaluation_Bt2, C, outputs);
  print_time(t, "store");

  delete[] evaluation_At;
  delete[] evaluation_Bt1;
  delete[] evaluation_Bt2;
  delete[] evaluation_Ht;
  delete[] evaluation_Lt;
  delete[] C;

  print_time(t_main, "Total time from input to output: ");
}

void stage_4(const char *curve, const char *mode, FILE *input_params, FILE *inputs, FILE *outputs)
{
  if (strcmp(curve, "MNT4753") == 0)
  {
    if (strcmp(mode, "compute") == 0)
    {
      calculateStage4<CURVE_MNT4>(false, input_params, inputs, outputs);
      fclose(input_params);
      fclose(inputs);
      fclose(outputs);
    }
    else if (strcmp(mode, "preprocess") == 0)
    {
      preprocessStage4<CURVE_MNT4>(input_params);
      fclose(input_params);
    }
  }
  else if (strcmp(curve, "MNT6753") == 0)
  {
    if (strcmp(mode, "compute") == 0)
    {
      calculateStage4<CURVE_MNT6>(false, input_params, inputs, outputs);
      fclose(input_params);
      fclose(inputs);
      fclose(outputs);
    }
    else if (strcmp(mode, "preprocess") == 0)
    {
      preprocessStage4<CURVE_MNT6>(input_params);
      fclose(input_params);
    }
  }
}
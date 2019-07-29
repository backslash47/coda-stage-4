#include "gpu_groth16_params.cuh"
#include "constants.hpp"
#include "utils.hpp"

template<ECurve curve>
void GpuGroth16Params<curve>::load(Groth16Params<curve>* params) {
  GEnum G2x = curve == CURVE_MNT4 ? G2_MNT4 : G2_MNT6;

  long m = params->m;
  long d = params->d;
  
  this->A = my_fixnum_array::create(params->A, getDimension(G1_MNT) * bytes_per_elem * (m + 1), bytes_per_elem);
  this->B1 = my_fixnum_array::create(params->B1, getDimension(G1_MNT) * bytes_per_elem * (m + 1), bytes_per_elem);
  this->B2 = my_fixnum_array::create(params->B2, getDimension(G2x) * bytes_per_elem * (m + 1), bytes_per_elem);
  this->H = my_fixnum_array::create(params->H, getDimension(G1_MNT) * bytes_per_elem * d, bytes_per_elem);
  this->L = my_fixnum_array::create(params->L, getDimension(G1_MNT) * bytes_per_elem * (m - 1), bytes_per_elem);
}

template<ECurve curve>
void GpuGroth16Params<curve>::destroy() {
  delete this->A;
  delete this->B1;
  delete this->B2;
  delete this->H;
  delete this->L;
}

template class GpuGroth16Params<CURVE_MNT4>;
template class GpuGroth16Params<CURVE_MNT6>;
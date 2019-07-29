#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

#include "constants.hpp"
#include "groth16_params.hpp"
#include "io.hpp"

using namespace libff;

template <ECurve curve>
void Groth16Params<curve>::load(FILE *params)
{
  bool is_mnt4 = curve == CURVE_MNT4;
  int b2_dimension = is_mnt4 ? 6 : 9;

  auto read_g1_montgomery_zero = is_mnt4 ? read_mnt4_g1_montgomery_zero : read_mnt6_g1_montgomery_zero;
  auto read_g2_montgomery_zero = is_mnt4 ? read_mnt4_g2_montgomery_zero : read_mnt6_g2_montgomery_zero;

  this->d = read_size_t(params);
  this->m = read_size_t(params);

  this->A = new uint8_t[3 * (this->m + 1) * bytes_per_elem];
  this->B1 = new uint8_t[3 * (this->m + 1) * bytes_per_elem];
  this->B2 = new uint8_t[b2_dimension * (this->m + 1) * bytes_per_elem];
  this->L = new uint8_t[3 * (this->m - 1) * bytes_per_elem];
  this->H = new uint8_t[3 * d * bytes_per_elem];

  memset(this->A, 0, 3 * (this->m + 1) * bytes_per_elem);
  memset(this->B1, 0, 3 * (this->m + 1) * bytes_per_elem);
  memset(this->B2, 0, b2_dimension * (this->m + 1) * bytes_per_elem);
  memset(this->L, 0, 3 * (this->m - 1) * bytes_per_elem);
  memset(this->H, 0, 3 * d * bytes_per_elem);

  for (size_t i = 0; i < this->m + 1; ++i)
  {
    read_g1_montgomery_zero(this->A + 3 * i * bytes_per_elem, params);
  }

  for (size_t i = 0; i < this->m + 1; ++i)
  {
    read_g1_montgomery_zero(this->B1 + 3 * i * bytes_per_elem, params);
  }

  for (size_t i = 0; i < this->m + 1; ++i)
  {
    read_g2_montgomery_zero(this->B2 + b2_dimension * i * bytes_per_elem, params);
  }

  for (size_t i = 0; i < this->m - 1; ++i)
  {
    read_g1_montgomery_zero(this->L + 3 * i * bytes_per_elem, params);
  }

  for (size_t i = 0; i < d; ++i)
  {
    read_g1_montgomery_zero(this->H + 3 * i * bytes_per_elem, params);
  }
}

template class Groth16Params<CURVE_MNT4>;
template class Groth16Params<CURVE_MNT6>;
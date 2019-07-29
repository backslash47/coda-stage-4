#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

#include "constants.hpp"
#include "groth16_input.hpp"
#include "io.hpp"

using namespace libff;

template <ECurve curve>
void Groth16Input<curve>::load(FILE *inputs, size_t d, size_t m)
{
  bool is_mnt4 = curve == CURVE_MNT4;
  auto read_fr_montgomery = is_mnt4 ? read_mnt4_fr_montgomery : read_mnt6_fr_montgomery;

  this->w = new uint8_t[(m + 1) * bytes_per_elem];
  this->ca = new uint8_t[(d + 1) * bytes_per_elem];
  this->cb = new uint8_t[(d + 1) * bytes_per_elem];
  this->cc = new uint8_t[(d + 1) * bytes_per_elem];
  this->r = new uint8_t[bytes_per_elem];

  memset(this->w, 0, (m + 1) * bytes_per_elem);
  memset(this->ca, 0, (d + 1) * bytes_per_elem);
  memset(this->cb, 0, (d + 1) * bytes_per_elem);
  memset(this->cc, 0, (d + 1) * bytes_per_elem);
  memset(this->r, 0, bytes_per_elem);

  for (size_t i = 0; i < m + 1; ++i)
  {
    read_fr_montgomery(this->w + i * bytes_per_elem, inputs);
  }

  for (size_t i = 0; i < d + 1; ++i)
  {
    read_fr_montgomery(this->ca + i * bytes_per_elem, inputs);
  }

  for (size_t i = 0; i < d + 1; ++i)
  {
    read_fr_montgomery(this->cb + i * bytes_per_elem, inputs);
  }

  for (size_t i = 0; i < d + 1; ++i)
  {
    read_fr_montgomery(this->cc + i * bytes_per_elem, inputs);
  }

  read_fr_montgomery(this->r, inputs);
}

template class Groth16Input<CURVE_MNT4>;
template class Groth16Input<CURVE_MNT6>;
#include "gpu_groth16_input.cuh"
#include "constants.hpp"
#include "utils.hpp"

template<ECurve curve>
void GpuGroth16Input<curve>::load(Groth16Input<curve>* input, long m) {
  this->w = my_fixnum_array::create(input->w, bytes_per_elem * (m + 1), bytes_per_elem);
}

template<ECurve curve>
void GpuGroth16Input<curve>::destroy() {
  delete this->w;
}

template<ECurve curve>
my_fixnum_array* GpuGroth16Input<curve>::offseted(int offset) {
  return my_fixnum_array::wrap(this->w->get_ptr() + offset * bytes_per_elem, this->w->length() - offset);
}

template class GpuGroth16Input<CURVE_MNT4>;
template class GpuGroth16Input<CURVE_MNT6>;
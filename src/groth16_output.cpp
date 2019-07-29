#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

#include "groth16_output.hpp"
#include "io.hpp"

using namespace libff;

template <ECurve ecurve>
void groth16_output<ecurve>::write(uint8_t *evaluation_At, uint8_t *evaluation_Bt2, uint8_t *C, FILE *output)
{
  if (ecurve == CURVE_MNT4)
  {
    write_mnt4_g1_montgomery(evaluation_At, output);
    write_mnt4_g2_montgomery(evaluation_Bt2, output);
    write_mnt4_g1_montgomery(C, output);
  }
  else
  {
    write_mnt6_g1_montgomery(evaluation_At, output);
    write_mnt6_g2_montgomery(evaluation_Bt2, output);
    write_mnt6_g1_montgomery(C, output);
  }
}

template class groth16_output<CURVE_MNT4>;
template class groth16_output<CURVE_MNT6>;
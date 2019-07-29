#include "params.hpp"
#include "utils.hpp"

#include "gpu_constants.cuh"
#include "gpu_params.cuh"

void init_params_mnt4()
{
  HostParams params;
  params.set_mnt_mod((fixnum *)mnt4_modulus);
  params.set_mnt_non_residue((fixnum *)mnt4_non_residue);
  params.set_mnt_coeff_a((fixnum *)mnt4_g1_coeff_a);
  params.set_mnt_coeff_a2((fixnum *)mnt4_g2_coeff_a2_c0, (fixnum *)mnt4_g2_coeff_a2_c1);
  params.set_twist_mul_by_a2((fixnum *)mnt4_g2_twist_mul_by_a_c0, (fixnum *)mnt4_g2_twist_mul_by_a_c1);

  set_host_params(params);
}

void init_params_mnt6()
{
  HostParams params;
  params.set_mnt_mod((fixnum *)mnt6_modulus);
  params.set_mnt_non_residue((fixnum *)mnt6_non_residue);
  params.set_mnt_coeff_a((fixnum *)mnt6_g1_coeff_a);
  params.set_mnt_coeff_a3((fixnum *)mnt6_g2_coeff_a3_c0, (fixnum *)mnt6_g2_coeff_a3_c1, (fixnum *)mnt6_g2_coeff_a3_c2);
  params.set_twist_mul_by_a2((fixnum *)mnt6_g2_twist_mul_by_a_c0, (fixnum *)mnt6_g2_twist_mul_by_a_c1, (fixnum *)mnt6_g2_twist_mul_by_a_c2);

  set_host_params(params);
}

template <ECurve ecurve>
void init_params()
{
  bool is_mnt4 = ecurve == CURVE_MNT4;

  if (is_mnt4)
  {
    init_params_mnt4();
  }
  else
  {
    init_params_mnt6();
  }
}

template void init_params<CURVE_MNT4>();
template void init_params<CURVE_MNT6>();
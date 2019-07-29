#include <libfqfft/evaluation_domain/get_evaluation_domain.hpp>
#include <libff/algebra/curves/public_params.hpp>
#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>
#include <libff/algebra/scalar_multiplication/multiexp.hpp>

#include "cpu_compute.hpp"
#include "constants.hpp"
#include "libff_convert.hpp"

using namespace libff;
using namespace libfqfft;
using namespace std;

template <typename curve>
void vectorFrMuleq(std::vector<Fr<curve>> &a, std::vector<Fr<curve>> &b, size_t size)
{
#ifdef MULTICORE
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; i++)
  {
    a.at(i) = a.at(i) * b.at(i);
  }
}

template <typename curve>
void vectorFrSubeq(std::vector<Fr<curve>> &a, std::vector<Fr<curve>> &b, size_t size)
{
#ifdef MULTICORE
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; i++)
  {
    a.at(i) = a.at(i) - b.at(i);
  }
}

template <ECurve ecurve>
uint8_t *CpuCompute<ecurve>::computeH(size_t d, uint8_t *ca_raw, uint8_t *cb_raw, uint8_t *cc_raw)
{
  typedef typename std::conditional<ecurve == CURVE_MNT4, mnt4753_pp, mnt6753_pp>::type curve;

  Converter<curve> converter;
  shared_ptr<evaluation_domain<Fr<curve>>> domain = get_evaluation_domain<Fr<curve>>(d);

  std::vector<Fr<curve>> ca = converter.convertRawToFrArray(ca_raw, d + 1);
  std::vector<Fr<curve>> cb = converter.convertRawToFrArray(cb_raw, d + 1);
  std::vector<Fr<curve>> cc = converter.convertRawToFrArray(cc_raw, d + 1);

  domain->iFFT(ca);
  domain->iFFT(cb);

  domain->cosetFFT(ca, Fr<curve>::multiplicative_generator);
  domain->cosetFFT(cb, Fr<curve>::multiplicative_generator);

  // Use ca to store H
  auto H_tmp = ca;

  size_t m = domain->m;

  vectorFrMuleq<curve>(H_tmp, cb, m);

  domain->iFFT(cc);
  domain->cosetFFT(cc, Fr<curve>::multiplicative_generator);

  m = domain->m;

  vectorFrSubeq<curve>(H_tmp, cc, m);

  domain->divide_by_Z_on_coset(H_tmp);

  domain->icosetFFT(H_tmp, Fr<curve>::multiplicative_generator);

  m = domain->m;

  std::vector<Fr<curve>> H_res(m + 1, Fr<curve>::zero());
  std::copy(H_tmp.begin(), H_tmp.begin() + m, H_res.begin());

  return converter.convertFrToRawArray(H_res);
}

template <ECurve ecurve>
uint8_t *CpuCompute<ecurve>::computeC(uint8_t *evaluation_Ht_raw, uint8_t *evaluation_Bt1_raw, uint8_t *evaluation_Lt_raw, uint8_t *r_raw)
{
  typedef typename std::conditional<ecurve == CURVE_MNT4, mnt4753_pp, mnt6753_pp>::type curve;

  Converter<curve> converter;

  G1<curve> evaluation_Ht = converter.convertRawToG1(evaluation_Ht_raw, true);
  G1<curve> evaluation_Bt1 = converter.convertRawToG1(evaluation_Bt1_raw, true);
  G1<curve> evaluation_Lt = converter.convertRawToG1(evaluation_Lt_raw, true);
  Fr<curve> r = converter.convertRawToFr(r_raw);

  G1<curve> scaled_Bt1 = r * evaluation_Bt1;
  G1<curve> Lt1_plus_scaled_Bt1 = evaluation_Lt + scaled_Bt1;
  G1<curve> C = evaluation_Ht + Lt1_plus_scaled_Bt1;

  C.to_affine_coordinates();

  uint8_t *C_raw = new uint8_t[3 * bytes_per_elem];
  converter.convertG1ToRaw(C, C_raw, true);
  return C_raw;
}

template <ECurve ecurve>
void CpuCompute<ecurve>::computeMultiExp(uint8_t *res, uint8_t *s, uint8_t *g, long n)
{
  typedef typename std::conditional<ecurve == CURVE_MNT4, mnt4753_pp, mnt6753_pp>::type curve;

#ifdef MULTICORE
  const size_t chunks =
      omp_get_max_threads(); // to override, set OMP_NUM_THREADS env var or call
                             // omp_set_num_threads()
#else
  const size_t chunks = 1;
#endif

  Converter<curve> converter;
  typename Converter<curve>::vectorFr sVector = converter.convertRawToFrArray(s, n);
  typename Converter<curve>::vectorG1 gVector = converter.convertRawToG1Array(g, n, true);

  G1<curve> result = multi_exp_with_mixed_addition<G1<curve>, Fr<curve>, multi_exp_method_BDLO12>(
      gVector.begin(), gVector.begin() + n, sVector.begin(), sVector.begin() + n, chunks);

  converter.convertG1ToRaw(result, res, true);
}

template <ECurve ecurve>
void CpuCompute<ecurve>::computeMultiExpG2(uint8_t *res, uint8_t *s, uint8_t *g, long n)
{
  typedef typename std::conditional<ecurve == CURVE_MNT4, mnt4753_pp, mnt6753_pp>::type curve;

#ifdef MULTICORE
  const size_t chunks =
      omp_get_max_threads(); // to override, set OMP_NUM_THREADS env var or call
                             // omp_set_num_threads()
#else
  const size_t chunks = 1;
#endif

  if (ecurve == CURVE_MNT4)
  {
    Converter<mnt4753_pp> converter;
    typename Converter<mnt4753_pp>::vectorFr sVector = converter.convertRawToFrArray(s, n);
    typename Converter<mnt4753_pp>::vectorMnt4G2 gVector = converter.convertRawToMnt4G2Array(g, n, true);

    G2<mnt4753_pp> result = multi_exp_with_mixed_addition<G2<mnt4753_pp>, Fr<mnt4753_pp>, multi_exp_method_BDLO12>(
        gVector.begin(), gVector.begin() + n, sVector.begin(), sVector.begin() + n, chunks);

    converter.convertMnt4G2ToRaw(result, res, true);
  }
  else
  {
    Converter<mnt6753_pp> converter;
    typename Converter<mnt6753_pp>::vectorFr sVector = converter.convertRawToFrArray(s, n);
    typename Converter<mnt6753_pp>::vectorMnt6G2 gVector = converter.convertRawToMnt6G2Array(g, n, true);

    G2<mnt6753_pp> result = multi_exp_with_mixed_addition<G2<mnt6753_pp>, Fr<mnt6753_pp>, multi_exp_method_BDLO12>(
        gVector.begin(), gVector.begin() + n, sVector.begin(), sVector.begin() + n, chunks);

    converter.convertMnt6G2ToRaw(result, res, true);
  }
}

template class CpuCompute<CURVE_MNT4>;
template class CpuCompute<CURVE_MNT6>;
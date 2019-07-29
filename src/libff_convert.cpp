#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

#include "libff_convert.hpp"
#include "constants.hpp"
#include "io.hpp"

using namespace libff;
using namespace std;

template <typename curve>
Fr<curve> Converter<curve>::convertRawToFr(uint8_t *src)
{
  Fr<curve> x;
  memcpy((void *)x.mont_repr.data, src, io_bytes_per_elem);
  return Fr<curve>(x.mont_repr);
}

template <typename curve>
std::vector<Fr<curve>> Converter<curve>::convertRawToFrArray(uint8_t *src, size_t num)
{
  std::vector<Fr<curve>> result;

  for (int i = 0; i < num; i++)
  {
    result.emplace_back(convertRawToFr(src + i * bytes_per_elem));
  }

  return result;
}

template <typename curve>
void Converter<curve>::convertFrToRaw(Fr<curve> &src, uint8_t *dst)
{
  memcpy(dst, (void *)src.as_bigint().data, io_bytes_per_elem);
}

template <typename curve>
uint8_t *Converter<curve>::convertFrToRawArray(vectorFr &src)
{
  size_t num = src.size();
  uint8_t *dst = new uint8_t[num * bytes_per_elem];
  memset(dst, 0, num * bytes_per_elem);

  for (int i = 0; i < num; i++)
  {
    convertFrToRaw(src.at(i), dst + i * bytes_per_elem);
  }

  return dst;
}

template <typename curve>
G1<curve> Converter<curve>::convertRawToG1(uint8_t *src, bool full_size)
{
  G1<curve> g1;
  Fq<curve> X, Y, Z;

  int member_size = full_size ? bytes_per_elem : io_bytes_per_elem;

  memcpy((void *)X.mont_repr.data, src + 0 * member_size, io_bytes_per_elem); // libff always contains only io_bytes_per_elem bytes
  memcpy((void *)Y.mont_repr.data, src + 1 * member_size, io_bytes_per_elem);
  memcpy((void *)Z.mont_repr.data, src + 2 * member_size, io_bytes_per_elem);

  return G1<curve>(Fq<curve>(X.mont_repr), Fq<curve>(Y.mont_repr), Fq<curve>(Z.mont_repr));
}

template <typename curve>
std::vector<G1<curve>> Converter<curve>::convertRawToG1Array(uint8_t *src, size_t num, bool full_size)
{
  std::vector<G1<curve>> result;

  int member_size = full_size ? bytes_per_elem : io_bytes_per_elem;

  for (int i = 0; i < num; i++)
  {
    result.emplace_back(convertRawToG1(src + i * 3 * member_size, full_size));
  }

  return result;
}

template <typename curve>
void Converter<curve>::convertG1ToRaw(G1<curve> &src, uint8_t *dst, bool full_size)
{
  int member_size = full_size ? bytes_per_elem : io_bytes_per_elem;

  memcpy(dst + 0 * member_size, (void *)src.X().as_bigint().data, io_bytes_per_elem); // libff always contains only io_bytes_per_elem bytes
  memcpy(dst + 1 * member_size, (void *)src.Y().as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 2 * member_size, (void *)src.Z().as_bigint().data, io_bytes_per_elem);
}

template <typename curve>
uint8_t *Converter<curve>::convertG1ToRawArray(vectorG1 &src, bool full_size)
{
  int member_size = full_size ? bytes_per_elem : io_bytes_per_elem;

  size_t num = src.size();
  uint8_t *dst = new uint8_t[3 * num * member_size];
  memset(dst, 0, 3 * num * member_size);

  for (int i = 0; i < num; i++)
  {
    convertG1ToRaw(src.at(i), dst + 3 * i * member_size, full_size);
  }

  return dst;
}

template <typename curve>
G2<mnt4753_pp> Converter<curve>::convertRawToMnt4G2(uint8_t *src, bool full_size)
{
  G2<mnt4753_pp> g2;
  Fq<mnt4753_pp> Xc0, Xc1, Yc0, Yc1, Zc0, Zc1;

  int member_size = full_size ? bytes_per_elem : io_bytes_per_elem;

  memcpy((void *)Xc0.mont_repr.data, src + 0 * member_size, io_bytes_per_elem); // libff always contains only io_bytes_per_elem bytes
  memcpy((void *)Xc1.mont_repr.data, src + 1 * member_size, io_bytes_per_elem);
  memcpy((void *)Yc0.mont_repr.data, src + 2 * member_size, io_bytes_per_elem);
  memcpy((void *)Yc1.mont_repr.data, src + 3 * member_size, io_bytes_per_elem);
  memcpy((void *)Zc0.mont_repr.data, src + 4 * member_size, io_bytes_per_elem);
  memcpy((void *)Zc1.mont_repr.data, src + 5 * member_size, io_bytes_per_elem);

  return G2<mnt4753_pp>(Fqe<mnt4753_pp>(Fq<mnt4753_pp>(Xc0.mont_repr), Fq<mnt4753_pp>(Xc1.mont_repr)), Fqe<mnt4753_pp>(Fq<mnt4753_pp>(Yc0.mont_repr), Fq<mnt4753_pp>(Yc1.mont_repr)), Fqe<mnt4753_pp>(Fq<mnt4753_pp>(Zc0.mont_repr), Fq<mnt4753_pp>(Zc1.mont_repr)));
}

template <typename curve>
std::vector<G2<mnt4753_pp>> Converter<curve>::convertRawToMnt4G2Array(uint8_t *src, size_t num, bool full_size)
{
  std::vector<G2<mnt4753_pp>> result;

  int member_size = full_size ? bytes_per_elem : io_bytes_per_elem;

  for (int i = 0; i < num; i++)
  {
    result.emplace_back(convertRawToMnt4G2(src + i * 6 * member_size, full_size));
  }

  return result;
}

template <typename curve>
void Converter<curve>::convertMnt4G2ToRaw(G2<mnt4753_pp> &src, uint8_t *dst, bool full_size)
{
  int member_size = full_size ? bytes_per_elem : io_bytes_per_elem;

  memcpy(dst + 0 * member_size, (void *)src.X().c0.as_bigint().data, io_bytes_per_elem); // libff always contains only io_bytes_per_elem bytes
  memcpy(dst + 1 * member_size, (void *)src.X().c1.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 2 * member_size, (void *)src.Y().c0.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 3 * member_size, (void *)src.Y().c1.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 4 * member_size, (void *)src.Z().c0.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 5 * member_size, (void *)src.Z().c1.as_bigint().data, io_bytes_per_elem);
}

template <typename curve>
uint8_t *Converter<curve>::convertMnt4G2ToRawArray(vectorMnt4G2 &src, bool full_size)
{
  int member_size = full_size ? bytes_per_elem : io_bytes_per_elem;

  size_t num = src.size();
  uint8_t *dst = new uint8_t[6 * num * member_size];
  memset(dst, 0, 6 * num * member_size);

  for (int i = 0; i < num; i++)
  {
    convertMnt4G2ToRaw(src.at(i), dst + 6 * i * member_size, full_size);
  }

  return dst;
}

template <typename curve>
G2<mnt6753_pp> Converter<curve>::convertRawToMnt6G2(uint8_t *src, bool full_size)
{
  G2<mnt6753_pp> g2;
  Fq<mnt6753_pp> Xc0, Xc1, Xc2, Yc0, Yc1, Yc2, Zc0, Zc1, Zc2;

  int member_size = full_size ? bytes_per_elem : io_bytes_per_elem;

  memcpy((void *)Xc0.mont_repr.data, src + 0 * member_size, io_bytes_per_elem); // libff always contains only io_bytes_per_elem bytes
  memcpy((void *)Xc1.mont_repr.data, src + 1 * member_size, io_bytes_per_elem);
  memcpy((void *)Xc2.mont_repr.data, src + 2 * member_size, io_bytes_per_elem);
  memcpy((void *)Yc0.mont_repr.data, src + 3 * member_size, io_bytes_per_elem);
  memcpy((void *)Yc1.mont_repr.data, src + 4 * member_size, io_bytes_per_elem);
  memcpy((void *)Yc2.mont_repr.data, src + 5 * member_size, io_bytes_per_elem);
  memcpy((void *)Zc0.mont_repr.data, src + 6 * member_size, io_bytes_per_elem);
  memcpy((void *)Zc1.mont_repr.data, src + 7 * member_size, io_bytes_per_elem);
  memcpy((void *)Zc2.mont_repr.data, src + 8 * member_size, io_bytes_per_elem);

  return G2<mnt6753_pp>(Fqe<mnt6753_pp>(Fq<mnt6753_pp>(Xc0.mont_repr), Fq<mnt6753_pp>(Xc1.mont_repr), Fq<mnt6753_pp>(Xc2.mont_repr)), Fqe<mnt6753_pp>(Fq<mnt6753_pp>(Yc0.mont_repr), Fq<mnt6753_pp>(Yc1.mont_repr), Fq<mnt6753_pp>(Yc2.mont_repr)), Fqe<mnt6753_pp>(Fq<mnt6753_pp>(Zc0.mont_repr), Fq<mnt6753_pp>(Zc1.mont_repr), Fq<mnt6753_pp>(Zc2.mont_repr)));
}

template <typename curve>
std::vector<G2<mnt6753_pp>> Converter<curve>::convertRawToMnt6G2Array(uint8_t *src, size_t num, bool full_size)
{
  std::vector<G2<mnt6753_pp>> result;

  int member_size = full_size ? bytes_per_elem : io_bytes_per_elem;

  for (int i = 0; i < num; i++)
  {
    result.emplace_back(convertRawToMnt6G2(src + i * 9 * member_size, full_size));
  }

  return result;
}

template <typename curve>
void Converter<curve>::convertMnt6G2ToRaw(G2<mnt6753_pp> &src, uint8_t *dst, bool full_size)
{
  int member_size = full_size ? bytes_per_elem : io_bytes_per_elem;

  memcpy(dst + 0 * member_size, (void *)src.X().c0.as_bigint().data, io_bytes_per_elem); // libff always contains only io_bytes_per_elem bytes
  memcpy(dst + 1 * member_size, (void *)src.X().c1.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 2 * member_size, (void *)src.X().c2.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 3 * member_size, (void *)src.Y().c0.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 4 * member_size, (void *)src.Y().c1.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 5 * member_size, (void *)src.Y().c2.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 6 * member_size, (void *)src.Z().c0.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 7 * member_size, (void *)src.Z().c1.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 8 * member_size, (void *)src.Z().c2.as_bigint().data, io_bytes_per_elem);
}

template <typename curve>
uint8_t *Converter<curve>::convertMnt6G2ToRawArray(vectorMnt6G2 &src, bool full_size)
{
  int member_size = full_size ? bytes_per_elem : io_bytes_per_elem;

  size_t num = src.size();
  uint8_t *dst = new uint8_t[9 * num * member_size];
  memset(dst, 0, 9 * num * member_size);

  for (int i = 0; i < num; i++)
  {
    convertMnt6G2ToRaw(src.at(i), dst + 9 * i * member_size, full_size);
  }

  return dst;
}

template <ECurve curve>
void ConverterNew<curve>::affineG1(uint8_t *src)
{
  g_to_affine_inplace<curve, G1_MNT>(src);
}

template <ECurve curve>
void ConverterNew<curve>::affineG2(uint8_t *src)
{
  g_to_affine_inplace<curve, G2_MNT>(src);
}

template <ECurve curve>
void ConverterNew<curve>::affineG1Array(uint8_t *src, long num)
{
  long offset = 3 * bytes_per_elem;

  for (int i = 0; i < num; i++)
  {
    affineG1(src + i * offset);
  }
}

template <ECurve curve>
void ConverterNew<curve>::affineG2Array(uint8_t *src, long num)
{
  int dimension = curve == CURVE_MNT4 ? 6 : 9;
  long offset = dimension * bytes_per_elem;

  for (int i = 0; i < num; i++)
  {
    affineG2(src + i * offset);
  }
}

template class Converter<mnt4753_pp>;
template class Converter<mnt6753_pp>;

template class ConverterNew<CURVE_MNT4>;
template class ConverterNew<CURVE_MNT6>;
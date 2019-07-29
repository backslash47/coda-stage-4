#include <libff/algebra/curves/public_params.hpp>
#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

#include "constants.hpp"

using namespace libff;
using namespace std;

// FIXME: libff

template <typename curve>
class Converter
{
public:
  typedef std::vector<Fr<curve>> vectorFr;
  typedef std::vector<G1<curve>> vectorG1;
  typedef std::vector<G2<mnt4753_pp>> vectorMnt4G2;
  typedef std::vector<G2<mnt6753_pp>> vectorMnt6G2;

public:
  Fr<curve> convertRawToFr(uint8_t *src);
  vectorFr convertRawToFrArray(uint8_t *src, size_t num);

  void convertFrToRaw(Fr<curve> &src, uint8_t *dst);
  uint8_t *convertFrToRawArray(vectorFr &src);

  G1<curve> convertRawToG1(uint8_t *src, bool full_size);
  vectorG1 convertRawToG1Array(uint8_t *src, size_t num, bool full_size);
  void convertG1ToRaw(G1<curve> &src, uint8_t *dst, bool full_size);
  uint8_t *convertG1ToRawArray(vectorG1 &src, bool full_size);

  G2<mnt4753_pp> convertRawToMnt4G2(uint8_t *src, bool full_size);
  vectorMnt4G2 convertRawToMnt4G2Array(uint8_t *src, size_t num, bool full_size);
  void convertMnt4G2ToRaw(G2<mnt4753_pp> &src, uint8_t *dst, bool full_size);
  uint8_t *convertMnt4G2ToRawArray(vectorMnt4G2 &src, bool full_size);

  G2<mnt6753_pp> convertRawToMnt6G2(uint8_t *src, bool full_size);
  vectorMnt6G2 convertRawToMnt6G2Array(uint8_t *src, size_t num, bool full_size);
  void convertMnt6G2ToRaw(G2<mnt6753_pp> &src, uint8_t *dst, bool full_size);
  uint8_t *convertMnt6G2ToRawArray(vectorMnt6G2 &src, bool full_size);
};

template <ECurve curve>
class ConverterNew
{
public:
  void affineG1(uint8_t *src);
  void affineG2(uint8_t *src);

  void affineG1Array(uint8_t *src, long num);
  void affineG2Array(uint8_t *src, long num);
};
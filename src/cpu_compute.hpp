#include <cstdint>
#include <cstdio>
#include "constants.hpp"

template <ECurve ecurve>
class CpuCompute
{
public:
  uint8_t *computeH(size_t d, uint8_t *ca, uint8_t *cb, uint8_t *cc);
  uint8_t *computeC(uint8_t *evaluation_Ht, uint8_t *evaluation_Bt1, uint8_t *evaluation_Lt, uint8_t *r);

  void computeMultiExp(uint8_t *res, uint8_t *scalars, uint8_t *vectors, long n);
  void computeMultiExpG2(uint8_t *res, uint8_t *scalars, uint8_t *vectors, long n);
};

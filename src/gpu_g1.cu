#include "constants.hpp"

#include "gpu_fq.cuh"
#include "gpu_g1.cuh"
#include "gpu_params.cuh"
#include "gpu_preprocess.cuh"
#include "knary.cuh"

extern __device__ GpuParams gpu_params;

extern __device__ bool test_bit(const fixnum &base, int bitno);

__device__ GpuFq coeff_a()
{
  return gpu_params.get_mnt_coeff_a();
}

__device__ GpuG1 GpuG1::zero()
{
  return GpuG1(GpuFq::zero(), GpuFq::one(), GpuFq::zero());
}

__device__ GpuG1 GpuG1::load_from_array(fixnum *array, long i)
{
  const long lane_idx = fixnum::layout::laneIdx();

  fixnum &a0 = array[3 * i * libms_per_elem + 0 * libms_per_elem + lane_idx];
  fixnum &a1 = array[3 * i * libms_per_elem + 1 * libms_per_elem + lane_idx];
  fixnum &a2 = array[3 * i * libms_per_elem + 2 * libms_per_elem + lane_idx];

  return GpuG1(GpuFq::load(a0), GpuFq::load(a1), GpuFq::load(a2));
}

// __device__ GpuG1 GpuG1::load_from_array_affine(fixnum *array, long i)
// {
//   const long lane_idx = fixnum::layout::laneIdx();

//   fixnum &a0 = array[3 * i * libms_per_elem + 0 * libms_per_elem + lane_idx];
//   fixnum &a1 = array[3 * i * libms_per_elem + 1 * libms_per_elem + lane_idx];

//   GpuFq a2 = lane_idx == 0 ? GpuFq::one() : GpuFq::zero();

//   return GpuG1(GpuFq::load(a0), GpuFq::load(a1), GpuFq::one());
// }

__device__ void GpuG1::save_to_array(fixnum *array, long i)
{
  const long lane_idx = fixnum::layout::laneIdx();

  fixnum &a0 = array[3 * i * libms_per_elem + 0 * libms_per_elem + lane_idx];
  fixnum &a1 = array[3 * i * libms_per_elem + 1 * libms_per_elem + lane_idx];
  fixnum &a2 = array[3 * i * libms_per_elem + 2 * libms_per_elem + lane_idx];

  this->save(a0, a1, a2);
}

__device__ void GpuG1::save(fixnum &c0, fixnum &c1, fixnum &c2)
{
  this->X.save(c0);
  this->Y.save(c1);
  this->Z.save(c2);
}

__device__ bool GpuG1::is_zero() const
{
  return this->X.is_zero() && this->Z.is_zero();
}

__device__ GpuG1 GpuG1::operator-() const
{
  return GpuG1(this->X, -(this->Y), this->Z);
}

__device__ GpuG1 GpuG1::operator-(const GpuG1 &other) const
{
  return (*this) + (-other);
}

__device__ GpuG1 GpuG1::operator+(const GpuG1 &other) const
{
  // handle special cases having to do with O
  if (this->is_zero())
  {
    return other;
  }

  if (other.is_zero())
  {
    return *this;
  }

  const GpuFq X1Z2 = this->X * other.Z; // X1Z2 = X1*Z2
  const GpuFq X2Z1 = this->Z * other.X; // X2Z1 = X2*Z1

  // (used both in add and double checks)

  const GpuFq Y1Z2 = this->Y * other.Z; // Y1Z2 = Y1*Z2
  const GpuFq Y2Z1 = this->Z * other.Y; // Y2Z1 = Y2*Z1

  if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
  {
    // perform dbl case
    const GpuFq XX = this->X.squared();              // XX  = X1^2
    const GpuFq ZZ = this->Z.squared();              // ZZ  = Z1^2
    const GpuFq w = coeff_a() * ZZ + (XX + XX + XX); // w   = a*ZZ + 3*XX
    const GpuFq Y1Z1 = this->Y * this->Z;
    const GpuFq s = Y1Z1 + Y1Z1;                       // s   = 2*Y1*Z1
    const GpuFq ss = s.squared();                      // ss  = s^2
    const GpuFq sss = s * ss;                          // sss = s*ss
    const GpuFq R = this->Y * s;                       // R   = Y1*s
    const GpuFq RR = R.squared();                      // RR  = R^2
    const GpuFq B = (this->X + R).squared() - XX - RR; // B   = (X1+R)^2 - XX - RR
    const GpuFq h = w.squared() - (B + B);             // h   = w^2 - 2*B
    const GpuFq X3 = h * s;                            // X3  = h*s
    const GpuFq Y3 = w * (B - h) - (RR + RR);          // Y3  = w*(B-h) - 2*RR
    const GpuFq Z3 = sss;                              // Z3  = sss

    return GpuG1(X3, Y3, Z3);
  }

  // if we have arrived here we are in the add case
  const GpuFq Z1Z2 = this->Z * other.Z;      // Z1Z2 = Z1*Z2
  const GpuFq u = Y2Z1 - Y1Z2;               // u    = Y2*Z1-Y1Z2
  const GpuFq uu = u.squared();              // uu   = u^2
  const GpuFq v = X2Z1 - X1Z2;               // v    = X2*Z1-X1Z2
  const GpuFq vv = v.squared();              // vv   = v^2
  const GpuFq vvv = v * vv;                  // vvv  = v*vv
  const GpuFq R = vv * X1Z2;                 // R    = vv*X1Z2
  const GpuFq A = uu * Z1Z2 - (vvv + R + R); // A    = uu*Z1Z2 - vvv - 2*R
  const GpuFq X3 = v * A;                    // X3   = v*A
  const GpuFq Y3 = u * (R - A) - vvv * Y1Z2; // Y3   = u*(R-A) - vvv*Y1Z2
  const GpuFq Z3 = vvv * Z1Z2;               // Z3   = vvv*Z1Z2

  return GpuG1(X3, Y3, Z3);
}

__device__ GpuG1 GpuG1::mixed_add(const GpuG1 &other) const
{
  // handle special cases having to do with O
  if (this->is_zero())
  {
    return other;
  }

  if (other.is_zero())
  {
    return *this;
  }

  const GpuFq &X1Z2 = (this->X);            // X1Z2 = X1*Z2 (but other is special and not zero)
  const GpuFq X2Z1 = (this->Z) * (other.X); // X2Z1 = X2*Z1

  // (used both in add and double checks)

  const GpuFq &Y1Z2 = (this->Y);            // Y1Z2 = Y1*Z2 (but other is special and not zero)
  const GpuFq Y2Z1 = (this->Z) * (other.Y); // Y2Z1 = Y2*Z1

  if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
  {
    return this->dbl();
  }

  const GpuFq u = Y2Z1 - this->Y;               // u = Y2*Z1-Y1
  const GpuFq uu = u.squared();                 // uu = u2
  const GpuFq v = X2Z1 - this->X;               // v = X2*Z1-X1
  const GpuFq vv = v.squared();                 // vv = v2
  const GpuFq vvv = v * vv;                     // vvv = v*vv
  const GpuFq R = vv * this->X;                 // R = vv*X1
  const GpuFq A = uu * this->Z - vvv - R - R;   // A = uu*Z1-vvv-2*R
  const GpuFq X3 = v * A;                       // X3 = v*A
  const GpuFq Y3 = u * (R - A) - vvv * this->Y; // Y3 = u*(R-A)-vvv*Y1
  const GpuFq Z3 = vvv * this->Z;               // Z3 = vvv*Z1

  return GpuG1(X3, Y3, Z3);
}

__device__ GpuG1 GpuG1::operator*(const fixnum &scalar)
{
  GpuG1 result = GpuG1::zero();

  bool found_one = false;

  for (long i = fixnum::msb(scalar); i >= 0; --i)
  {
    if (found_one)
    {
      result = result.dbl();
    }

    if (test_bit(scalar, i))
    {
      found_one = true;
      result = result.mixed_add(*this);
    }
  }

  return result;
}

__device__ void GpuG1::operator=(const GpuG1 &g)
{
  this->X = g.X;
  this->Y = g.Y;
  this->Z = g.Z;
}

__device__ GpuG1 GpuG1::dbl() const
{
  if (this->is_zero())
  {
    return (*this);
  }
  else
  {
    // NOTE: does not handle O and pts of order 2,4
    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-dbl-2007-bl
    const GpuFq XX = (this->X).squared();            // XX  = X1^2
    const GpuFq ZZ = (this->Z).squared();            // ZZ  = Z1^2
    const GpuFq w = coeff_a() * ZZ + (XX + XX + XX); // w   = a*ZZ + 3*XX
    const GpuFq Y1Z1 = (this->Y) * (this->Z);
    const GpuFq s = Y1Z1 + Y1Z1;                         // s   = 2*Y1*Z1
    const GpuFq ss = s.squared();                        // ss  = s^2
    const GpuFq sss = s * ss;                            // sss = s*ss
    const GpuFq R = (this->Y) * s;                       // R   = Y1*s
    const GpuFq RR = R.squared();                        // RR  = R^2
    const GpuFq B = ((this->X) + R).squared() - XX - RR; // B   = (X1+R)^2 - XX - RR
    const GpuFq h = w.squared() - (B + B);               // h   = w^2 - 2*B
    const GpuFq X3 = h * s;                              // X3  = h*s
    const GpuFq Y3 = w * (B - h) - (RR + RR);            // Y3  = w*(B-h) - 2*RR
    const GpuFq Z3 = sss;                                // Z3  = sss

    return GpuG1(X3, Y3, Z3);
  }
}

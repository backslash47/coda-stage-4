#include "gpu_fq.cuh"
#include "gpu_fq2.cuh"
#include "gpu_fq3.cuh"
#include "gpu_g2.cuh"
#include "gpu_params.cuh"

extern __device__ GpuParams gpu_params;

extern __device__ bool test_bit(const fixnum &base, int bitno);

__device__ GpuFq2 coeff_a2()
{
  return gpu_params.get_mnt_coeff_a2();
}

__device__ GpuFq twist_mul_by_a_c0()
{
  return gpu_params.get_twist_mul_by_a_c0();
}

__device__ GpuFq twist_mul_by_a_c1()
{
  return gpu_params.get_twist_mul_by_a_c1();
}

__device__ GpuFq twist_mul_by_a_c2()
{
  return gpu_params.get_twist_mul_by_a_c2();
}

__device__ GpuFq3 coeff_a3()
{
  return gpu_params.get_mnt_coeff_a3();
}

__device__ GpuMnt4G2 GpuMnt4G2::zero()
{
  return GpuMnt4G2(GpuFq2::zero(), GpuFq2::one(), GpuFq2::zero());
}

// __device__ GpuMnt4G2 GpuMnt4G2::load_from_array_affine(fixnum *array, long i)
// {
//   const long lane_idx = fixnum::layout::laneIdx();

//   fixnum &a0c0 = array[6 * i * libms_per_elem + 0 * libms_per_elem + lane_idx];
//   fixnum &a0c1 = array[6 * i * libms_per_elem + 1 * libms_per_elem + lane_idx];
//   fixnum &a1c0 = array[6 * i * libms_per_elem + 2 * libms_per_elem + lane_idx];
//   fixnum &a1c1 = array[6 * i * libms_per_elem + 3 * libms_per_elem + lane_idx];

//   return GpuMnt4G2(GpuFq2(GpuFq::load(a0c0), GpuFq::load(a0c1)),
//                    GpuFq2(GpuFq::load(a1c0), GpuFq::load(a1c1)),
//                    GpuFq2::one());
// }

__device__ GpuMnt4G2 GpuMnt4G2::load_from_array(fixnum *array, long i)
{
  const long lane_idx = fixnum::layout::laneIdx();

  fixnum &a0c0 = array[6 * i * libms_per_elem + 0 * libms_per_elem + lane_idx];
  fixnum &a0c1 = array[6 * i * libms_per_elem + 1 * libms_per_elem + lane_idx];
  fixnum &a1c0 = array[6 * i * libms_per_elem + 2 * libms_per_elem + lane_idx];
  fixnum &a1c1 = array[6 * i * libms_per_elem + 3 * libms_per_elem + lane_idx];
  fixnum &a2c0 = array[6 * i * libms_per_elem + 4 * libms_per_elem + lane_idx];
  fixnum &a2c1 = array[6 * i * libms_per_elem + 5 * libms_per_elem + lane_idx];

  return GpuMnt4G2(GpuFq2(GpuFq::load(a0c0), GpuFq::load(a0c1)),
                   GpuFq2(GpuFq::load(a1c0), GpuFq::load(a1c1)),
                   GpuFq2(GpuFq::load(a2c0), GpuFq::load(a2c1)));
}

__device__ void GpuMnt4G2::save_to_array(fixnum *array, long i)
{
  const long lane_idx = fixnum::layout::laneIdx();

  fixnum &a0c0 = array[6 * i * libms_per_elem + 0 * libms_per_elem + lane_idx];
  fixnum &a0c1 = array[6 * i * libms_per_elem + 1 * libms_per_elem + lane_idx];
  fixnum &a1c0 = array[6 * i * libms_per_elem + 2 * libms_per_elem + lane_idx];
  fixnum &a1c1 = array[6 * i * libms_per_elem + 3 * libms_per_elem + lane_idx];
  fixnum &a2c0 = array[6 * i * libms_per_elem + 4 * libms_per_elem + lane_idx];
  fixnum &a2c1 = array[6 * i * libms_per_elem + 5 * libms_per_elem + lane_idx];

  this->X.save(a0c0, a0c1);
  this->Y.save(a1c0, a1c1);
  this->Z.save(a2c0, a2c1);
}

__device__ void GpuMnt4G2::save(fixnum &xc0, fixnum &xc1, fixnum &yc0, fixnum &yc1, fixnum &zc0, fixnum &zc1)
{
  this->X.save(xc0, xc1);
  this->Y.save(yc0, yc1);
  this->Z.save(zc0, zc1);
}

__device__ bool GpuMnt4G2::is_zero() const
{
  return this->X.is_zero() && this->Z.is_zero();
}

__device__ GpuMnt4G2 GpuMnt4G2::operator-() const
{
  return GpuMnt4G2(this->X, -(this->Y), this->Z);
}

__device__ GpuMnt4G2 GpuMnt4G2::operator-(const GpuMnt4G2 &other) const
{
  return (*this) + (-other);
}

__device__ GpuMnt4G2 GpuMnt4G2::operator+(const GpuMnt4G2 &other) const
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

  const GpuFq2 X1Z2 = this->X * other.Z; // X1Z2 = X1*Z2
  const GpuFq2 X2Z1 = this->Z * other.X; // X2Z1 = X2*Z1

  // (used both in add and double checks)

  const GpuFq2 Y1Z2 = this->Y * other.Z; // Y1Z2 = Y1*Z2
  const GpuFq2 Y2Z1 = this->Z * other.Y; // Y2Z1 = Y2*Z1

  if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
  {
    // perform dbl case
    const GpuFq2 XX = this->X.squared();                       // XX  = X1^2
    const GpuFq2 ZZ = this->Z.squared();                       // ZZ  = Z1^2
    const GpuFq2 w = GpuMnt4G2::mul_by_a(ZZ) + (XX + XX + XX); // w   = a*ZZ + 3*XX
    const GpuFq2 Y1Z1 = this->Y * this->Z;
    const GpuFq2 s = Y1Z1 + Y1Z1;                       // s   = 2*Y1*Z1
    const GpuFq2 ss = s.squared();                      // ss  = s^2
    const GpuFq2 sss = s * ss;                          // sss = s*ss
    const GpuFq2 R = this->Y * s;                       // R   = Y1*s
    const GpuFq2 RR = R.squared();                      // RR  = R^2
    const GpuFq2 B = (this->X + R).squared() - XX - RR; // B   = (X1+R)^2 - XX - RR
    const GpuFq2 h = w.squared() - (B + B);             // h   = w^2 - 2*B
    const GpuFq2 X3 = h * s;                            // X3  = h*s
    const GpuFq2 Y3 = w * (B - h) - (RR + RR);          // Y3  = w*(B-h) - 2*RR
    const GpuFq2 Z3 = sss;                              // Z3  = sss

    return GpuMnt4G2(X3, Y3, Z3);
  }

  // if we have arrived here we are in the add case
  const GpuFq2 Z1Z2 = this->Z * other.Z;      // Z1Z2 = Z1*Z2
  const GpuFq2 u = Y2Z1 - Y1Z2;               // u    = Y2*Z1-Y1Z2
  const GpuFq2 uu = u.squared();              // uu   = u^2
  const GpuFq2 v = X2Z1 - X1Z2;               // v    = X2*Z1-X1Z2
  const GpuFq2 vv = v.squared();              // vv   = v^2
  const GpuFq2 vvv = v * vv;                  // vvv  = v*vv
  const GpuFq2 R = vv * X1Z2;                 // R    = vv*X1Z2
  const GpuFq2 A = uu * Z1Z2 - (vvv + R + R); // A    = uu*Z1Z2 - vvv - 2*R
  const GpuFq2 X3 = v * A;                    // X3   = v*A
  const GpuFq2 Y3 = u * (R - A) - vvv * Y1Z2; // Y3   = u*(R-A) - vvv*Y1Z2
  const GpuFq2 Z3 = vvv * Z1Z2;               // Z3   = vvv*Z1Z2

  return GpuMnt4G2(X3, Y3, Z3);
}

__device__ GpuMnt4G2 GpuMnt4G2::dbl() const
{
  if (this->is_zero())
  {
    return (*this);
  }
  else
  {
    // NOTE: does not handle O and pts of order 2,4
    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-dbl-2007-bl
    const GpuFq2 XX = (this->X).squared();                     // XX  = X1^2
    const GpuFq2 ZZ = (this->Z).squared();                     // ZZ  = Z1^2
    const GpuFq2 w = GpuMnt4G2::mul_by_a(ZZ) + (XX + XX + XX); // w   = a*ZZ + 3*XX
    const GpuFq2 Y1Z1 = (this->Y) * (this->Z);
    const GpuFq2 s = Y1Z1 + Y1Z1;                         // s   = 2*Y1*Z1
    const GpuFq2 ss = s.squared();                        // ss  = s^2
    const GpuFq2 sss = s * ss;                            // sss = s*ss
    const GpuFq2 R = (this->Y) * s;                       // R   = Y1*s
    const GpuFq2 RR = R.squared();                        // RR  = R^2
    const GpuFq2 B = ((this->X) + R).squared() - XX - RR; // B   = (X1+R)^2 - XX - RR
    const GpuFq2 h = w.squared() - (B + B);               // h   = w^2 - 2*B
    const GpuFq2 X3 = h * s;                              // X3  = h*s
    const GpuFq2 Y3 = w * (B - h) - (RR + RR);            // Y3  = w*(B-h) - 2*RR
    const GpuFq2 Z3 = sss;                                // Z3  = sss

    return GpuMnt4G2(X3, Y3, Z3);
  }
}

__device__ GpuMnt4G2 GpuMnt4G2::mixed_add(const GpuMnt4G2 &other) const
{
  if (this->is_zero())
  {
    return other;
  }

  if (other.is_zero())
  {
    return *this;
  }

  const GpuFq2 &X1Z2 = (this->X);            // X1Z2 = X1*Z2 (but other is special and not zero)
  const GpuFq2 X2Z1 = (this->Z) * (other.X); // X2Z1 = X2*Z1

  // (used both in add and double checks)

  const GpuFq2 &Y1Z2 = (this->Y);            // Y1Z2 = Y1*Z2 (but other is special and not zero)
  const GpuFq2 Y2Z1 = (this->Z) * (other.Y); // Y2Z1 = Y2*Z1

  if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
  {
    return this->dbl();
  }

  const GpuFq2 u = Y2Z1 - this->Y;               // u = Y2*Z1-Y1
  const GpuFq2 uu = u.squared();                 // uu = u2
  const GpuFq2 v = X2Z1 - this->X;               // v = X2*Z1-X1
  const GpuFq2 vv = v.squared();                 // vv = v2
  const GpuFq2 vvv = v * vv;                     // vvv = v*vv
  const GpuFq2 R = vv * this->X;                 // R = vv*X1
  const GpuFq2 A = uu * this->Z - vvv - R - R;   // A = uu*Z1-vvv-2*R
  const GpuFq2 X3 = v * A;                       // X3 = v*A
  const GpuFq2 Y3 = u * (R - A) - vvv * this->Y; // Y3 = u*(R-A)-vvv*Y1
  const GpuFq2 Z3 = vvv * this->Z;               // Z3 = vvv*Z1

  return GpuMnt4G2(X3, Y3, Z3);
}

__device__ GpuFq2 GpuMnt4G2::mul_by_a(const GpuFq2 &elt)
{
  return GpuFq2(twist_mul_by_a_c0() * elt.c0, twist_mul_by_a_c1() * elt.c1);
}

__device__ GpuMnt4G2 GpuMnt4G2::operator*(const fixnum &scalar)
{
  GpuMnt4G2 result = GpuMnt4G2::zero();

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
      // result = result + *this;
      result = result.mixed_add(*this);
    }
  }

  return result;
}

__device__ void GpuMnt6G2::save(fixnum &xc0, fixnum &xc1, fixnum &xc2, fixnum &yc0, fixnum &yc1, fixnum &yc2, fixnum &zc0, fixnum &zc1, fixnum &zc2)
{
  this->X.save(xc0, xc1, xc2);
  this->Y.save(yc0, yc1, yc2);
  this->Z.save(zc0, zc1, zc2);
}

__device__ GpuMnt6G2 GpuMnt6G2::zero()
{
  return GpuMnt6G2(GpuFq3::zero(), GpuFq3::one(), GpuFq3::zero());
}

// __device__ GpuMnt6G2 GpuMnt6G2::load_from_array_affine(fixnum *array, long i)
// {
//   const long lane_idx = fixnum::layout::laneIdx();

//   fixnum &a0c0 = array[9 * i * libms_per_elem + 0 * libms_per_elem + lane_idx];
//   fixnum &a0c1 = array[9 * i * libms_per_elem + 1 * libms_per_elem + lane_idx];
//   fixnum &a0c2 = array[9 * i * libms_per_elem + 2 * libms_per_elem + lane_idx];
//   fixnum &a1c0 = array[9 * i * libms_per_elem + 3 * libms_per_elem + lane_idx];
//   fixnum &a1c1 = array[9 * i * libms_per_elem + 4 * libms_per_elem + lane_idx];
//   fixnum &a1c2 = array[9 * i * libms_per_elem + 5 * libms_per_elem + lane_idx];

//   return GpuMnt6G2(GpuFq3(GpuFq::load(a0c0), GpuFq::load(a0c1), GpuFq::load(a0c2)),
//                    GpuFq3(GpuFq::load(a1c0), GpuFq::load(a1c1), GpuFq::load(a1c2)),
//                    GpuFq3::one());
// }

__device__ GpuMnt6G2 GpuMnt6G2::load_from_array(fixnum *array, long i)
{
  const long lane_idx = fixnum::layout::laneIdx();

  fixnum &a0c0 = array[9 * i * libms_per_elem + 0 * libms_per_elem + lane_idx];
  fixnum &a0c1 = array[9 * i * libms_per_elem + 1 * libms_per_elem + lane_idx];
  fixnum &a0c2 = array[9 * i * libms_per_elem + 2 * libms_per_elem + lane_idx];
  fixnum &a1c0 = array[9 * i * libms_per_elem + 3 * libms_per_elem + lane_idx];
  fixnum &a1c1 = array[9 * i * libms_per_elem + 4 * libms_per_elem + lane_idx];
  fixnum &a1c2 = array[9 * i * libms_per_elem + 5 * libms_per_elem + lane_idx];
  fixnum &a2c0 = array[9 * i * libms_per_elem + 6 * libms_per_elem + lane_idx];
  fixnum &a2c1 = array[9 * i * libms_per_elem + 7 * libms_per_elem + lane_idx];
  fixnum &a2c2 = array[9 * i * libms_per_elem + 8 * libms_per_elem + lane_idx];

  return GpuMnt6G2(GpuFq3(GpuFq::load(a0c0), GpuFq::load(a0c1), GpuFq::load(a0c2)),
                   GpuFq3(GpuFq::load(a1c0), GpuFq::load(a1c1), GpuFq::load(a1c2)),
                   GpuFq3(GpuFq::load(a2c0), GpuFq::load(a2c1), GpuFq::load(a2c2)));
}

__device__ void GpuMnt6G2::save_to_array(fixnum *array, long i)
{
  const long lane_idx = fixnum::layout::laneIdx();

  fixnum &a0c0 = array[9 * i * libms_per_elem + 0 * libms_per_elem + lane_idx];
  fixnum &a0c1 = array[9 * i * libms_per_elem + 1 * libms_per_elem + lane_idx];
  fixnum &a0c2 = array[9 * i * libms_per_elem + 2 * libms_per_elem + lane_idx];
  fixnum &a1c0 = array[9 * i * libms_per_elem + 3 * libms_per_elem + lane_idx];
  fixnum &a1c1 = array[9 * i * libms_per_elem + 4 * libms_per_elem + lane_idx];
  fixnum &a1c2 = array[9 * i * libms_per_elem + 5 * libms_per_elem + lane_idx];
  fixnum &a2c0 = array[9 * i * libms_per_elem + 6 * libms_per_elem + lane_idx];
  fixnum &a2c1 = array[9 * i * libms_per_elem + 7 * libms_per_elem + lane_idx];
  fixnum &a2c2 = array[9 * i * libms_per_elem + 8 * libms_per_elem + lane_idx];

  this->X.save(a0c0, a0c1, a0c2);
  this->Y.save(a1c0, a1c1, a1c2);
  this->Z.save(a2c0, a2c1, a2c2);
}

__device__ bool GpuMnt6G2::is_zero() const
{
  return this->X.is_zero() && this->Z.is_zero();
}

__device__ GpuMnt6G2 GpuMnt6G2::operator+(const GpuMnt6G2 &other) const
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

  const GpuFq3 X1Z2 = this->X * other.Z; // X1Z2 = X1*Z2
  const GpuFq3 X2Z1 = this->Z * other.X; // X2Z1 = X2*Z1

  // (used both in add and double checks)

  const GpuFq3 Y1Z2 = this->Y * other.Z; // Y1Z2 = Y1*Z2
  const GpuFq3 Y2Z1 = this->Z * other.Y; // Y2Z1 = Y2*Z1

  if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
  {
    // perform dbl case
    const GpuFq3 XX = this->X.squared();               // XX  = X1^2
    const GpuFq3 ZZ = this->Z.squared();               // ZZ  = Z1^2
    const GpuFq3 w = coeff_a3() * ZZ + (XX + XX + XX); // w   = a*ZZ + 3*XX
    const GpuFq3 Y1Z1 = this->Y * this->Z;
    const GpuFq3 s = Y1Z1 + Y1Z1;                       // s   = 2*Y1*Z1
    const GpuFq3 ss = s.squared();                      // ss  = s^2
    const GpuFq3 sss = s * ss;                          // sss = s*ss
    const GpuFq3 R = this->Y * s;                       // R   = Y1*s
    const GpuFq3 RR = R.squared();                      // RR  = R^2
    const GpuFq3 B = (this->X + R).squared() - XX - RR; // B   = (X1+R)^2 - XX - RR
    const GpuFq3 h = w.squared() - (B + B);             // h   = w^2 - 2*B
    const GpuFq3 X3 = h * s;                            // X3  = h*s
    const GpuFq3 Y3 = w * (B - h) - (RR + RR);          // Y3  = w*(B-h) - 2*RR
    const GpuFq3 Z3 = sss;                              // Z3  = sss

    return GpuMnt6G2(X3, Y3, Z3);
  }

  // if we have arrived here we are in the add case
  const GpuFq3 Z1Z2 = this->Z * other.Z;      // Z1Z2 = Z1*Z2
  const GpuFq3 u = Y2Z1 - Y1Z2;               // u    = Y2*Z1-Y1Z2
  const GpuFq3 uu = u.squared();              // uu   = u^2
  const GpuFq3 v = X2Z1 - X1Z2;               // v    = X2*Z1-X1Z2
  const GpuFq3 vv = v.squared();              // vv   = v^2
  const GpuFq3 vvv = v * vv;                  // vvv  = v*vv
  const GpuFq3 R = vv * X1Z2;                 // R    = vv*X1Z2
  const GpuFq3 A = uu * Z1Z2 - (vvv + R + R); // A    = uu*Z1Z2 - vvv - 2*R
  const GpuFq3 X3 = v * A;                    // X3   = v*A
  const GpuFq3 Y3 = u * (R - A) - vvv * Y1Z2; // Y3   = u*(R-A) - vvv*Y1Z2
  const GpuFq3 Z3 = vvv * Z1Z2;               // Z3   = vvv*Z1Z2

  return GpuMnt6G2(X3, Y3, Z3);
}

__device__ GpuMnt6G2 GpuMnt6G2::dbl() const
{
  if (this->is_zero())
  {
    return (*this);
  }
  else
  {
    // NOTE: does not handle O and pts of order 2,4
    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-dbl-2007-bl
    const GpuFq3 XX = (this->X).squared();                     // XX  = X1^2
    const GpuFq3 ZZ = (this->Z).squared();                     // ZZ  = Z1^2
    const GpuFq3 w = GpuMnt6G2::mul_by_a(ZZ) + (XX + XX + XX); // w   = a*ZZ + 3*XX
    const GpuFq3 Y1Z1 = (this->Y) * (this->Z);
    const GpuFq3 s = Y1Z1 + Y1Z1;                         // s   = 2*Y1*Z1
    const GpuFq3 ss = s.squared();                        // ss  = s^2
    const GpuFq3 sss = s * ss;                            // sss = s*ss
    const GpuFq3 R = (this->Y) * s;                       // R   = Y1*s
    const GpuFq3 RR = R.squared();                        // RR  = R^2
    const GpuFq3 B = ((this->X) + R).squared() - XX - RR; // B   = (X1+R)^2 - XX - RR
    const GpuFq3 h = w.squared() - (B + B);               // h   = w^2 - 2*B
    const GpuFq3 X3 = h * s;                              // X3  = h*s
    const GpuFq3 Y3 = w * (B - h) - (RR + RR);            // Y3  = w*(B-h) - 2*RR
    const GpuFq3 Z3 = sss;                                // Z3  = sss

    return GpuMnt6G2(X3, Y3, Z3);
  }
}

__device__ GpuFq3 GpuMnt6G2::mul_by_a(const GpuFq3 &elt)
{
  return GpuFq3(twist_mul_by_a_c0() * elt.c1, twist_mul_by_a_c1() * elt.c2, twist_mul_by_a_c2() * elt.c0);
}

__device__ GpuMnt6G2 GpuMnt6G2::mixed_add(const GpuMnt6G2 &other) const
{
  if (this->is_zero())
  {
    return other;
  }

  if (other.is_zero())
  {
    return *this;
  }

  const GpuFq3 &X1Z2 = (this->X);            // X1Z2 = X1*Z2 (but other is special and not zero)
  const GpuFq3 X2Z1 = (this->Z) * (other.X); // X2Z1 = X2*Z1

  // (used both in add and double checks)

  const GpuFq3 &Y1Z2 = (this->Y);            // Y1Z2 = Y1*Z2 (but other is special and not zero)
  const GpuFq3 Y2Z1 = (this->Z) * (other.Y); // Y2Z1 = Y2*Z1

  if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
  {
    return this->dbl();
  }

  const GpuFq3 u = Y2Z1 - this->Y;               // u = Y2*Z1-Y1
  const GpuFq3 uu = u.squared();                 // uu = u2
  const GpuFq3 v = X2Z1 - this->X;               // v = X2*Z1-X1
  const GpuFq3 vv = v.squared();                 // vv = v2
  const GpuFq3 vvv = v * vv;                     // vvv = v*vv
  const GpuFq3 R = vv * this->X;                 // R = vv*X1
  const GpuFq3 A = uu * this->Z - vvv - R - R;   // A = uu*Z1-vvv-2*R
  const GpuFq3 X3 = v * A;                       // X3 = v*A
  const GpuFq3 Y3 = u * (R - A) - vvv * this->Y; // Y3 = u*(R-A)-vvv*Y1
  const GpuFq3 Z3 = vvv * this->Z;               // Z3 = vvv*Z1

  return GpuMnt6G2(X3, Y3, Z3);
}

__device__ GpuMnt6G2 GpuMnt6G2::operator-() const
{
  return GpuMnt6G2(this->X, -(this->Y), this->Z);
}

__device__ GpuMnt6G2 GpuMnt6G2::operator-(const GpuMnt6G2 &other) const
{
  return (*this) + (-other);
}

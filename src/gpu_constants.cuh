#pragma once

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"

#include "constants.hpp"

using namespace cuFIXNUM;

typedef warp_fixnum<bytes_per_elem, u64_fixnum> fixnum;
typedef fixnum_array<fixnum> my_fixnum_array;
// redc may be worth trying over cios
typedef modnum_monty_cios<fixnum> modnum;

#pragma once

#include <cstdint>
#include "gpu_constants.cuh"

uint8_t *get_1D_fixnum_array(my_fixnum_array *res, int n);
uint8_t *get_2D_fixnum_array(my_fixnum_array *res0, my_fixnum_array *res1, int nelts);
uint8_t *get_3D_fixnum_array(my_fixnum_array *res0, my_fixnum_array *res1, my_fixnum_array *res2, int nelts);
uint8_t *get_6D_fixnum_array( my_fixnum_array *res0c0, my_fixnum_array *res0c1, 
                              my_fixnum_array *res1c0, my_fixnum_array *res1c1, 
                              my_fixnum_array *res2c0, my_fixnum_array *res2c1, 
                              int nelts);
uint8_t *get_9D_fixnum_array( my_fixnum_array *res0c0, my_fixnum_array *res0c1, my_fixnum_array *res0c2,
                              my_fixnum_array *res1c0, my_fixnum_array *res1c1, my_fixnum_array *res1c2,
                              my_fixnum_array *res2c0, my_fixnum_array *res2c1, my_fixnum_array *res2c2,
                              int nelts);


void get_1D_fixnum_array(uint8_t * dst, my_fixnum_array *src, int n);
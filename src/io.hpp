#include <cstdio>
#include <cstdint>
#include "constants.hpp"

void read_mnt_fq(uint8_t *dest, FILE *inputs);
void read_mnt_fq2(uint8_t *dest, FILE *inputs);
void read_mnt_fq3(uint8_t *dest, FILE *inputs);

void read_mnt4_g2(uint8_t *dest, FILE *inputs);
void read_mnt6_g2(uint8_t *dest, FILE *inputs);

void write_mnt_fq(uint8_t *fq, FILE *outputs);
void write_mnt_fq2(uint8_t *src, FILE *outputs);
void write_mnt_fq3(uint8_t *src, FILE *outputs);
void write_mnt4_g2(uint8_t *src, FILE *outputs);
void write_mnt6_g2(uint8_t *src, FILE *outputs);

void read_mnt4_fq_montgomery(uint8_t *dest, FILE *inputs);
void read_mnt6_fq_montgomery(uint8_t *dest, FILE *inputs);
void read_mnt4_g1_montgomery(uint8_t *dest, FILE *inputs);
void read_mnt4_g1_montgomery_zero(uint8_t *dest, FILE *inputs);
void read_mnt6_g1_montgomery(uint8_t *dest, FILE *inputs);
void read_mnt6_g1_montgomery_zero(uint8_t *dest, FILE *inputs);
void read_mnt4_g2_montgomery(uint8_t *dest, FILE *inputs);
void read_mnt4_g2_montgomery_zero(uint8_t *dest, FILE *inputs);
void read_mnt6_g2_montgomery_zero(uint8_t *dest, FILE *inputs);
void read_mnt6_g2_montgomery(uint8_t *dest, FILE *inputs);
void write_mnt4_g1_montgomery(uint8_t *src, FILE *outputs);
void write_mnt6_g1_montgomery(uint8_t *src, FILE *outputs);
void write_mnt4_g2_montgomery(uint8_t *src, FILE *outputs);
void write_mnt6_g2_montgomery(uint8_t *src, FILE *outputs);

void read_mnt4_fr_montgomery(uint8_t *dest, FILE *inputs);
void read_mnt6_fr_montgomery(uint8_t *dest, FILE *inputs);

void init_libff();

template <typename curve>
uint8_t *g1_to_affine(uint8_t *src);

template <typename curve>
uint8_t *g2_to_affine(uint8_t *src);

template <ECurve curve, GEnum Gx>
void g_to_affine_inplace(uint8_t *src);

size_t read_size_t(FILE *input);
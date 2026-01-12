// include/kaldi-math.h - minimal C23 math helpers
#ifndef KALDI_NATIVE_FBANK_CSRC_KALDI_MATH_H_
#define KALDI_NATIVE_FBANK_CSRC_KALDI_MATH_H_

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

constexpr double KNF_PI = 3.14159265358979323846;
constexpr double KNF_TWO_PI = 6.28318530717958647692;
constexpr double KNF_SQRT2 = 1.41421356237309504880;
constexpr float KNF_PI_F = 3.14159265358979323846f;
constexpr float KNF_TWO_PI_F = 6.28318530717958647692f;
constexpr float KNF_SQRT2_F = 1.41421356237309504880f;

typedef struct {
  unsigned seed;
} knf_random_state;

void knf_random_state_init(knf_random_state *state);
int knf_rand(knf_random_state *state);
float knf_rand_uniform(knf_random_state *state);
float knf_rand_gauss(knf_random_state *state);
void knf_sqrt_inplace(float *in_out, int32_t n);

#endif // KALDI_NATIVE_FBANK_CSRC_KALDI_MATH_H_

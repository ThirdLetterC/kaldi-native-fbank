// Minimal math helpers for C23 port.

#include <math.h>
#include <stdlib.h>

#include "kaldi-math.h"

void knf_random_state_init(knf_random_state *state) {
  state->seed = (unsigned)rand();
}

int knf_rand(knf_random_state *state) {
  if (state == nullptr) {
    return rand();
  }
  state->seed = (1103515245 * state->seed + 12345) & 0x7fffffff;
  return (int)state->seed;
}

float knf_rand_uniform(knf_random_state *state) {
  return (float)((knf_rand(state) + 1.0) / (RAND_MAX + 2.0));
}

float knf_rand_gauss(knf_random_state *state) {
  float u1 = knf_rand_uniform(state);
  float u2 = knf_rand_uniform(state);
  return (float)(sqrtf(-2.0f * logf(u1)) * cosf(2.0f * KNF_PI_F * u2));
}

void knf_sqrt_inplace(float *in_out, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    in_out[i] = sqrtf(in_out[i]);
  }
}

// Real FFT wrapper implemented in C on top of FFTW.

#include <fftw3.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "rfft.h"

struct knf_rfft_state {
  fftwf_complex *freq;
  float *time;
  fftwf_plan plan;
};

knf_rfft *knf_rfft_create(int32_t n, bool inverse) {
  if ((n & 1) != 0 || n <= 0) {
    return nullptr;
  }

  auto fft = (knf_rfft *)calloc(1, sizeof(knf_rfft));
  if (fft == nullptr)
    return nullptr;

  auto state =
      (struct knf_rfft_state *)calloc(1, sizeof(struct knf_rfft_state));
  if (state == nullptr) {
    free(fft);
    return nullptr;
  }

  fft->n = n;
  fft->inverse = inverse;
  fft->scale = inverse ? 1.0f : 1.0f;
  fft->plan = state;
  state->freq =
      (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (n / 2 + 1));
  state->time = (float *)fftwf_malloc(sizeof(float) * n);

  if (state->freq == nullptr || state->time == nullptr) {
    knf_rfft_destroy(fft);
    return nullptr;
  }

  if (!inverse) {
    state->plan =
        fftwf_plan_dft_r2c_1d(n, state->time, state->freq, FFTW_MEASURE);
  } else {
    state->plan =
        fftwf_plan_dft_c2r_1d(n, state->freq, state->time, FFTW_MEASURE);
  }

  if (state->plan == nullptr) {
    knf_rfft_destroy(fft);
    return nullptr;
  }

  return fft;
}

void knf_rfft_destroy(knf_rfft *fft) {
  if (!fft)
    return;
  struct knf_rfft_state *state = (struct knf_rfft_state *)fft->plan;
  if (state) {
    if (state->plan)
      fftwf_destroy_plan(state->plan);
    if (state->freq)
      fftwf_free(state->freq);
    if (state->time)
      fftwf_free(state->time);
    free(state);
  }
  free(fft);
}

void knf_rfft_compute(knf_rfft *fft, float *in_out) {
  KNF_CHECK(fft != nullptr);
  KNF_CHECK(in_out != nullptr);

  struct knf_rfft_state *state = (struct knf_rfft_state *)fft->plan;
  if (!fft->inverse) {
    memcpy(state->time, in_out, sizeof(float) * fft->n);
    fftwf_execute(state->plan);

    in_out[0] = state->freq[0][0];
    in_out[1] = state->freq[fft->n / 2][0];
    for (int32_t i = 1; i < fft->n / 2; ++i) {
      in_out[2 * i] = state->freq[i][0];
      in_out[2 * i + 1] = state->freq[i][1];
    }
  } else {
    state->freq[0][0] = in_out[0];
    state->freq[0][1] = 0.0f;
    state->freq[fft->n / 2][0] = in_out[1];
    state->freq[fft->n / 2][1] = 0.0f;

    for (int32_t i = 1; i < fft->n / 2; ++i) {
      state->freq[i][0] = in_out[2 * i];
      state->freq[i][1] = in_out[2 * i + 1];
    }

    fftwf_execute(state->plan);
    memcpy(in_out, state->time, sizeof(float) * fft->n);
  }
}

// Real FFT wrapper implemented in C23 on top of pocketfft.

#include <stdlib.h>
#include <string.h>

#include "kaldi-native-fbank/log.h"
#include "pocketfft/pocketfft.h"
#include "kaldi-native-fbank/rfft.h"

struct knf_rfft_state {
  rfft_plan plan;
  double *buffer;
};

[[nodiscard]] knf_rfft *knf_rfft_create(int32_t n, bool inverse) {
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
  state->plan = make_rfft_plan((size_t)n);
  state->buffer = (double *)calloc((size_t)n, sizeof(double));

  if (state->plan == nullptr || state->buffer == nullptr) {
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
      destroy_rfft_plan(state->plan);
    free(state->buffer);
    free(state);
  }
  free(fft);
}

void knf_rfft_compute(knf_rfft *fft, float *in_out) {
  KNF_CHECK(fft != nullptr);
  KNF_CHECK(in_out != nullptr);

  struct knf_rfft_state *state = (struct knf_rfft_state *)fft->plan;
  if (!fft->inverse) {
    for (int32_t i = 0; i < fft->n; ++i)
      state->buffer[i] = (double)in_out[i];
    const int status = rfft_forward(state->plan, state->buffer, 1.0);
    if (status != 0) {
      KNF_LOG_ERROR("rfft_forward failed with status %d", status);
      return;
    }

    in_out[0] = (float)state->buffer[0];
    in_out[1] = (float)state->buffer[1];
    for (int32_t i = 1; i < fft->n / 2; ++i) {
      in_out[2 * i] = (float)state->buffer[2 * i];
      in_out[2 * i + 1] = (float)state->buffer[2 * i + 1];
    }
  } else {
    state->buffer[0] = (double)in_out[0];
    state->buffer[1] = (double)in_out[1];

    for (int32_t i = 1; i < fft->n / 2; ++i) {
      state->buffer[2 * i] = (double)in_out[2 * i];
      state->buffer[2 * i + 1] = (double)in_out[2 * i + 1];
    }

    const int status = rfft_backward(state->plan, state->buffer, 1.0);
    if (status != 0) {
      KNF_LOG_ERROR("rfft_backward failed with status %d", status);
      return;
    }

    for (int32_t i = 0; i < fft->n; ++i)
      in_out[i] = (float)state->buffer[i];
  }
}

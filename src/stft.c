// Short-time Fourier Transform implementation in C23.

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "rfft.h"
#include "stft.h"

void knf_stft_config_default(knf_stft_config *cfg) {
  cfg->n_fft = 400;      // 25ms at 16k
  cfg->hop_length = 160; // 10ms at 16k
  cfg->win_length = 400;
  cfg->center = true;
  strncpy(cfg->pad_mode, "reflect", sizeof(cfg->pad_mode));
  cfg->normalized = false;
  cfg->window_override.data = nullptr;
  cfg->window_override.size = 0;
  strncpy(cfg->window_type, "povey", sizeof(cfg->window_type));
}

static void knf_pad_reflect(const float *data, int32_t n, int32_t pad,
                            float *out) {
  // left pad
  for (int32_t i = 0; i < pad; ++i) {
    int32_t src = pad - i;
    if (src >= n)
      src = n - 1;
    out[i] = data[src];
  }
  memcpy(out + pad, data, sizeof(float) * n);
  for (int32_t i = 0; i < pad; ++i) {
    int32_t src = n - 2 - i;
    if (src < 0)
      src = 0;
    out[pad + n + i] = data[src];
  }
}

static void knf_pad_replicate(const float *data, int32_t n, int32_t pad,
                              float *out) {
  for (int32_t i = 0; i < pad; ++i)
    out[i] = data[0];
  memcpy(out + pad, data, sizeof(float) * n);
  for (int32_t i = 0; i < pad; ++i)
    out[pad + n + i] = data[n - 1];
}

static void knf_pad_constant(const float *data, int32_t n, int32_t pad,
                             float *out) {
  memset(out, 0, sizeof(float) * (n + 2 * pad));
  memcpy(out + pad, data, sizeof(float) * n);
}

bool knf_stft_compute(const knf_stft_config *cfg, const float *data, int32_t n,
                      knf_stft_result *out) {
  KNF_CHECK(cfg != nullptr);
  KNF_CHECK(data != nullptr);
  if (cfg->n_fft <= 0 || cfg->hop_length <= 0 || cfg->win_length <= 0) {
    return false;
  }

  knf_window window = cfg->window_override;
  bool owns_window = false;
  if (window.size == 0) {
    owns_window =
        knf_make_window(cfg->window_type, cfg->win_length, 0.42f, &window);
    if (!owns_window)
      return false;
  }

  int32_t pad = cfg->center ? cfg->n_fft / 2 : 0;
  int32_t padded_len = n + 2 * pad;
  float *padded =
      (float *)calloc((size_t)(padded_len > 0 ? padded_len : 1), sizeof(float));
  if (padded == nullptr) {
    if (owns_window)
      knf_free_window(&window);
    return false;
  }

  if (cfg->center) {
    if (strncmp(cfg->pad_mode, "reflect", 7) == 0) {
      knf_pad_reflect(data, n, pad, padded);
    } else if (strncmp(cfg->pad_mode, "replicate", 9) == 0) {
      knf_pad_replicate(data, n, pad, padded);
    } else {
      knf_pad_constant(data, n, pad, padded);
    }
    data = padded;
    n = padded_len;
  } else {
    memcpy(padded, data, sizeof(float) * n);
  }

  int64_t num_frames = 1 + (n - cfg->n_fft) / cfg->hop_length;
  if (num_frames <= 0) {
    free(padded);
    if (owns_window)
      knf_free_window(&window);
    return false;
  }

  knf_rfft *fft = knf_rfft_create(cfg->n_fft, false);
  if (fft == nullptr) {
    free(padded);
    if (owns_window)
      knf_free_window(&window);
    return false;
  }

  int32_t bins = cfg->n_fft / 2 + 1;
  out->real = (float *)calloc((size_t)num_frames * bins, sizeof(float));
  out->imag = (float *)calloc((size_t)num_frames * bins, sizeof(float));
  out->num_frames = (int32_t)num_frames;
  out->n_fft = cfg->n_fft;

  if (out->real == nullptr || out->imag == nullptr) {
    free(out->real);
    free(out->imag);
    free(padded);
    knf_rfft_destroy(fft);
    if (owns_window)
      knf_free_window(&window);
    return false;
  }

  float *frame = (float *)calloc((size_t)cfg->n_fft, sizeof(float));
  if (frame == nullptr) {
    free(out->real);
    free(out->imag);
    free(padded);
    knf_rfft_destroy(fft);
    if (owns_window)
      knf_free_window(&window);
    return false;
  }
  for (int32_t i = 0; i < num_frames; ++i) {
    memcpy(frame, data + i * cfg->hop_length, sizeof(float) * cfg->n_fft);
    if (window.size > 0) {
      knf_apply_window(&window, frame);
    }
    knf_rfft_compute(fft, frame);
    for (int32_t k = 0; k < cfg->n_fft / 2; ++k) {
      if (k == 0) {
        out->real[i * bins] = frame[0];
        out->real[i * bins + cfg->n_fft / 2] = frame[1];
      } else {
        out->real[i * bins + k] = frame[2 * k];
        out->imag[i * bins + k] = frame[2 * k + 1];
      }
    }
  }

  if (cfg->normalized) {
    float scale = 1.0f / sqrtf((float)cfg->n_fft);
    for (int64_t i = 0; i < (int64_t)num_frames * bins; ++i) {
      out->real[i] *= scale;
      out->imag[i] *= scale;
    }
  }

  free(frame);
  free(padded);
  knf_rfft_destroy(fft);
  if (owns_window)
    knf_free_window(&window);
  return true;
}

void knf_stft_result_free(knf_stft_result *res) {
  if (res == nullptr)
    return;
  free(res->real);
  free(res->imag);
  res->real = nullptr;
  res->imag = nullptr;
  res->num_frames = 0;
  res->n_fft = 0;
}

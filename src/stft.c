// Short-time Fourier Transform implementation in C23.

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "kaldi-native-fbank/log.h"
#include "kaldi-native-fbank/rfft.h"
#include "kaldi-native-fbank/stft.h"

static bool knf_fixed_cstr_eq(const char *text, size_t text_cap,
                              const char *literal) {
  if (text == nullptr || literal == nullptr || text_cap == 0) {
    return false;
  }

  size_t i = 0;
  while (literal[i] != '\0') {
    if (i >= text_cap || text[i] != literal[i]) {
      return false;
    }
    ++i;
  }
  return i < text_cap && text[i] == '\0';
}

void knf_stft_config_default(knf_stft_config *cfg) {
  if (cfg == nullptr) {
    return;
  }

  cfg->n_fft = 400;       // 25ms at 16k
  cfg->hop_length = 160;  // 10ms at 16k
  cfg->win_length = 400;
  cfg->center = true;
  memcpy(cfg->pad_mode, "reflect", sizeof("reflect"));
  cfg->normalized = false;
  cfg->window_override.data = nullptr;
  cfg->window_override.size = 0;
  memcpy(cfg->window_type, "povey", sizeof("povey"));
}

static void knf_pad_reflect(const float *data, int32_t n, int32_t pad,
                            float *out) {
  if (n <= 0) {
    memset(out, 0, sizeof(float) * (size_t)(2 * pad));
    return;
  }
  // left pad
  for (int32_t i = 0; i < pad; ++i) {
    int32_t src = pad - i;
    if (src >= n) src = n - 1;
    out[i] = data[src];
  }
  memcpy(out + pad, data, sizeof(float) * n);
  for (int32_t i = 0; i < pad; ++i) {
    int32_t src = n - 2 - i;
    if (src < 0) src = 0;
    out[pad + n + i] = data[src];
  }
}

static void knf_pad_replicate(const float *data, int32_t n, int32_t pad,
                              float *out) {
  if (n <= 0) {
    memset(out, 0, sizeof(float) * (size_t)(2 * pad));
    return;
  }
  for (int32_t i = 0; i < pad; ++i) out[i] = data[0];
  memcpy(out + pad, data, sizeof(float) * n);
  for (int32_t i = 0; i < pad; ++i) out[pad + n + i] = data[n - 1];
}

static void knf_pad_constant(const float *data, int32_t n, int32_t pad,
                             float *out) {
  memset(out, 0, sizeof(float) * (size_t)(n + 2 * pad));
  memcpy(out + pad, data, sizeof(float) * n);
}

[[nodiscard]] bool knf_stft_compute(const knf_stft_config *cfg,
                                    const float *data, int32_t n,
                                    knf_stft_result *out) {
  if (out == nullptr) {
    return false;
  }
  memset(out, 0, sizeof(*out));

  if (cfg == nullptr || data == nullptr || n <= 0 || cfg->n_fft <= 0 ||
      cfg->hop_length <= 0 || cfg->win_length <= 0 ||
      cfg->win_length > cfg->n_fft || (cfg->n_fft & 1) != 0) {
    return false;
  }
  if (cfg->window_override.size < 0 || cfg->window_override.size > cfg->n_fft) {
    return false;
  }
  if (cfg->window_override.size > 0 && cfg->window_override.data == nullptr) {
    return false;
  }

  knf_window window = cfg->window_override;
  bool owns_window = false;
  if (window.size == 0) {
    owns_window =
        knf_make_window(cfg->window_type, cfg->win_length, 0.42f, &window);
    if (!owns_window) return false;
  }

  int32_t pad = cfg->center ? cfg->n_fft / 2 : 0;
  int64_t padded_len64 = (int64_t)n + (int64_t)2 * pad;
  if (padded_len64 <= 0 || padded_len64 > INT32_MAX) {
    if (owns_window) knf_free_window(&window);
    return false;
  }
  int32_t padded_len = (int32_t)padded_len64;
  float *padded =
      (float *)calloc((size_t)(padded_len > 0 ? padded_len : 1), sizeof(float));
  if (padded == nullptr) {
    if (owns_window) knf_free_window(&window);
    return false;
  }

  if (cfg->center) {
    if (knf_fixed_cstr_eq(cfg->pad_mode, sizeof(cfg->pad_mode), "reflect")) {
      knf_pad_reflect(data, n, pad, padded);
    } else if (knf_fixed_cstr_eq(cfg->pad_mode, sizeof(cfg->pad_mode),
                                 "replicate")) {
      knf_pad_replicate(data, n, pad, padded);
    } else {
      knf_pad_constant(data, n, pad, padded);
    }
    data = padded;
    n = padded_len;
  } else {
    memcpy(padded, data, sizeof(float) * n);
  }

  int64_t num_frames = 1 + ((int64_t)n - cfg->n_fft) / cfg->hop_length;
  if (num_frames <= 0 || num_frames > INT32_MAX) {
    free(padded);
    if (owns_window) knf_free_window(&window);
    return false;
  }

  knf_rfft *fft = knf_rfft_create(cfg->n_fft, false);
  if (fft == nullptr) {
    free(padded);
    if (owns_window) knf_free_window(&window);
    return false;
  }

  int32_t bins = cfg->n_fft / 2 + 1;
  if ((size_t)num_frames > SIZE_MAX / (size_t)bins) {
    free(padded);
    knf_rfft_destroy(fft);
    if (owns_window) knf_free_window(&window);
    return false;
  }
  size_t spec_elems = (size_t)num_frames * (size_t)bins;
  out->real = (float *)calloc(spec_elems, sizeof(float));
  out->imag = (float *)calloc(spec_elems, sizeof(float));
  out->num_frames = (int32_t)num_frames;
  out->n_fft = cfg->n_fft;

  if (out->real == nullptr || out->imag == nullptr) {
    free(out->real);
    free(out->imag);
    out->real = nullptr;
    out->imag = nullptr;
    out->num_frames = 0;
    out->n_fft = 0;
    free(padded);
    knf_rfft_destroy(fft);
    if (owns_window) knf_free_window(&window);
    return false;
  }

  float *frame = (float *)calloc((size_t)cfg->n_fft, sizeof(float));
  if (frame == nullptr) {
    free(out->real);
    free(out->imag);
    out->real = nullptr;
    out->imag = nullptr;
    out->num_frames = 0;
    out->n_fft = 0;
    free(padded);
    knf_rfft_destroy(fft);
    if (owns_window) knf_free_window(&window);
    return false;
  }
  for (int32_t i = 0; i < out->num_frames; ++i) {
    memcpy(frame, data + i * cfg->hop_length, sizeof(float) * cfg->n_fft);
    if (window.size > 0) {
      knf_apply_window(&window, frame);
    }
    if (!knf_rfft_compute(fft, frame)) {
      free(frame);
      free(padded);
      knf_rfft_destroy(fft);
      knf_stft_result_free(out);
      if (owns_window) knf_free_window(&window);
      return false;
    }
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
  if (owns_window) knf_free_window(&window);
  return true;
}

void knf_stft_result_free(knf_stft_result *res) {
  if (res == nullptr) return;
  free(res->real);
  free(res->imag);
  res->real = nullptr;
  res->imag = nullptr;
  res->num_frames = 0;
  res->n_fft = 0;
}

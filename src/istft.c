// Inverse STFT implementation in C.

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "kaldi-native-fbank/feature-window.h"
#include "kaldi-native-fbank/istft.h"
#include "kaldi-native-fbank/log.h"
#include "kaldi-native-fbank/rfft.h"

void knf_istft_config_default(knf_istft_config *cfg) {
  cfg->n_fft = 400;
  cfg->hop_length = 160;
  cfg->win_length = 400;
  strncpy(cfg->window_type, "povey", sizeof(cfg->window_type));
  cfg->window = nullptr;
  cfg->window_size = 0;
  cfg->center = true;
  cfg->normalized = false;
}

[[nodiscard]] bool knf_istft_compute(const knf_istft_config *cfg,
                                     const knf_stft_result *stft,
                                     float **out_samples,
                                     int32_t *num_samples) {
  KNF_CHECK(cfg != nullptr);
  KNF_CHECK(stft != nullptr);
  int32_t n_fft = cfg->n_fft;
  int32_t hop = cfg->hop_length;
  int32_t frames = stft->num_frames;
  int32_t bins = n_fft / 2 + 1;

  if (frames <= 0 || n_fft <= 0)
    return false;

  int32_t total = n_fft + (frames - 1) * hop;
  float *samples = (float *)calloc((size_t)total, sizeof(float));
  float *denom = (float *)calloc((size_t)total, sizeof(float));
  if (samples == nullptr || denom == nullptr) {
    free(samples);
    free(denom);
    return false;
  }

  knf_window window = {nullptr, 0};
  bool owns_window = false;
  if (cfg->window != nullptr && cfg->window_size > 0) {
    window.data = cfg->window;
    window.size = cfg->window_size;
  } else {
    owns_window =
        knf_make_window(cfg->window_type, cfg->win_length, 0.42f, &window);
    if (!owns_window) {
      free(samples);
      free(denom);
      return false;
    }
  }

  knf_rfft *fft = knf_rfft_create(n_fft, true);
  if (fft == nullptr) {
    if (owns_window)
      knf_free_window(&window);
    free(samples);
    free(denom);
    return false;
  }

  float *frame = (float *)calloc((size_t)n_fft, sizeof(float));
  if (frame == nullptr) {
    if (owns_window)
      knf_free_window(&window);
    free(samples);
    free(denom);
    knf_rfft_destroy(fft);
    return false;
  }
  float inv_n = 1.0f / (float)n_fft;
  float pre_scale = cfg->normalized ? sqrtf((float)n_fft) : 1.0f;

  for (int32_t i = 0; i < frames; ++i) {
    const float *real = stft->real + i * bins;
    const float *imag = stft->imag + i * bins;

    frame[0] = real[0] * pre_scale;
    frame[1] = real[n_fft / 2] * pre_scale;
    for (int32_t k = 1; k < n_fft / 2; ++k) {
      frame[2 * k] = real[k] * pre_scale;
      frame[2 * k + 1] = imag[k] * pre_scale;
    }

    knf_rfft_compute(fft, frame);
    for (int32_t k = 0; k < n_fft; ++k) {
      frame[k] *= inv_n;
    }

    if (window.size > 0) {
      for (int32_t k = 0; k < n_fft && k < window.size; ++k) {
        frame[k] *= window.data[k];
      }
    }

    int32_t start = i * hop;
    for (int32_t k = 0; k < n_fft; ++k) {
      samples[start + k] += frame[k];
    }
  }

  if (window.size > 0) {
    for (int32_t i = 0; i < frames; ++i) {
      int32_t start = i * hop;
      for (int32_t k = 0; k < n_fft && k < window.size; ++k) {
        float w = window.data[k];
        denom[start + k] += w * w;
      }
    }
  } else {
    for (int32_t i = 0; i < total; ++i)
      denom[i] = 1.0f;
  }

  for (int32_t i = 0; i < total; ++i) {
    if (denom[i] != 0.0f)
      samples[i] /= denom[i];
  }

  if (cfg->center) {
    int32_t cut = n_fft / 2;
    int32_t trimmed = total - 2 * cut;
    if (trimmed < 0)
      trimmed = 0;
    size_t alloc = trimmed > 0 ? (size_t)trimmed : 1;
    float *centered = (float *)calloc(alloc, sizeof(float));
    if (centered == nullptr) {
      free(frame);
      free(samples);
      free(denom);
      knf_rfft_destroy(fft);
      if (owns_window)
        knf_free_window(&window);
      return false;
    }
    memcpy(centered, samples + cut, sizeof(float) * (size_t)trimmed);
    free(samples);
    samples = centered;
    total = trimmed;
  }

  free(frame);
  free(denom);
  knf_rfft_destroy(fft);
  if (owns_window)
    knf_free_window(&window);

  *out_samples = samples;
  *num_samples = total;
  return true;
}

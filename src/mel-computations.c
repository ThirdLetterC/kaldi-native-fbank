// Mel filter bank implementation in C23.

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "mel-computations.h"

static float knf_mel_scale(float freq) {
  return 1127.0f * logf(1.0f + freq / 700.0f);
}

static float knf_inverse_mel_scale(float mel) {
  return 700.0f * (expf(mel / 1127.0f) - 1.0f);
}

void knf_mel_opts_default(knf_mel_opts *opts) {
  opts->num_bins = 25;
  opts->low_freq = 20.0f;
  opts->high_freq = 0.0f;
  opts->vtln_low = 100.0f;
  opts->vtln_high = -500.0f;
  opts->htk_mode = false;
  opts->is_librosa = false;
  opts->use_slaney_mel_scale = true;
  strncpy(opts->norm, "slaney", sizeof(opts->norm));
  opts->floor_to_int_bin = false;
  opts->debug_mel = false;
}

static float knf_vtln_warp(float vtln_low, float vtln_high, float low_freq,
                           float high_freq, float vtln_warp, float freq) {
  if (freq < low_freq || freq > high_freq)
    return freq;
  float l = vtln_low * fmaxf(1.0f, vtln_warp);
  float h = vtln_high * fminf(1.0f, vtln_warp);
  float scale = 1.0f / vtln_warp;
  float Fl = scale * l;
  float Fh = scale * h;
  float scale_left = (Fl - low_freq) / (l - low_freq);
  float scale_right = (high_freq - Fh) / (high_freq - h);
  if (freq < l)
    return low_freq + scale_left * (freq - low_freq);
  if (freq < h)
    return scale * freq;
  return high_freq + scale_right * (freq - high_freq);
}

static float knf_vtln_warp_mel(float vtln_low, float vtln_high, float low_freq,
                               float high_freq, float vtln_warp, float mel) {
  float freq = knf_inverse_mel_scale(mel);
  float warped =
      knf_vtln_warp(vtln_low, vtln_high, low_freq, high_freq, vtln_warp, freq);
  return knf_mel_scale(warped);
}

static bool knf_init_weights(const knf_mel_opts *opts,
                             const knf_frame_opts *frame_opts, float vtln_warp,
                             knf_mel_banks *banks) {
  float sample_freq = frame_opts->samp_freq;
  int32_t window_length_padded = knf_padded_window_size(frame_opts);
  KNF_CHECK_EQ(window_length_padded % 2, 0);
  int32_t num_fft_bins = window_length_padded / 2;
  float nyquist = 0.5f * sample_freq;

  float low_freq = opts->low_freq;
  float high_freq =
      opts->high_freq > 0.0f ? opts->high_freq : nyquist + opts->high_freq;

  if (low_freq < 0.0f || low_freq >= nyquist || high_freq <= 0.0f ||
      high_freq > nyquist || high_freq <= low_freq) {
    knf_fail("mel_option", __FILE__, __func__, __LINE__,
             "invalid low/high freq");
  }

  float fft_bin_width = sample_freq / window_length_padded;
  float mel_low = knf_mel_scale(low_freq);
  float mel_high = knf_mel_scale(high_freq);
  float mel_delta = (mel_high - mel_low) / (opts->num_bins + 1);

  float vtln_low = opts->vtln_low;
  float vtln_high =
      opts->vtln_high < 0.0f ? opts->vtln_high + nyquist : opts->vtln_high;

  banks->num_bins = opts->num_bins;
  banks->num_fft_bins = num_fft_bins;
  banks->weights =
      (float *)calloc((size_t)opts->num_bins * num_fft_bins, sizeof(float));
  if (banks->weights == nullptr) {
    return false;
  }

  for (int32_t bin = 0; bin < opts->num_bins; ++bin) {
    float left_mel = mel_low + bin * mel_delta;
    float center_mel = mel_low + (bin + 1) * mel_delta;
    float right_mel = mel_low + (bin + 2) * mel_delta;

    if (vtln_warp != 1.0f) {
      left_mel = knf_vtln_warp_mel(vtln_low, vtln_high, low_freq, high_freq,
                                   vtln_warp, left_mel);
      center_mel = knf_vtln_warp_mel(vtln_low, vtln_high, low_freq, high_freq,
                                     vtln_warp, center_mel);
      right_mel = knf_vtln_warp_mel(vtln_low, vtln_high, low_freq, high_freq,
                                    vtln_warp, right_mel);
    }

    int32_t first = -1, last = -1;
    for (int32_t i = 0; i < num_fft_bins; ++i) {
      float freq = fft_bin_width * i;
      float mel = knf_mel_scale(freq);
      float weight = 0.0f;
      if (mel > left_mel && mel < right_mel) {
        if (mel <= center_mel) {
          weight = (mel - left_mel) / (center_mel - left_mel);
        } else {
          weight = (right_mel - mel) / (right_mel - center_mel);
        }
      }
      if (weight != 0.0f) {
        if (first == -1)
          first = i;
        last = i;
        banks->weights[bin * num_fft_bins + i] = weight;
      }
    }
    KNF_CHECK(first != -1 && last != -1);
  }
  return true;
}

knf_mel_banks *knf_mel_banks_create(const knf_mel_opts *opts,
                                    const knf_frame_opts *frame_opts,
                                    float vtln_warp) {
  auto banks = (knf_mel_banks *)calloc(1, sizeof(knf_mel_banks));
  if (banks == nullptr)
    return nullptr;
  if (!knf_init_weights(opts, frame_opts, vtln_warp, banks)) {
    free(banks);
    return nullptr;
  }
  return banks;
}

void knf_mel_banks_destroy(knf_mel_banks *banks) {
  if (banks == nullptr)
    return;
  free(banks->weights);
  banks->weights = nullptr;
  free(banks);
}

void knf_mel_compute(const knf_mel_banks *banks, const float *fft_energies,
                     float *mel_energies_out) {
  int32_t num_bins = banks->num_bins;
  int32_t cols = banks->num_fft_bins;
  for (int32_t r = 0; r < num_bins; ++r) {
    float sum = 0.0f;
    const float *w = banks->weights + r * cols;
    for (int32_t c = 0; c < cols; ++c) {
      sum += w[c] * fft_energies[c];
    }
    mel_energies_out[r] = sum;
  }
}

// MFCC computation in C23.

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "kaldi-native-fbank/feature-functions.h"
#include "kaldi-native-fbank/feature-mfcc.h"
#include "kaldi-native-fbank/kaldi-math.h"

static void knf_compute_dct(int32_t rows, int32_t cols, float *out) {
  float norm0 = sqrtf(1.0f / cols);
  for (int32_t i = 0; i < cols; ++i)
    out[i] = norm0;

  float norm = sqrtf(2.0f / cols);
  for (int32_t k = 1; k < rows; ++k) {
    for (int32_t n = 0; n < cols; ++n) {
      out[k * cols + n] = norm * cosf((float)KNF_PI / cols * (n + 0.5f) * k);
    }
  }
}

static void knf_compute_lifter(float lifter, int32_t num_ceps, float *out) {
  for (int32_t i = 0; i < num_ceps; ++i) {
    out[i] = 1.0f + 0.5f * lifter * sinf((float)KNF_PI * i / lifter);
  }
}

void knf_mfcc_opts_default(knf_mfcc_opts *opts) {
  knf_frame_opts_default(&opts->frame_opts);
  knf_mel_opts_default(&opts->mel_opts);
  opts->num_ceps = 13;
  opts->cepstral_lifter = 22.0f;
  opts->use_energy = true;
  opts->raw_energy = true;
  opts->htk_compat = false;
  opts->energy_floor = 0.0f;
}

[[nodiscard]] bool knf_mfcc_computer_create(const knf_mfcc_opts *opts,
                                            knf_mfcc_computer *out) {
  memset(out, 0, sizeof(*out));
  out->opts = *opts;
  int32_t n_fft = knf_padded_window_size(&opts->frame_opts);
  out->rfft = knf_rfft_create(n_fft, false);
  if (!out->rfft)
    return false;
  out->mel_banks =
      knf_mel_banks_create(&opts->mel_opts, &opts->frame_opts, 1.0f);
  if (!out->mel_banks) {
    knf_rfft_destroy(out->rfft);
    return false;
  }
  out->mel_energies =
      (float *)calloc((size_t)opts->mel_opts.num_bins, sizeof(float));
  if (out->mel_energies == nullptr) {
    knf_mfcc_computer_destroy(out);
    return false;
  }
  out->dct_matrix = (float *)calloc(
      (size_t)opts->num_ceps * opts->mel_opts.num_bins, sizeof(float));
  if (out->dct_matrix == nullptr) {
    knf_mfcc_computer_destroy(out);
    return false;
  }
  knf_compute_dct(opts->num_ceps, opts->mel_opts.num_bins, out->dct_matrix);
  if (opts->cepstral_lifter != 0.0f) {
    out->lifter_coeffs = (float *)calloc((size_t)opts->num_ceps, sizeof(float));
    if (out->lifter_coeffs == nullptr) {
      knf_mfcc_computer_destroy(out);
      return false;
    }
    knf_compute_lifter(opts->cepstral_lifter, opts->num_ceps,
                       out->lifter_coeffs);
  }
  out->log_energy_floor =
      opts->energy_floor > 0.0f ? logf(opts->energy_floor) : -1e10f;
  return true;
}

void knf_mfcc_computer_destroy(knf_mfcc_computer *c) {
  if (!c)
    return;
  knf_rfft_destroy(c->rfft);
  c->rfft = nullptr;
  knf_mel_banks_destroy(c->mel_banks);
  c->mel_banks = nullptr;
  free(c->mel_energies);
  c->mel_energies = nullptr;
  free(c->dct_matrix);
  c->dct_matrix = nullptr;
  free(c->lifter_coeffs);
  c->lifter_coeffs = nullptr;
}

const knf_frame_opts *knf_mfcc_frame_opts(const knf_mfcc_computer *c) {
  return &c->opts.frame_opts;
}

int32_t knf_mfcc_dim(const knf_mfcc_computer *c) { return c->opts.num_ceps; }

bool knf_mfcc_need_raw_log_energy(const knf_mfcc_computer *c) {
  return c->opts.use_energy && c->opts.raw_energy;
}

void knf_mfcc_compute(knf_mfcc_computer *c, float signal_raw_log_energy,
                      [[maybe_unused]] float vtln_warp, float *signal_frame,
                      float *feature) {
  const knf_mfcc_opts *opts = &c->opts;
  int32_t padded = knf_padded_window_size(&opts->frame_opts);
  if (opts->use_energy && !opts->raw_energy) {
    float energy = knf_inner_product(signal_frame, signal_frame, padded);
    if (energy < 1e-20f)
      energy = 1e-20f;
    signal_raw_log_energy = logf(energy);
  }

  knf_rfft_compute(c->rfft, signal_frame);
  knf_compute_power_spectrum(signal_frame, padded);
  knf_mel_compute(c->mel_banks, signal_frame, c->mel_energies);
  for (int32_t i = 0; i < opts->mel_opts.num_bins; ++i) {
    float v = c->mel_energies[i];
    if (v < 1e-20f)
      v = 1e-20f;
    c->mel_energies[i] = logf(v);
  }

  for (int32_t i = 0; i < opts->num_ceps; ++i) {
    feature[i] = knf_inner_product(c->dct_matrix + i * opts->mel_opts.num_bins,
                                   c->mel_energies, opts->mel_opts.num_bins);
  }
  if (opts->cepstral_lifter != 0.0f) {
    for (int32_t i = 0; i < opts->num_ceps; ++i) {
      feature[i] *= c->lifter_coeffs ? c->lifter_coeffs[i] : 1.0f;
    }
  }
  if (opts->use_energy) {
    if (opts->energy_floor > 0.0f &&
        signal_raw_log_energy < c->log_energy_floor) {
      signal_raw_log_energy = c->log_energy_floor;
    }
    feature[0] = signal_raw_log_energy;
  }
  if (opts->htk_compat) {
    float energy = feature[0];
    for (int32_t i = 0; i < opts->num_ceps - 1; ++i) {
      feature[i] = feature[i + 1];
    }
    if (!opts->use_energy) {
      energy *= (float)KNF_SQRT2;
    }
    feature[opts->num_ceps - 1] = energy;
  }
}

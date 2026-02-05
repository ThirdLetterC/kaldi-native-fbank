// Fbank computation in C23.

#include <math.h>
#include <string.h>

#include "kaldi-native-fbank/feature-fbank.h"
#include "kaldi-native-fbank/feature-functions.h"
#include "kaldi-native-fbank/kaldi-math.h"

void knf_fbank_opts_default(knf_fbank_opts *opts) {
  knf_frame_opts_default(&opts->frame_opts);
  knf_mel_opts_default(&opts->mel_opts);
  opts->use_energy = true;
  opts->raw_energy = true;
  opts->htk_compat = false;
  opts->energy_floor = 0.0f;
  opts->use_log_fbank = true;
  opts->use_power = true;
}

[[nodiscard]] bool knf_fbank_computer_create(const knf_fbank_opts *opts,
                                             knf_fbank_computer *out) {
  memset(out, 0, sizeof(*out));
  out->opts = *opts;
  if (opts->energy_floor > 0.0f) {
    out->log_energy_floor = logf(opts->energy_floor);
  } else {
    out->log_energy_floor = -1e10f;
  }
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
  return true;
}

void knf_fbank_computer_destroy(knf_fbank_computer *c) {
  if (!c)
    return;
  knf_rfft_destroy(c->rfft);
  knf_mel_banks_destroy(c->mel_banks);
}

const knf_frame_opts *knf_fbank_frame_opts(const knf_fbank_computer *c) {
  return &c->opts.frame_opts;
}

int32_t knf_fbank_dim(const knf_fbank_computer *c) {
  return c->opts.mel_opts.num_bins + (c->opts.use_energy ? 1 : 0);
}

bool knf_fbank_need_raw_log_energy(const knf_fbank_computer *c) {
  return c->opts.use_energy && c->opts.raw_energy;
}

void knf_fbank_compute(knf_fbank_computer *c, float signal_raw_log_energy,
                       [[maybe_unused]] float vtln_warp, float *signal_frame,
                       float *feature) {
  const knf_fbank_opts *opts = &c->opts;
  int32_t padded = knf_padded_window_size(&opts->frame_opts);

  if (opts->use_energy && !opts->raw_energy) {
    float energy = knf_inner_product(signal_frame, signal_frame, padded);
    if (energy < 1e-20f)
      energy = 1e-20f;
    signal_raw_log_energy = logf(energy);
  }

  knf_rfft_compute(c->rfft, signal_frame);
  knf_compute_power_spectrum(signal_frame, padded);
  if (!opts->use_power) {
    knf_sqrt_inplace(signal_frame, padded / 2 + 1);
  }

  int32_t mel_offset = (opts->use_energy && !opts->htk_compat) ? 1 : 0;
  float *mel_out = feature + mel_offset;
  knf_mel_compute(c->mel_banks, signal_frame, mel_out);

  if (opts->use_log_fbank) {
    for (int32_t i = 0; i < opts->mel_opts.num_bins; ++i) {
      float v = mel_out[i];
      if (v < 1e-20f)
        v = 1e-20f;
      mel_out[i] = logf(v);
    }
  }

  if (opts->use_energy) {
    if (opts->energy_floor > 0.0f &&
        signal_raw_log_energy < c->log_energy_floor) {
      signal_raw_log_energy = c->log_energy_floor;
    }
    int32_t energy_index = opts->htk_compat ? opts->mel_opts.num_bins : 0;
    feature[energy_index] = signal_raw_log_energy;
  }
}

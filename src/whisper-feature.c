#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "feature-functions.h"
#include "whisper-feature.h"

void knf_whisper_opts_default(knf_whisper_opts *opts) {
  knf_frame_opts_default(&opts->frame_opts);
  opts->frame_opts.samp_freq = 16000.0f;
  opts->frame_opts.frame_shift_ms = 10.0f;
  opts->frame_opts.frame_length_ms = 25.0f;
  opts->frame_opts.dither = 0.0f;
  opts->frame_opts.preemph_coeff = 0.0f;
  opts->frame_opts.remove_dc_offset = false;
  strncpy(opts->frame_opts.window_type, "hann",
          sizeof(opts->frame_opts.window_type));
  opts->frame_opts.round_to_power_of_two = false;
  opts->frame_opts.snip_edges = false;
  opts->dim = 80;
}

bool knf_whisper_computer_create(const knf_whisper_opts *opts,
                                 knf_whisper_computer *out) {
  memset(out, 0, sizeof(*out));
  out->opts = *opts;
  knf_mel_opts mel_opts;
  knf_mel_opts_default(&mel_opts);
  mel_opts.num_bins = opts->dim;
  mel_opts.low_freq = 0.0f;
  mel_opts.is_librosa = true;
  mel_opts.use_slaney_mel_scale = true;
  strncpy(mel_opts.norm, "slaney", sizeof(mel_opts.norm));

  out->rfft = knf_rfft_create(knf_window_size(&opts->frame_opts), false);
  if (!out->rfft)
    return false;
  out->mel_banks = knf_mel_banks_create(&mel_opts, &opts->frame_opts, 1.0f);
  if (!out->mel_banks) {
    knf_rfft_destroy(out->rfft);
    return false;
  }
  return true;
}

void knf_whisper_computer_destroy(knf_whisper_computer *c) {
  if (!c)
    return;
  knf_rfft_destroy(c->rfft);
  knf_mel_banks_destroy(c->mel_banks);
}

const knf_frame_opts *knf_whisper_frame_opts(const knf_whisper_computer *c) {
  return &c->opts.frame_opts;
}

int32_t knf_whisper_dim(const knf_whisper_computer *c) { return c->opts.dim; }
int knf_whisper_need_raw_log_energy(const knf_whisper_computer *c) {
  (void)c;
  return 0;
}

void knf_whisper_compute(knf_whisper_computer *c, float signal_raw_log_energy,
                         float vtln_warp, float *signal_frame, float *feature) {
  (void)signal_raw_log_energy;
  (void)vtln_warp;
  int32_t n_fft = knf_window_size(&c->opts.frame_opts);
  knf_rfft_compute(c->rfft, signal_frame);
  knf_compute_power_spectrum(signal_frame, n_fft);
  knf_mel_compute(c->mel_banks, signal_frame, feature);
}

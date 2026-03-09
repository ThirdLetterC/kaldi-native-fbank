// Raw audio frame passthrough in C.

#include <string.h>

#include "kaldi-native-fbank/feature-raw-audio-samples.h"

void knf_raw_audio_opts_default(knf_raw_audio_opts *opts) {
  knf_frame_opts_default(&opts->frame_opts);
}

[[nodiscard]] bool knf_raw_audio_computer_create(const knf_raw_audio_opts *opts,
                                                 knf_raw_audio_computer *out) {
  if (opts == nullptr || out == nullptr) {
    return false;
  }
  out->opts = *opts;
  return true;
}

void knf_raw_audio_computer_destroy(
    [[maybe_unused]] knf_raw_audio_computer *c) {}

const knf_frame_opts *knf_raw_audio_frame_opts(
    const knf_raw_audio_computer *c) {
  if (c == nullptr) {
    return nullptr;
  }
  return &c->opts.frame_opts;
}

int32_t knf_raw_audio_dim(const knf_raw_audio_computer *c) {
  if (c == nullptr) {
    return 0;
  }
  return knf_padded_window_size(&c->opts.frame_opts);
}

bool knf_raw_audio_need_raw_log_energy(
    [[maybe_unused]] const knf_raw_audio_computer *c) {
  return false;
}

void knf_raw_audio_compute(knf_raw_audio_computer *c,
                           [[maybe_unused]] float signal_raw_log_energy,
                           [[maybe_unused]] float vtln_warp,
                           float *signal_frame, float *feature) {
  if (c == nullptr || signal_frame == nullptr || feature == nullptr) {
    return;
  }
  int32_t dim = knf_padded_window_size(&c->opts.frame_opts);
  if (dim <= 0) {
    return;
  }
  memcpy(feature, signal_frame, sizeof(float) * dim);
}

// Raw audio frame passthrough in C23.
#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_RAW_AUDIO_SAMPLES_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_RAW_AUDIO_SAMPLES_H_

#include <stdint.h>

#include "kaldi-native-fbank/feature-window.h"

typedef struct {
  knf_frame_opts frame_opts;
} knf_raw_audio_opts;

typedef struct {
  knf_raw_audio_opts opts;
} knf_raw_audio_computer;

void knf_raw_audio_opts_default(knf_raw_audio_opts *opts);
[[nodiscard]] bool knf_raw_audio_computer_create(const knf_raw_audio_opts *opts,
                                                 knf_raw_audio_computer *out);
void knf_raw_audio_computer_destroy(knf_raw_audio_computer *c);
const knf_frame_opts *knf_raw_audio_frame_opts(const knf_raw_audio_computer *c);
int32_t knf_raw_audio_dim(const knf_raw_audio_computer *c);
bool knf_raw_audio_need_raw_log_energy(const knf_raw_audio_computer *c);
void knf_raw_audio_compute(knf_raw_audio_computer *c,
                           float signal_raw_log_energy, float vtln_warp,
                           float *signal_frame, float *feature);

#endif // KALDI_NATIVE_FBANK_CSRC_FEATURE_RAW_AUDIO_SAMPLES_H_

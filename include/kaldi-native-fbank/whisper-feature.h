// Whisper-style mel feature computation in C23.
#ifndef KALDI_NATIVE_FBANK_CSRC_WHISPER_FEATURE_H_
#define KALDI_NATIVE_FBANK_CSRC_WHISPER_FEATURE_H_

#include <stdint.h>

#include "kaldi-native-fbank/feature-window.h"
#include "kaldi-native-fbank/mel-computations.h"
#include "kaldi-native-fbank/rfft.h"

typedef struct {
  knf_frame_opts frame_opts;
  int32_t dim;
} knf_whisper_opts;

typedef struct {
  knf_whisper_opts opts;
  knf_mel_banks *mel_banks;
  knf_rfft *rfft;
} knf_whisper_computer;

void knf_whisper_opts_default(knf_whisper_opts *opts);
[[nodiscard]] bool knf_whisper_computer_create(const knf_whisper_opts *opts,
                                               knf_whisper_computer *out);
void knf_whisper_computer_destroy(knf_whisper_computer *c);
const knf_frame_opts *knf_whisper_frame_opts(const knf_whisper_computer *c);
int32_t knf_whisper_dim(const knf_whisper_computer *c);
bool knf_whisper_need_raw_log_energy(const knf_whisper_computer *c);
void knf_whisper_compute(knf_whisper_computer *c, float signal_raw_log_energy,
                         float vtln_warp, float *signal_frame, float *feature);

#endif // KALDI_NATIVE_FBANK_CSRC_WHISPER_FEATURE_H_

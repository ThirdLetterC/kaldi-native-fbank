// Fbank computation in C23.
#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_FBANK_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_FBANK_H_

#include <stdint.h>

#include "feature-window.h"
#include "mel-computations.h"
#include "rfft.h"

typedef struct {
  knf_frame_opts frame_opts;
  knf_mel_opts mel_opts;
  bool use_energy;
  bool raw_energy;
  bool htk_compat;
  float energy_floor;
  bool use_log_fbank;
  bool use_power;
} knf_fbank_opts;

typedef struct {
  knf_fbank_opts opts;
  knf_rfft *rfft;
  knf_mel_banks *mel_banks;
  float log_energy_floor;
} knf_fbank_computer;

void knf_fbank_opts_default(knf_fbank_opts *opts);
[[nodiscard]] bool knf_fbank_computer_create(const knf_fbank_opts *opts,
                                             knf_fbank_computer *out);
void knf_fbank_computer_destroy(knf_fbank_computer *c);
const knf_frame_opts *knf_fbank_frame_opts(const knf_fbank_computer *c);
int32_t knf_fbank_dim(const knf_fbank_computer *c);
bool knf_fbank_need_raw_log_energy(const knf_fbank_computer *c);
void knf_fbank_compute(knf_fbank_computer *c, float signal_raw_log_energy,
                       float vtln_warp, float *signal_frame, float *feature);

#endif // KALDI_NATIVE_FBANK_CSRC_FEATURE_FBANK_H_

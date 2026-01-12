// MFCC computation in C23.
#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_MFCC_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_MFCC_H_

#include <stdint.h>

#include "feature-window.h"
#include "mel-computations.h"
#include "rfft.h"

typedef struct {
  knf_frame_opts frame_opts;
  knf_mel_opts mel_opts;
  int32_t num_ceps;
  float cepstral_lifter;
  bool use_energy;
  bool raw_energy;
  bool htk_compat;
  float energy_floor;
} knf_mfcc_opts;

typedef struct {
  knf_mfcc_opts opts;
  knf_rfft *rfft;
  knf_mel_banks *mel_banks;
  float *mel_energies;
  float *dct_matrix;
  float *lifter_coeffs;
  float log_energy_floor;
} knf_mfcc_computer;

void knf_mfcc_opts_default(knf_mfcc_opts *opts);
[[nodiscard]] bool knf_mfcc_computer_create(const knf_mfcc_opts *opts,
                                            knf_mfcc_computer *out);
void knf_mfcc_computer_destroy(knf_mfcc_computer *c);
const knf_frame_opts *knf_mfcc_frame_opts(const knf_mfcc_computer *c);
int32_t knf_mfcc_dim(const knf_mfcc_computer *c);
bool knf_mfcc_need_raw_log_energy(const knf_mfcc_computer *c);
void knf_mfcc_compute(knf_mfcc_computer *c, float signal_raw_log_energy,
                      float vtln_warp, float *signal_frame, float *feature);

#endif // KALDI_NATIVE_FBANK_CSRC_FEATURE_MFCC_H_

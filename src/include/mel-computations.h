// Mel filter bank creation in C23.
#ifndef KALDI_NATIVE_FBANK_CSRC_MEL_COMPUTATIONS_H_
#define KALDI_NATIVE_FBANK_CSRC_MEL_COMPUTATIONS_H_

#include <stdint.h>

#include "feature-window.h"

typedef struct {
  int32_t num_bins;
  float low_freq;
  float high_freq;
  float vtln_low;
  float vtln_high;
  bool htk_mode;
  bool is_librosa;
  bool use_slaney_mel_scale;
  char norm[8]; // "", "slaney"
  bool floor_to_int_bin;
  bool debug_mel;
} knf_mel_opts;

typedef struct {
  int32_t num_bins;
  int32_t num_fft_bins; // equals padded_window/2
  float *weights;       // flattened [num_bins][num_fft_bins]
} knf_mel_banks;

void knf_mel_opts_default(knf_mel_opts *opts);
[[nodiscard]] knf_mel_banks *
knf_mel_banks_create(const knf_mel_opts *opts, const knf_frame_opts *frame_opts,
                     float vtln_warp);
void knf_mel_banks_destroy(knf_mel_banks *banks);
void knf_mel_compute(const knf_mel_banks *banks, const float *fft_energies,
                     float *mel_energies_out);

#endif // KALDI_NATIVE_FBANK_CSRC_MEL_COMPUTATIONS_H_

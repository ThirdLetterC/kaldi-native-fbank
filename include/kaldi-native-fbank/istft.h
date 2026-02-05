// Inverse STFT implemented in C23.
#ifndef KALDI_NATIVE_FBANK_CSRC_ISTFT_H_
#define KALDI_NATIVE_FBANK_CSRC_ISTFT_H_

#include <stdint.h>

#include "kaldi-native-fbank/stft.h"

typedef struct {
  int32_t n_fft;
  int32_t hop_length;
  int32_t win_length;
  char window_type[16];
  float *window; // optional override, length win_length
  int32_t window_size;
  bool center;
  bool normalized;
} knf_istft_config;

void knf_istft_config_default(knf_istft_config *cfg);
[[nodiscard]] bool knf_istft_compute(const knf_istft_config *cfg,
                                     const knf_stft_result *stft,
                                     float **out_samples, int32_t *num_samples);

#endif // KALDI_NATIVE_FBANK_CSRC_ISTFT_H_

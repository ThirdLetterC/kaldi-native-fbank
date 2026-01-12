// Short-time Fourier Transform in C23.
#ifndef KALDI_NATIVE_FBANK_CSRC_STFT_H_
#define KALDI_NATIVE_FBANK_CSRC_STFT_H_

#include <stdint.h>

#include "feature-window.h"

typedef struct {
  int32_t n_fft;
  int32_t hop_length;
  int32_t win_length;
  bool center;
  char pad_mode[16]; // reflect, constant, replicate
  bool normalized;
  knf_window window_override; // optional; if size>0 overrides window_type
  char window_type[16];
} knf_stft_config;

typedef struct {
  float *real; // size num_frames*(n_fft/2+1)
  float *imag; // size num_frames*(n_fft/2+1)
  int32_t num_frames;
  int32_t n_fft;
} knf_stft_result;

void knf_stft_config_default(knf_stft_config *cfg);
[[nodiscard]] bool knf_stft_compute(const knf_stft_config *cfg,
                                    const float *data, int32_t n,
                                    knf_stft_result *out);
void knf_stft_result_free(knf_stft_result *res);

#endif // KALDI_NATIVE_FBANK_CSRC_STFT_H_

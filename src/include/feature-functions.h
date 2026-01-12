// Utility FFT helpers in C.
#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_FUNCTIONS_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_FUNCTIONS_H_

#include <stdint.h>

void knf_compute_power_spectrum(float *complex_fft, int32_t dim);

#endif // KALDI_NATIVE_FBANK_CSRC_FEATURE_FUNCTIONS_H_

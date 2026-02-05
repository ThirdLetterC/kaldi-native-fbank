// include/rfft.h
// Simple real FFT wrapper backed by pocketfft for C23 build.

#ifndef KALDI_NATIVE_FBANK_CSRC_RFFT_H_
#define KALDI_NATIVE_FBANK_CSRC_RFFT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t n;
  bool inverse;
  float scale;
  void *plan;
  float *work; // size n
} knf_rfft;

[[nodiscard]] knf_rfft *knf_rfft_create(int32_t n, bool inverse);
void knf_rfft_destroy(knf_rfft *fft);
void knf_rfft_compute(knf_rfft *fft, float *in_out);

#ifdef __cplusplus
}
#endif

#endif // KALDI_NATIVE_FBANK_CSRC_RFFT_H_

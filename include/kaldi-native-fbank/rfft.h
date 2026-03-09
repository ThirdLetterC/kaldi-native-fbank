// include/rfft.h
// Simple real FFT wrapper backed by pocketfft for C23 build.

#pragma once

#include <stdint.h>

typedef struct {
  int32_t n;
  bool inverse;
  float scale;
  void *plan;
  float *work;  // size n
} knf_rfft;

[[nodiscard]] knf_rfft *knf_rfft_create(int32_t n,
                                        bool inverse);  // Owning pointer, or
                                                        // nullptr.
void knf_rfft_destroy(knf_rfft *fft);
[[nodiscard]] bool knf_rfft_compute(knf_rfft *fft, float *in_out);

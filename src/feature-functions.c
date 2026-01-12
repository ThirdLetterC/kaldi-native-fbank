// Utility helpers for FFT output.
#include "feature-functions.h"

void knf_compute_power_spectrum(float *complex_fft, int32_t dim) {
  int32_t half_dim = dim / 2;
  float first_energy = complex_fft[0] * complex_fft[0];
  float last_energy = complex_fft[1] * complex_fft[1];

  for (int32_t i = 1; i < half_dim; ++i) {
    float real = complex_fft[i * 2];
    float im = complex_fft[i * 2 + 1];
    complex_fft[i] = real * real + im * im;
  }
  complex_fft[0] = first_energy;
  complex_fft[half_dim] = last_energy;
}

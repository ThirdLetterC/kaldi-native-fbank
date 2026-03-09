// Utility FFT helpers in C.
#pragma once

#include <stdint.h>

void knf_compute_power_spectrum(float *complex_fft, int32_t dim);

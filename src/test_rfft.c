#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "rfft.h"

int main() {
  float signal[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  float original[8];
  for (int i = 0; i < 8; ++i)
    original[i] = signal[i];

  knf_rfft *fft = knf_rfft_create(8, false);
  assert(fft != nullptr);
  knf_rfft_compute(fft, signal);
  knf_rfft_destroy(fft);

  knf_rfft *ifft = knf_rfft_create(8, true);
  assert(ifft != nullptr);
  knf_rfft_compute(ifft, signal);
  knf_rfft_destroy(ifft);

  for (int i = 0; i < 8; ++i) {
    float expected = original[i] * 8.0f;
    assert(fabsf(signal[i] - expected) < 1e-3f);
  }

  printf("test_rfft passed\n");
  return 0;
}

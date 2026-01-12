#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "feature-window.h"
#include "mel-computations.h"

int main() {
  knf_frame_opts fopts;
  knf_frame_opts_default(&fopts);
  fopts.frame_length_ms = 25.0f;
  fopts.frame_shift_ms = 10.0f;
  knf_mel_opts mopts;
  knf_mel_opts_default(&mopts);
  mopts.num_bins = 10;
  knf_mel_banks *banks = knf_mel_banks_create(&mopts, &fopts, 1.0f);
  assert(banks != nullptr);
  int cols = banks->num_fft_bins;
  float *fft = (float *)calloc(cols, sizeof(float));
  assert(fft != nullptr);
  for (int i = 0; i < cols; ++i)
    fft[i] = (float)i;
  float *out = (float *)calloc(mopts.num_bins, sizeof(float));
  assert(out != nullptr);
  knf_mel_compute(banks, fft, out);
  for (int i = 0; i < mopts.num_bins; ++i) {
    assert(isfinite(out[i]));
  }
  knf_mel_banks_destroy(banks);
  free(fft);
  free(out);
  printf("test_mel_banks passed\n");
  return 0;
}

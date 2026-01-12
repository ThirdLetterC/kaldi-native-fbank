#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "feature-window.h"
#include "whisper-feature.h"

constexpr float KNF_PI = 3.14159265358979323846f;
int main() {
  knf_whisper_opts opts;
  knf_whisper_opts_default(&opts);

  knf_whisper_computer comp;
  assert(knf_whisper_computer_create(&opts, &comp));

  int32_t n = knf_window_size(&opts.frame_opts);
  float *wave = (float *)calloc((size_t)n, sizeof(float));
  assert(wave != nullptr);
  for (int i = 0; i < n; ++i)
    wave[i] = sinf(2.0f * KNF_PI * 300.0f * i / 16000.0f);

  float *feat = (float *)calloc(opts.dim, sizeof(float));
  assert(feat != nullptr);
  knf_whisper_compute(&comp, 0.0f, 1.0f, wave, feat);
  for (int i = 0; i < opts.dim; ++i) {
    assert(isfinite(feat[i]));
  }

  free(feat);
  free(wave);
  knf_whisper_computer_destroy(&comp);
  printf("test_whisper passed\n");
  return 0;
}

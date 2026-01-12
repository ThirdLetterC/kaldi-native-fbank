#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "feature-mfcc.h"
#include "feature-window.h"

constexpr float KNF_PI = 3.14159265358979323846f;
int main() {
  knf_mfcc_opts opts;
  knf_mfcc_opts_default(&opts);
  opts.frame_opts.dither = 0.0f;
  opts.frame_opts.preemph_coeff = 0.0f;

  knf_mfcc_computer comp;
  assert(knf_mfcc_computer_create(&opts, &comp));

  int32_t padded = knf_padded_window_size(&opts.frame_opts);
  float *wave = (float *)calloc((size_t)padded, sizeof(float));
  assert(wave != nullptr);
  for (int i = 0; i < padded; ++i) {
    wave[i] =
        cosf(2.0f * KNF_PI * 220.0f * ((float)i / opts.frame_opts.samp_freq));
  }

  knf_window win;
  assert(knf_make_window_from_opts(&opts.frame_opts, &win));
  float raw_log_energy = 0.0f;
  knf_process_window(&opts.frame_opts, &win, wave, &raw_log_energy);

  float *feat = (float *)calloc(knf_mfcc_dim(&comp), sizeof(float));
  assert(feat != nullptr);
  knf_mfcc_compute(&comp, raw_log_energy, 1.0f, wave, feat);
  for (int i = 0; i < knf_mfcc_dim(&comp); ++i) {
    assert(isfinite(feat[i]));
  }

  free(feat);
  free(wave);
  knf_free_window(&win);
  knf_mfcc_computer_destroy(&comp);
  printf("test_mfcc passed\n");
  return 0;
}

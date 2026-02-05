#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "kaldi-native-fbank/feature-fbank.h"
#include "kaldi-native-fbank/feature-window.h"

constexpr float KNF_PI = 3.14159265358979323846f;

int main() {
  knf_fbank_opts opts;
  knf_fbank_opts_default(&opts);
  opts.frame_opts.dither = 0.0f;
  opts.frame_opts.preemph_coeff = 0.0f;

  knf_fbank_computer comp;
  assert(knf_fbank_computer_create(&opts, &comp));

  int32_t padded = knf_padded_window_size(&opts.frame_opts);
  float *wave = (float *)calloc((size_t)padded, sizeof(float));
  assert(wave != nullptr);
  for (int i = 0; i < padded; ++i) {
    wave[i] =
        sinf(2.0f * KNF_PI * 440.0f * ((float)i / opts.frame_opts.samp_freq));
  }

  knf_window win;
  assert(knf_make_window_from_opts(&opts.frame_opts, &win));
  float raw_log_energy = 0.0f;
  knf_process_window(&opts.frame_opts, &win, wave, &raw_log_energy);

  float *feat = (float *)calloc(knf_fbank_dim(&comp), sizeof(float));
  assert(feat != nullptr);
  knf_fbank_compute(&comp, raw_log_energy, 1.0f, wave, feat);
  for (int i = 0; i < knf_fbank_dim(&comp); ++i) {
    assert(isfinite(feat[i]));
  }

  free(feat);
  free(wave);
  knf_free_window(&win);
  knf_fbank_computer_destroy(&comp);
  printf("test_fbank passed\n");
  return 0;
}

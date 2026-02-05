#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "kaldi-native-fbank/online-feature.h"

constexpr float KNF_PI = 3.14159265358979323846f;
static void fill_wave(float *wave, int n, float freq, float samp_freq) {
  for (int i = 0; i < n; ++i) {
    wave[i] = sinf(2.0f * KNF_PI * freq * ((float)i / samp_freq));
  }
}

int main() {
  knf_fbank_opts fopts;
  knf_fbank_opts_default(&fopts);
  fopts.frame_opts.dither = 0.0f;

  knf_online_feature feat;
  assert(knf_online_fbank_create(&fopts, &feat));

  int n = 3200;
  float *wave = (float *)calloc((size_t)n, sizeof(float));
  if (wave == nullptr) {
    knf_online_feature_destroy(&feat);
    return 1;
  }
  fill_wave(wave, n, 1000.0f, fopts.frame_opts.samp_freq);
  knf_online_accept_waveform(&feat, fopts.frame_opts.samp_freq, wave, n);
  knf_online_input_finished(&feat);

  int32_t ready = knf_online_num_frames_ready(&feat);
  assert(ready > 0);
  const float *frame = knf_online_get_frame(&feat, 0);
  assert(frame != nullptr);
  for (int i = 0; i < knf_fbank_dim((knf_fbank_computer *)feat.computer); ++i) {
    assert(isfinite(frame[i]));
  }

  knf_online_feature_destroy(&feat);
  free(wave);
  printf("test_online passed\n");
  return 0;
}

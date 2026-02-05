#include <assert.h>
#include <stdio.h>

#include "kaldi-native-fbank/feature-window.h"

int main() {
  knf_frame_opts opts;
  knf_frame_opts_default(&opts);
  assert(knf_window_size(&opts) == 400);
  assert(knf_padded_window_size(&opts) == 512);

  knf_window window = {nullptr, 0};
  assert(knf_make_window_from_opts(&opts, &window));
  assert(window.size == 400);

  float sample[400];
  for (int i = 0; i < 400; ++i)
    sample[i] = 1.0f;
  knf_apply_window(&window, sample);
  assert(sample[0] <= 1.0f && sample[0] >= 0.0f);
  knf_free_window(&window);

  float wave[512] = {0};
  for (int i = 0; i < 512; ++i)
    wave[i] = (float)i;
  float window_buf[512];
  float log_energy = 0.0f;
  bool ok = knf_extract_window(0, wave, 512, 0, &opts, nullptr, window_buf,
                               &log_energy);
  assert(ok);
  (void)log_energy;

  printf("test_feature_window passed\n");
  return 0;
}

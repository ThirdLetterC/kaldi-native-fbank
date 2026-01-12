#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "feature-fbank.h"
#include "feature-window.h"
#include "online-feature.h"

static float uniform_neg1_1() {
  return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

int main() {
  const int sample_rate = 16000;
  const int num_seconds = 1;
  const int num_samples = sample_rate * num_seconds;
  const int frames_to_check = 3;

  srand(202250929);

  float *wave = (float *)calloc((size_t)num_samples, sizeof(float));
  if (wave == nullptr) {
    return 1;
  }
  for (int i = 0; i < num_samples; ++i) {
    wave[i] = uniform_neg1_1();
  }

  knf_fbank_opts opts;
  knf_fbank_opts_default(&opts);
  opts.frame_opts.dither = 0.0f;
  opts.frame_opts.window_type[0] = '\0';
  strncpy(opts.frame_opts.window_type, "hann",
          sizeof(opts.frame_opts.window_type));
  opts.mel_opts.num_bins = 23;
  opts.use_energy = false; // match python demo (no energy column)
  opts.raw_energy = false; // not used when use_energy=false

  knf_fbank_computer comp;
  assert(knf_fbank_computer_create(&opts, &comp));

  knf_window win;
  assert(knf_make_window_from_opts(&opts.frame_opts, &win));

  int32_t padded = knf_padded_window_size(&opts.frame_opts);
  float *window_buf = (float *)calloc((size_t)padded, sizeof(float));
  if (window_buf == nullptr) {
    knf_free_window(&win);
    knf_fbank_computer_destroy(&comp);
    free(wave);
    return 1;
  }

  float offline[frames_to_check][32]; // num_bins=23 so 32 is safe
  memset(offline, 0, sizeof(offline));

  for (int frame = 0; frame < frames_to_check; ++frame) {
    float raw_log_energy = 0.0f;
    memset(window_buf, 0, sizeof(float) * padded);
    bool ok = knf_extract_window(0, wave, num_samples, frame, &opts.frame_opts,
                                 &win, window_buf, &raw_log_energy);
    assert(ok);
    knf_fbank_compute(&comp, raw_log_energy, 1.0f, window_buf, offline[frame]);
  }

  knf_online_feature online;
  assert(knf_online_fbank_create(&opts, &online));
  knf_online_accept_waveform(&online, (float)sample_rate, wave, num_samples);
  knf_online_input_finished(&online);
  assert(knf_online_num_frames_ready(&online) >= frames_to_check);

  float tol = 1e-3f;
  for (int frame = 0; frame < frames_to_check; ++frame) {
    const float *on = knf_online_get_frame(&online, frame);
    assert(on != nullptr);
    for (int i = 0; i < opts.mel_opts.num_bins; ++i) {
      float diff = fabsf(on[i] - offline[frame][i]);
      if (diff > tol) {
        fprintf(stderr,
                "Mismatch at frame %d bin %d: online=%f offline=%f diff=%f\n",
                frame, i, on[i], offline[frame][i], diff);
      }
      assert(diff < 5e-3f); // allow small numeric drift
    }
  }

  knf_online_feature_destroy(&online);
  knf_fbank_computer_destroy(&comp);
  knf_free_window(&win);
  free(window_buf);
  free(wave);

  printf("test_feature_demo passed\n");
  return 0;
}

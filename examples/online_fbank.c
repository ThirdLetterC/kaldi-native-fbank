#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "kaldi-native-fbank/online-feature.h"

constexpr float KNF_PI = 3.14159265358979323846f;
constexpr int32_t KNF_SAMPLE_RATE = 16000;
constexpr int32_t KNF_CHUNK_SAMPLES = 1600;
constexpr int32_t KNF_MAX_FRAMES_TO_PRINT = 3;
constexpr int32_t KNF_MAX_BINS_TO_PRINT = 6;

static void fill_sine_wave(float *wave, int32_t sample_count,
                           float frequency_hz, float sample_rate) {
  if (wave == nullptr || sample_count <= 0 || sample_rate <= 0.0f) {
    return;
  }

  for (int32_t i = 0; i < sample_count; ++i) {
    wave[i] =
        0.2f * sinf(2.0f * KNF_PI * frequency_hz * ((float)i / sample_rate));
  }
}

int main() {
  bool ok = false;
  knf_online_feature feature = {0};
  float *wave = nullptr;

  constexpr float duration_seconds = 0.5f;
  constexpr float tone_hz = 440.0f;
  const int32_t sample_count = (int32_t)(KNF_SAMPLE_RATE * duration_seconds);

  knf_fbank_opts opts;
  knf_fbank_opts_default(&opts);
  opts.frame_opts.samp_freq = (float)KNF_SAMPLE_RATE;
  opts.frame_opts.dither = 0.0f;
  opts.use_energy = false;
  opts.raw_energy = false;

  if (!knf_online_fbank_create(&opts, &feature)) {
    fprintf(stderr, "failed to create online fbank extractor\n");
    goto cleanup;
  }

  wave = (float *)calloc((size_t)sample_count, sizeof(float));
  if (wave == nullptr) {
    fprintf(stderr, "failed to allocate %d samples\n", sample_count);
    goto cleanup;
  }
  fill_sine_wave(wave, sample_count, tone_hz, (float)KNF_SAMPLE_RATE);

  // Feed the waveform in chunks to demonstrate streaming use.
  for (int32_t offset = 0; offset < sample_count; offset += KNF_CHUNK_SAMPLES) {
    int32_t chunk_size = sample_count - offset;
    if (chunk_size > KNF_CHUNK_SAMPLES) {
      chunk_size = KNF_CHUNK_SAMPLES;
    }
    if (!knf_online_accept_waveform(&feature, (float)KNF_SAMPLE_RATE,
                                    wave + offset, chunk_size)) {
      fprintf(stderr, "failed to accept waveform chunk at offset %d\n", offset);
      goto cleanup;
    }
  }

  if (!knf_online_input_finished(&feature)) {
    fprintf(stderr, "failed to finalize online feature extraction\n");
    goto cleanup;
  }

  const int32_t ready = knf_online_num_frames_ready(&feature);
  if (ready <= 0) {
    fprintf(stderr, "no frames were produced\n");
    goto cleanup;
  }

  const knf_fbank_computer *computer =
      (const knf_fbank_computer *)feature.computer;
  const int32_t dim = knf_fbank_dim(computer);
  const int32_t frames_to_print =
      ready < KNF_MAX_FRAMES_TO_PRINT ? ready : KNF_MAX_FRAMES_TO_PRINT;
  const int32_t bins_to_print =
      dim < KNF_MAX_BINS_TO_PRINT ? dim : KNF_MAX_BINS_TO_PRINT;

  printf("online fbank example\n");
  printf("samples=%d frames=%d dim=%d\n", sample_count, ready, dim);
  for (int32_t frame_index = 0; frame_index < frames_to_print; ++frame_index) {
    const float *frame = knf_online_get_frame(&feature, frame_index);
    if (frame == nullptr) {
      fprintf(stderr, "frame %d is unavailable\n", frame_index);
      goto cleanup;
    }

    printf("frame[%d]:", frame_index);
    for (int32_t bin_index = 0; bin_index < bins_to_print; ++bin_index) {
      printf(" %.4f", frame[bin_index]);
    }
    putchar('\n');
  }

  ok = true;

cleanup:
  free(wave);
  knf_online_feature_destroy(&feature);
  return ok ? 0 : 1;
}

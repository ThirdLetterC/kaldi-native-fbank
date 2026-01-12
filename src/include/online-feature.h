// Simple online feature extractor wrappers in C23.
#ifndef KALDI_NATIVE_FBANK_CSRC_ONLINE_FEATURE_H_
#define KALDI_NATIVE_FBANK_CSRC_ONLINE_FEATURE_H_

#include <stdint.h>

#include "feature-fbank.h"
#include "feature-mfcc.h"
#include "feature-raw-audio-samples.h"
#include "feature-window.h"
#include "whisper-feature.h"

typedef enum {
  KNF_ONLINE_FBANK,
  KNF_ONLINE_MFCC,
  KNF_ONLINE_RAW,
  KNF_ONLINE_WHISPER
} knf_online_kind;

typedef void (*knf_compute_fn)(void *computer, float raw_log_energy,
                               float vtln_warp, float *window, float *feature);
typedef const knf_frame_opts *(*knf_frame_fn)(const void *computer);
typedef int32_t (*knf_dim_fn)(const void *computer);
typedef bool (*knf_need_raw_energy_fn)(const void *computer);

typedef struct {
  knf_online_kind kind;
  void *computer;
  knf_compute_fn compute;
  knf_frame_fn frame_opts;
  knf_dim_fn dim;
  knf_need_raw_energy_fn need_raw_energy;

  knf_window window_fn;
  float *waveform;
  int32_t waveform_size;
  int32_t waveform_cap;
  int64_t waveform_offset;
  bool input_finished;

  float **features;
  int32_t num_features;
  int32_t features_cap;
} knf_online_feature;

bool knf_online_fbank_create(const knf_fbank_opts *opts,
                             knf_online_feature *out);
bool knf_online_mfcc_create(const knf_mfcc_opts *opts, knf_online_feature *out);
bool knf_online_raw_create(const knf_raw_audio_opts *opts,
                           knf_online_feature *out);
bool knf_online_whisper_create(const knf_whisper_opts *opts,
                               knf_online_feature *out);

void knf_online_feature_destroy(knf_online_feature *f);
void knf_online_accept_waveform(knf_online_feature *f, float sampling_rate,
                                const float *waveform, int32_t n);
void knf_online_input_finished(knf_online_feature *f);
int32_t knf_online_num_frames_ready(const knf_online_feature *f);
const float *knf_online_get_frame(const knf_online_feature *f, int32_t frame);

#endif // KALDI_NATIVE_FBANK_CSRC_ONLINE_FEATURE_H_

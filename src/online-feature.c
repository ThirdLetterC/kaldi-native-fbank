// Simple online feature extraction orchestrator.

#include <stdlib.h>
#include <string.h>

#include "feature-window.h"
#include "log.h"
#include "online-feature.h"

static bool knf_online_init_common(knf_online_feature *f, void *computer,
                                   knf_online_kind kind, knf_compute_fn compute,
                                   knf_frame_fn frame_fn, knf_dim_fn dim_fn,
                                   knf_need_raw_energy_fn need_fn) {
  memset(f, 0, sizeof(*f));
  f->kind = kind;
  f->computer = computer;
  f->compute = compute;
  f->frame_opts = frame_fn;
  f->dim = dim_fn;
  f->need_raw_energy = need_fn;

  if (!knf_make_window_from_opts(frame_fn(computer), &f->window_fn)) {
    return false;
  }
  f->waveform_cap = 1024;
  f->waveform = (float *)calloc((size_t)f->waveform_cap, sizeof(float));
  if (f->waveform == nullptr) {
    knf_free_window(&f->window_fn);
    return false;
  }
  return true;
}

static void knf_online_compute_new(knf_online_feature *f) {
  const knf_frame_opts *opts = f->frame_opts(f->computer);
  int64_t total_samples = f->waveform_offset + f->waveform_size;
  int32_t prev_frames = f->num_features;
  int32_t new_frames = knf_num_frames(total_samples, opts, f->input_finished);
  if (new_frames <= prev_frames)
    return;

  int32_t padded = knf_padded_window_size(opts);
  float *window = (float *)calloc((size_t)padded, sizeof(float));
  if (window == nullptr) {
    knf_fail("alloc", __FILE__, __func__, __LINE__, "window alloc failed");
  }
  for (int32_t frame = prev_frames; frame < new_frames; ++frame) {
    float raw_log_energy = 0.0f;
    knf_extract_window(f->waveform_offset, f->waveform, f->waveform_size, frame,
                       opts, &f->window_fn, window,
                       f->need_raw_energy(f->computer) ? &raw_log_energy
                                                       : nullptr);
    if (f->num_features == f->features_cap) {
      f->features_cap = f->features_cap ? f->features_cap * 2 : 16;
      auto new_features = (float **)realloc(
          f->features, sizeof(float *) * (size_t)f->features_cap);
      if (new_features == nullptr) {
        free(window);
        knf_fail("alloc", __FILE__, __func__, __LINE__, "feature list growth");
      }
      f->features = new_features;
    }
    int32_t dim = f->dim(f->computer);
    f->features[f->num_features] = (float *)calloc((size_t)dim, sizeof(float));
    if (f->features[f->num_features] == nullptr) {
      free(window);
      knf_fail("alloc", __FILE__, __func__, __LINE__, "feature alloc failed");
    }
    f->compute(f->computer, raw_log_energy, 1.0f, window,
               f->features[f->num_features]);
    f->num_features++;
  }
  free(window);

  int64_t first_sample_next = knf_first_sample_of_frame(new_frames, opts);
  int32_t discard = (int32_t)(first_sample_next - f->waveform_offset);
  if (discard > 0 && discard <= f->waveform_size) {
    memmove(f->waveform, f->waveform + discard,
            sizeof(float) * (f->waveform_size - discard));
    f->waveform_size -= discard;
    f->waveform_offset += discard;
  }
}

static int32_t knf_online_dim_fbank(const void *c) {
  return knf_fbank_dim((const knf_fbank_computer *)c);
}
static const knf_frame_opts *knf_online_frame_fbank(const void *c) {
  return knf_fbank_frame_opts((const knf_fbank_computer *)c);
}
static bool knf_online_need_fbank(const void *c) {
  return knf_fbank_need_raw_log_energy((const knf_fbank_computer *)c);
}
static void knf_online_compute_fbank(void *c, float e, float v, float *w,
                                     float *f) {
  knf_fbank_compute((knf_fbank_computer *)c, e, v, w, f);
}

static int32_t knf_online_dim_mfcc(const void *c) {
  return knf_mfcc_dim((const knf_mfcc_computer *)c);
}
static const knf_frame_opts *knf_online_frame_mfcc(const void *c) {
  return knf_mfcc_frame_opts((const knf_mfcc_computer *)c);
}
static bool knf_online_need_mfcc(const void *c) {
  return knf_mfcc_need_raw_log_energy((const knf_mfcc_computer *)c);
}
static void knf_online_compute_mfcc(void *c, float e, float v, float *w,
                                    float *f) {
  knf_mfcc_compute((knf_mfcc_computer *)c, e, v, w, f);
}

static int32_t knf_online_dim_raw(const void *c) {
  return knf_raw_audio_dim((const knf_raw_audio_computer *)c);
}
static const knf_frame_opts *knf_online_frame_raw(const void *c) {
  return knf_raw_audio_frame_opts((const knf_raw_audio_computer *)c);
}
static bool knf_online_need_raw(const void *c) {
  return knf_raw_audio_need_raw_log_energy((const knf_raw_audio_computer *)c);
}
static void knf_online_compute_raw(void *c, float e, float v, float *w,
                                   float *f) {
  knf_raw_audio_compute((knf_raw_audio_computer *)c, e, v, w, f);
}

static int32_t knf_online_dim_whisper(const void *c) {
  return knf_whisper_dim((const knf_whisper_computer *)c);
}
static const knf_frame_opts *knf_online_frame_whisper(const void *c) {
  return knf_whisper_frame_opts((const knf_whisper_computer *)c);
}
static bool knf_online_need_whisper(const void *c) {
  return knf_whisper_need_raw_log_energy((const knf_whisper_computer *)c);
}
static void knf_online_compute_whisper(void *c, float e, float v, float *w,
                                       float *f) {
  knf_whisper_compute((knf_whisper_computer *)c, e, v, w, f);
}

bool knf_online_fbank_create(const knf_fbank_opts *opts,
                             knf_online_feature *out) {
  knf_fbank_computer *c =
      (knf_fbank_computer *)calloc(1, sizeof(knf_fbank_computer));
  if (c == nullptr)
    return false;
  if (!knf_fbank_computer_create(opts, c)) {
    free(c);
    return false;
  }
  if (!knf_online_init_common(out, c, KNF_ONLINE_FBANK,
                              knf_online_compute_fbank, knf_online_frame_fbank,
                              knf_online_dim_fbank, knf_online_need_fbank)) {
    knf_fbank_computer_destroy(c);
    free(c);
    return false;
  }
  return true;
}

bool knf_online_mfcc_create(const knf_mfcc_opts *opts,
                            knf_online_feature *out) {
  knf_mfcc_computer *c =
      (knf_mfcc_computer *)calloc(1, sizeof(knf_mfcc_computer));
  if (c == nullptr)
    return false;
  if (!knf_mfcc_computer_create(opts, c)) {
    free(c);
    return false;
  }
  if (!knf_online_init_common(out, c, KNF_ONLINE_MFCC, knf_online_compute_mfcc,
                              knf_online_frame_mfcc, knf_online_dim_mfcc,
                              knf_online_need_mfcc)) {
    knf_mfcc_computer_destroy(c);
    free(c);
    return false;
  }
  return true;
}

bool knf_online_raw_create(const knf_raw_audio_opts *opts,
                           knf_online_feature *out) {
  knf_raw_audio_computer *c =
      (knf_raw_audio_computer *)calloc(1, sizeof(knf_raw_audio_computer));
  if (c == nullptr)
    return false;
  if (!knf_raw_audio_computer_create(opts, c)) {
    free(c);
    return false;
  }
  if (!knf_online_init_common(out, c, KNF_ONLINE_RAW, knf_online_compute_raw,
                              knf_online_frame_raw, knf_online_dim_raw,
                              knf_online_need_raw)) {
    knf_raw_audio_computer_destroy(c);
    free(c);
    return false;
  }
  return true;
}

bool knf_online_whisper_create(const knf_whisper_opts *opts,
                               knf_online_feature *out) {
  knf_whisper_computer *c =
      (knf_whisper_computer *)calloc(1, sizeof(knf_whisper_computer));
  if (c == nullptr)
    return false;
  if (!knf_whisper_computer_create(opts, c)) {
    free(c);
    return false;
  }
  if (!knf_online_init_common(out, c, KNF_ONLINE_WHISPER,
                              knf_online_compute_whisper,
                              knf_online_frame_whisper, knf_online_dim_whisper,
                              knf_online_need_whisper)) {
    knf_whisper_computer_destroy(c);
    free(c);
    return false;
  }
  return true;
}

void knf_online_feature_destroy(knf_online_feature *f) {
  if (f == nullptr)
    return;
  if (f->computer != nullptr) {
    switch (f->kind) {
    case KNF_ONLINE_FBANK:
      knf_fbank_computer_destroy((knf_fbank_computer *)f->computer);
      break;
    case KNF_ONLINE_MFCC:
      knf_mfcc_computer_destroy((knf_mfcc_computer *)f->computer);
      break;
    case KNF_ONLINE_RAW:
      knf_raw_audio_computer_destroy((knf_raw_audio_computer *)f->computer);
      break;
    case KNF_ONLINE_WHISPER:
      knf_whisper_computer_destroy((knf_whisper_computer *)f->computer);
      break;
    }
    free(f->computer);
  }
  knf_free_window(&f->window_fn);
  free(f->waveform);
  for (int32_t i = 0; i < f->num_features; ++i)
    free(f->features[i]);
  free(f->features);
}

void knf_online_accept_waveform(knf_online_feature *f, float sampling_rate,
                                const float *waveform, int32_t n) {
  if (n < 0) {
    knf_fail("length", __FILE__, __func__, __LINE__,
             "negative sample count %" PRId32, n);
  }
  if (n == 0)
    return;
  if (f->input_finished) {
    knf_fail("finished", __FILE__, __func__, __LINE__,
             "AcceptWaveform after finish");
  }
  if (sampling_rate != f->frame_opts(f->computer)->samp_freq) {
    knf_fail("rate", __FILE__, __func__, __LINE__, "sampling rate mismatch");
  }
  if (f->waveform_size + n > f->waveform_cap) {
    while (f->waveform_size + n > f->waveform_cap)
      f->waveform_cap *= 2;
    auto *new_waveform =
        (float *)realloc(f->waveform, sizeof(float) * (size_t)f->waveform_cap);
    if (new_waveform == nullptr) {
      knf_fail("alloc", __FILE__, __func__, __LINE__, "waveform growth failed");
    }
    f->waveform = new_waveform;
  }
  memcpy(f->waveform + f->waveform_size, waveform, sizeof(float) * n);
  f->waveform_size += n;
  knf_online_compute_new(f);
}

void knf_online_input_finished(knf_online_feature *f) {
  f->input_finished = true;
  knf_online_compute_new(f);
}

int32_t knf_online_num_frames_ready(const knf_online_feature *f) {
  return f->num_features;
}

const float *knf_online_get_frame(const knf_online_feature *f, int32_t frame) {
  if (frame < 0 || frame >= f->num_features)
    return nullptr;
  return f->features[frame];
}

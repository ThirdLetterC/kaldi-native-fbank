// C23 implementation of feature window utilities.

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "feature-window.h"
#include "log.h"

constexpr double KNF_PI = 3.14159265358979323846;

int32_t knf_round_up_power_of_two(int32_t n) {
  KNF_CHECK_GT(n, 0);
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

void knf_frame_opts_default(knf_frame_opts *opts) {
  opts->samp_freq = 16000.0f;
  opts->frame_shift_ms = 10.0f;
  opts->frame_length_ms = 25.0f;
  opts->dither = 0.00003f;
  opts->preemph_coeff = 0.97f;
  opts->remove_dc_offset = true;
  strncpy(opts->window_type, "povey", sizeof(opts->window_type));
  opts->round_to_power_of_two = true;
  opts->blackman_coeff = 0.42f;
  opts->snip_edges = true;
}

int32_t knf_window_shift(const knf_frame_opts *opts) {
  return (int32_t)(opts->samp_freq * 0.001f * opts->frame_shift_ms);
}

int32_t knf_window_size(const knf_frame_opts *opts) {
  return (int32_t)(opts->samp_freq * 0.001f * opts->frame_length_ms);
}

int32_t knf_padded_window_size(const knf_frame_opts *opts) {
  int32_t raw = knf_window_size(opts);
  return opts->round_to_power_of_two ? knf_round_up_power_of_two(raw) : raw;
}

static bool knf_window_match(const char *type, const char *target) {
  return strncmp(type, target, 15) == 0;
}

[[nodiscard]] bool knf_make_window(const char *window_type, int32_t window_size,
                                   float blackman_coeff, knf_window *out) {
  if (window_size <= 0)
    return false;
  out->data = (float *)calloc((size_t)window_size, sizeof(float));
  if (out->data == nullptr)
    return false;
  out->size = window_size;

  auto a = 2.0 * KNF_PI / (window_size - 1);
  if (knf_window_match(window_type, "hann")) {
    a = 2.0 * KNF_PI / window_size;
  }

  for (int32_t i = 0; i < window_size; ++i) {
    auto x = (double)i;
    if (knf_window_match(window_type, "hanning")) {
      out->data[i] = (float)(0.5 - 0.5 * cos(a * x));
    } else if (knf_window_match(window_type, "sine")) {
      out->data[i] = (float)sin(0.5 * a * x);
    } else if (knf_window_match(window_type, "hamming")) {
      out->data[i] = (float)(0.54 - 0.46 * cos(a * x));
    } else if (knf_window_match(window_type, "hann")) {
      out->data[i] = (float)(0.50 - 0.50 * cos(a * x));
    } else if (knf_window_match(window_type, "povey")) {
      out->data[i] = (float)pow(0.5 - 0.5 * cos(a * x), 0.85);
    } else if (knf_window_match(window_type, "rectangular")) {
      out->data[i] = 1.0f;
    } else if (knf_window_match(window_type, "blackman")) {
      out->data[i] = (float)(blackman_coeff - 0.5 * cos(a * x) +
                             (0.5 - blackman_coeff) * cos(2 * a * x));
    } else {
      free(out->data);
      out->data = nullptr;
      out->size = 0;
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool knf_make_window_from_opts(const knf_frame_opts *opts,
                                             knf_window *out) {
  return knf_make_window(opts->window_type, knf_window_size(opts),
                         opts->blackman_coeff, out);
}

void knf_free_window(knf_window *window) {
  if (window != nullptr && window->data != nullptr) {
    free(window->data);
    window->data = nullptr;
    window->size = 0;
  }
}

void knf_apply_window(const knf_window *window, float *wave) {
  if (window == nullptr || window->data == nullptr)
    return;
  for (int32_t i = 0; i < window->size; ++i) {
    wave[i] *= window->data[i];
  }
}

int64_t knf_first_sample_of_frame(int32_t frame, const knf_frame_opts *opts) {
  int64_t frame_shift = knf_window_shift(opts);
  if (opts->snip_edges) {
    return (int64_t)frame * frame_shift;
  }
  int64_t midpoint = frame_shift * frame + frame_shift / 2;
  return midpoint - knf_window_size(opts) / 2;
}

int32_t knf_num_frames(int64_t num_samples, const knf_frame_opts *opts,
                       bool flush) {
  int64_t frame_shift = knf_window_shift(opts);
  int64_t frame_length = knf_window_size(opts);
  if (opts->snip_edges) {
    if (num_samples < frame_length)
      return 0;
    return 1 + (int32_t)((num_samples - frame_length) / frame_shift);
  }

  int32_t num_frames =
      (int32_t)((num_samples + (frame_shift / 2)) / frame_shift);
  if (flush)
    return num_frames;

  int64_t end_sample =
      knf_first_sample_of_frame(num_frames - 1, opts) + frame_length;
  while (num_frames > 0 && end_sample > num_samples) {
    num_frames--;
    end_sample -= frame_shift;
  }
  return num_frames;
}

static float knf_rand_uniform() {
  return (float)rand() / (float)RAND_MAX - 0.5f;
}

[[nodiscard]] bool knf_extract_window(int64_t sample_offset, const float *wave,
                                      int32_t wave_size, int32_t frame_index,
                                      const knf_frame_opts *opts,
                                      const knf_window *window_function,
                                      float *window,
                                      float *log_energy_pre_window) {
  KNF_CHECK(sample_offset >= 0);
  KNF_CHECK(wave != nullptr);
  int32_t frame_length = knf_window_size(opts);
  int32_t frame_length_padded = knf_padded_window_size(opts);
  int64_t num_samples = sample_offset + wave_size;
  int64_t start_sample = knf_first_sample_of_frame(frame_index, opts);
  int64_t end_sample = start_sample + frame_length;

  if (opts->snip_edges) {
    if (!(start_sample >= sample_offset && end_sample <= num_samples)) {
      return false;
    }
  } else if (!(sample_offset == 0 || start_sample >= sample_offset)) {
    return false;
  }

  memset(window, 0, sizeof(float) * frame_length_padded);
  int32_t wave_start = (int32_t)(start_sample - sample_offset);
  int32_t wave_end = wave_start + frame_length;

  if (wave_start >= 0 && wave_end <= wave_size) {
    memcpy(window, wave + wave_start, sizeof(float) * frame_length);
  } else {
    for (int32_t s = 0; s < frame_length; ++s) {
      int32_t s_in_wave = s + wave_start;
      while (s_in_wave < 0 || s_in_wave >= wave_size) {
        if (s_in_wave < 0) {
          s_in_wave = -s_in_wave - 1;
        } else {
          s_in_wave = 2 * wave_size - 1 - s_in_wave;
        }
      }
      window[s] = wave[s_in_wave];
    }
  }

  knf_process_window(opts, window_function, window, log_energy_pre_window);
  for (int32_t i = frame_length; i < frame_length_padded; ++i) {
    window[i] = 0.0f;
  }
  return true;
}

void knf_process_window(const knf_frame_opts *opts,
                        const knf_window *window_function, float *window,
                        float *log_energy_pre_window) {
  int32_t window_size = knf_window_size(opts);

  if (opts->dither != 0.0f) {
    for (int32_t i = 0; i < window_size; ++i) {
      window[i] += opts->dither * knf_rand_uniform();
    }
  }

  if (opts->remove_dc_offset) {
    double sum = 0.0;
    for (int32_t i = 0; i < window_size; ++i)
      sum += window[i];
    float mean = (float)(sum / window_size);
    for (int32_t i = 0; i < window_size; ++i)
      window[i] -= mean;
  }

  if (opts->preemph_coeff != 0.0f) {
    float last = window[0];
    for (int32_t i = window_size - 1; i > 0; --i) {
      float prev = window[i - 1];
      window[i] -= opts->preemph_coeff * prev;
      last = prev;
    }
    window[0] -= opts->preemph_coeff * last;
  }

  if (log_energy_pre_window != nullptr) {
    float energy = 0.0f;
    for (int32_t i = 0; i < window_size; ++i)
      energy += window[i] * window[i];
    if (energy < 1e-10f)
      energy = 1e-10f;
    *log_energy_pre_window = logf(energy);
  }

  if (window_function) {
    knf_apply_window(window_function, window);
  }
}

float knf_inner_product(const float *a, const float *b, int32_t n) {
  double s = 0.0;
  for (int32_t i = 0; i < n; ++i) {
    s += (double)a[i] * (double)b[i];
  }
  return (float)s;
}

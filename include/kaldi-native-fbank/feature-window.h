// Feature window utilities rewritten for C23.
#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_WINDOW_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_WINDOW_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float samp_freq;
  float frame_shift_ms;
  float frame_length_ms;
  float dither;
  float preemph_coeff;
  bool remove_dc_offset;
  char window_type[16]; // povey, hamming, hann, sine, rectangular, blackman
  bool round_to_power_of_two;
  float blackman_coeff;
  bool snip_edges;
} knf_frame_opts;

typedef struct {
  float *data;
  int32_t size;
} knf_window;

int32_t knf_round_up_power_of_two(int32_t n);
void knf_frame_opts_default(knf_frame_opts *opts);
int32_t knf_window_shift(const knf_frame_opts *opts);
int32_t knf_window_size(const knf_frame_opts *opts);
int32_t knf_padded_window_size(const knf_frame_opts *opts);

[[nodiscard]] bool knf_make_window(const char *window_type, int32_t window_size,
                                   float blackman_coeff, knf_window *out);
[[nodiscard]] bool knf_make_window_from_opts(const knf_frame_opts *opts,
                                             knf_window *out);
void knf_free_window(knf_window *window);
void knf_apply_window(const knf_window *window, float *wave);

int64_t knf_first_sample_of_frame(int32_t frame, const knf_frame_opts *opts);
int32_t knf_num_frames(int64_t num_samples, const knf_frame_opts *opts,
                       bool flush);
[[nodiscard]] bool knf_extract_window(int64_t sample_offset, const float *wave,
                                      int32_t wave_size, int32_t frame_index,
                                      const knf_frame_opts *opts,
                                      const knf_window *window_function,
                                      float *window,
                                      float *log_energy_pre_window);
void knf_process_window(const knf_frame_opts *opts,
                        const knf_window *window_function, float *window,
                        float *log_energy_pre_window);
float knf_inner_product(const float *a, const float *b, int32_t n);

#ifdef __cplusplus
}
#endif

#endif // KALDI_NATIVE_FBANK_CSRC_FEATURE_WINDOW_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

constexpr float KNF_PI = 3.14159265358979323846f;

#include "istft.h"
#include "stft.h"

int main() {
  const int n = 640;
  float *wave = (float *)calloc((size_t)n, sizeof(float));
  if (wave == nullptr) {
    return 1;
  }
  for (int i = 0; i < n; ++i) {
    wave[i] = sinf(2.0f * KNF_PI * 440.0f * ((float)i / 16000.0f));
  }

  knf_stft_config stft_cfg;
  knf_stft_config_default(&stft_cfg);
  knf_stft_result res = {0};
  assert(knf_stft_compute(&stft_cfg, wave, n, &res));

  knf_istft_config istft_cfg;
  knf_istft_config_default(&istft_cfg);
  istft_cfg.n_fft = stft_cfg.n_fft;
  istft_cfg.hop_length = stft_cfg.hop_length;
  istft_cfg.win_length = stft_cfg.win_length;
  istft_cfg.center = stft_cfg.center;
  istft_cfg.normalized = stft_cfg.normalized;
  float *recon = nullptr;
  int32_t recon_n = 0;
  assert(knf_istft_compute(&istft_cfg, &res, &recon, &recon_n));
  assert(recon_n == n);

  float max_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    float err = fabsf(recon[i] - wave[i]);
    if (err > max_err)
      max_err = err;
  }
  printf("max reconstruction error: %f\n", max_err);
  assert(max_err < 1e-2f);

  free(recon);
  knf_stft_result_free(&res);
  free(wave);
  printf("test_stft_istft passed\n");
  return 0;
}

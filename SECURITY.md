# Security Model

`kaldi-native-fbank` is an in-process C23 DSP library for Kaldi-style
filterbank, MFCC, Whisper-style mel features, STFT/ISTFT, and simple online
feature extraction. It operates on caller-provided configuration structs and
audio/sample buffers.

The library does not perform network I/O, filesystem I/O, privilege
management, sandboxing, authentication, authorization, or secret storage. The
only direct side effects in library code are heap allocation, diagnostic
logging to `stderr`, and process termination through `knf_fail()` on some
contract violations.

Everything passed into the public API should be treated as untrusted input.
That includes option structs, frame dimensions, sample counts, window
descriptors, STFT buffers, and all caller-provided pointers.

## Trust Boundaries

- All public structs in `include/kaldi-native-fbank/*.h` are caller-controlled.
- The library assumes every input pointer is valid for the documented object
  size and lifetime.
- Several compute APIs mutate caller-owned buffers in place, including
  `knf_rfft_compute()`, `knf_fbank_compute()`, `knf_mfcc_compute()`,
  `knf_whisper_compute()`, and `knf_raw_audio_compute()`.
- `knf_stft_compute()` allocates `knf_stft_result.real` and
  `knf_stft_result.imag`; the caller must release them with
  `knf_stft_result_free()`.
- `knf_istft_compute()` allocates `*out_samples`; the caller owns that buffer
  and must release it with `free()`.
- Online extraction keeps mutable internal state and heap-backed feature
  storage inside `knf_online_feature`. That state is caller-owned after
  creation and must be destroyed with `knf_online_feature_destroy()`.
- The bundled FFT backend is local vendor code in `src/pocketfft.c` and
  `include/pocketfft/pocketfft.h`.

## Protected Properties

- No direct network or filesystem attack surface in the core DSP routines.
- Allocation failures are checked and usually reported via `false` or
  `nullptr`, depending on the API.
- STFT/ISTFT code uses explicit dimension checks and `int64_t` intermediates
  before narrowing sizes back to `int32_t`.
- Destructors/free helpers are provided for every heap-owning public object.
- The codebase has regression tests for FFT, STFT/ISTFT, windowing, mel banks,
  fbank, MFCC, online extraction, and Whisper-style features.

## Security-Relevant Limits And Caller Responsibilities

- This library is not hardened against invalid pointers, dangling pointers,
  undersized writable buffers, or forged object layouts. Passing malformed
  pointers is undefined at the C level.
- Public APIs do not provide a uniform "never abort" contract. With
  `KNF_ENABLE_CHECK` enabled, failed internal `KNF_CHECK` assertions terminate
  the process. Normal public entry points are expected to reject malformed
  dimensions, invalid mel option ranges, online state misuse, and allocation
  growth overflow by returning `false`/`nullptr` instead of calling
  `knf_fail()`.
- The library is not thread-safe by default. Objects such as `knf_rfft`,
  `knf_fbank_computer`, `knf_mfcc_computer`, `knf_whisper_computer`, and
  `knf_online_feature` contain mutable state and require external
  synchronization if shared across threads.
- Callers must enforce input size limits. Large FFT sizes, long waveforms, high
  frame counts, or unbounded online streams can trigger large heap allocations.
- Floating-point inputs are treated as ordinary numeric data. The library does
  not reject `NaN`, `Inf`, denormals, or adversarially chosen sample values.
- The dither path uses `rand()`. It is for signal processing only and provides
  no security or unpredictability guarantees.
- String fields such as `window_type`, `pad_mode`, and `norm` are compared as
  fixed-size C strings. Callers should initialize option structs through the
  provided `*_default()` helpers before mutating fields.
- `knf_mel_banks_create()` and dependent constructors reject invalid mel ranges
  and malformed bin layouts by returning `nullptr`, but callers still need to
  bound option values to avoid pathological allocation requests.
- The library is not a sandbox or policy engine. Validation of file paths,
  model provenance, audio provenance, rate limiting, and tenant isolation
  belongs in the embedding application.
- No constant-time or side-channel-resistant behavior is claimed.

## Defensive Posture In This Codebase

- In-tree build metadata targets C23 (`-std=c23` in `build.zig` and
  `compile_flags.txt`).
- `compile_flags.txt` enables `-Wall -Wextra -Wpedantic -Werror` for editor and
  tooling integration.
- Heap allocations in library code are checked against `nullptr` before use.
- Ownership is generally explicit: constructors allocate, `*_destroy()` or
  `*_free()` release.
- Public online ingestion, mel-bank construction, STFT/ISTFT setup, and FFT
  execution paths now validate caller-controlled sizes and fail closed on
  invalid inputs instead of aborting.
- The repository currently includes functional regression tests, but no
  in-tree fuzzing harnesses or sanitizer-specific build targets.

## Verification

Validation paths present in this checkout:

- `zig build`
- `zig build test`
- `just test`

The test suite exercises the main DSP paths and some failure cases, but it is
not a substitute for application-level bounds checking, fuzzing, or hostile
input validation.

## Supported Versions

This repository does not currently publish a maintained-branches matrix. Report
security issues against the current `main` branch unless the maintainer states
otherwise.

## Reporting

This repository does not currently publish a dedicated private security contact
in-tree. If you need to report a vulnerability, use a maintainer-controlled
private channel when one is available and avoid posting exploit details
publicly before the issue has been assessed.

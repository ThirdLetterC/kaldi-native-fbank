# `kaldi-native-fbank`

Minimal C23 implementation of Kaldi-style filterbank, MFCC, STFT/ISTFT, and related utilities with a Zig build.

## Prerequisites
- `zig 0.15.x`
- Uses the bundled `pocketfft` implementation; no external FFT library needed.

## Build
- Build the static library and install artifacts to `zig-out`:  
  `zig build`
- Run the C test executables:  
  `zig build test`
- Run the sample program under `examples/`:  
  `zig build run-online_fbank_example`

## Layout
- Core sources: `src/*.c`, public headers in `include/kaldi-native-fbank`
- Examples: `examples/*.c`
- Zig build script: `build.zig` (installs `kaldi-native-fbank-core`, examples, and test binaries)
- Build outputs: `zig-out`, cache in `.zig-cache` (both gitignored)

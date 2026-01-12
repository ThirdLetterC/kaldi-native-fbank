# `kaldi-native-fbank`

Minimal C23 implementation of Kaldi-style filterbank, MFCC, STFT/ISTFT, and related utilities with a Zig build.

## Prerequisites
- `zig 0.15.x`
- `fftw3f` development headers and shared library (`libfftw3f`); on Debian/Ubuntu: `sudo apt install libfftw3-dev`

## Build
- Build the static library and install artifacts to `zig-out`:  
  `zig build`
- Run the C test executables:  
  `zig build test`

## Layout
- Core sources: `src/*.c`, headers in `src/include`
- Zig build script: `build.zig` (installs `kaldi-native-fbank-core` and test binaries)
- Build outputs: `zig-out`, cache in `.zig-cache` (both gitignored)

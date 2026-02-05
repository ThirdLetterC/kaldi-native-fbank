set shell := ["bash", "-uc"]

build:
    zig build

test:
    zig build test

run *args:
    zig build run -- {{args}}

fmt:
    zig fmt build.zig
    clang-format -i src/*.c include/kaldi-native-fbank/*.h

clean:
    rm -rf zig-cache zig-out .zig-cache

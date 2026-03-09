set shell := ["bash", "-uc"]

build:
    zig build

test:
    zig build test

example:
    zig build run-online_fbank_example

run *args:
    zig build run -- {{args}}

fmt:
    zig fmt build.zig
    clang-format -i src/*.c examples/*.c include/kaldi-native-fbank/*.h

clean:
    rm -rf zig-cache zig-out .zig-cache

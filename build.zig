const std = @import("std");

const c_flags = [_][]const u8{"-std=c23"};

const core_sources = [_][]const u8{
    "src/log.c",
    "src/kaldi-math.c",
    "src/rfft.c",
    "src/pocketfft.c",
    "src/feature-window.c",
    "src/feature-functions.c",
    "src/mel-computations.c",
    "src/feature-fbank.c",
    "src/feature-mfcc.c",
    "src/feature-raw-audio-samples.c",
    "src/online-feature.c",
    "src/whisper-feature.c",
    "src/stft.c",
    "src/istft.c",
};

const test_sources = [_]struct { name: []const u8, path: []const u8 }{
    .{ .name = "test_rfft", .path = "src/test_rfft.c" },
    .{ .name = "test_stft_istft", .path = "src/test_stft_istft.c" },
    .{ .name = "test_feature_window", .path = "src/test_feature_window.c" },
    .{ .name = "test_mel_banks", .path = "src/test_mel_banks.c" },
    .{ .name = "test_fbank", .path = "src/test_fbank.c" },
    .{ .name = "test_mfcc", .path = "src/test_mfcc.c" },
    .{ .name = "test_online", .path = "src/test_online.c" },
    .{ .name = "test_feature_demo", .path = "src/test_feature_demo.c" },
    .{ .name = "test_whisper", .path = "src/test_whisper.c" },
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_module = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    lib_module.addIncludePath(.{ .src_path = .{ .owner = b, .sub_path = "src" } });
    lib_module.addIncludePath(.{ .src_path = .{ .owner = b, .sub_path = "src/include" } });
    lib_module.addCSourceFiles(.{ .files = &core_sources, .flags = &c_flags });

    const lib = b.addLibrary(.{
        .name = "kaldi-native-fbank-core",
        .root_module = lib_module,
        .linkage = .static,
    });
    b.installArtifact(lib);

    const test_step = b.step("test", "Run C test executables");
    for (test_sources) |t| {
        const test_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        test_module.addIncludePath(.{ .src_path = .{ .owner = b, .sub_path = "src" } });
        test_module.addIncludePath(.{ .src_path = .{ .owner = b, .sub_path = "src/include" } });
        test_module.addCSourceFiles(.{
            .files = &[_][]const u8{t.path},
            .flags = &c_flags,
        });

        const exe = b.addExecutable(.{
            .name = t.name,
            .root_module = test_module,
        });
        exe.linkLibrary(lib);
        linkCoreDeps(exe, target);

        b.installArtifact(exe);

        const run = b.addRunArtifact(exe);
        test_step.dependOn(&run.step);
    }
}

fn linkCoreDeps(step: *std.Build.Step.Compile, target: std.Build.ResolvedTarget) void {
    step.linkLibC();

    const os_tag = target.result.os.tag;
    if (os_tag != .windows and os_tag != .uefi) {
        step.linkSystemLibrary("m");
    }
}

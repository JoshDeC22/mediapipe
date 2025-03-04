licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

BIN_DATA = [
    "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
    "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
]

LIB_DEPS = [
    "//mediapipe/examples/desktop:demo_run_graph_main",
    "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu",
    "//mediapipe/graphs/hand_tracking:mobile_calculators",
    "//mediapipe/calculators/core:packet_presence_calculator",
    "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
]

# Windows DLL
cc_binary(
    target_compatible_with = [
        "@platforms//os:windows",
    ],
    name = "libhand.dll",
    deps = [":libhand_win"],
    data = BIN_DATA,
    linkshared = 1,
)

# Linux SO
cc_binary(
    target_compatible_with = [
        "@platforms//os:linux",
        name = "libhand.so",
        deps = [":libhand_linux"],
        data = BIN_DATA,
        linkopts = ["-Wl,-soname,libhand.so"],
        linkshared = 1,
    ]
)

# Mac DYLIB
cc_binary(
    target_compatible_with = [
        "@platforms//os:macos",
    ],
    name = "libhand.dylib",
    deps = ":libhand_mac",
    data = BIN_DATA,
    linkopts = ["-Wl,-install_name,@rpath/libhand.dylib"],
    linkshared = 1,
)

# Library for windows
cc_library(
    target_compatible_with = [
        "@platforms//os:windows",
    ],
    name = "libhand_win",
    srcs = ["libhand_main.cc"],
    hdrs = ["libhand_main.h"],
    visibility = ["//visibility:public"],
    deps = LIB_DEPS,
    defines = ["COMPILING_DLL=1"],
    alwayslink = 1,
)

# Library for Linux
cc_library(
    target_compatible_with = [
        "@platforms//os:linux",
    ],
    name = "libhand_linux",
    srcs = ["libhand_main.cc"],
    hdrs = ["libhand_main.h"],
    visibility = ["//visibility:public"],
    deps = LIB_DEPS,
    alwayslink = 1,
)

# Library for Mac
cc_library(
    target_compatible_with = [
        "@platforms//os:macos",
    ],
    name = "libhand_mac",
    srcs = ["libhand_main.cc"],
    hdrs = ["libhand_main.h"],
    visibility = ["//visibility:public"],
    deps = LIB_DEPS,
    alwayslink = 1,
)
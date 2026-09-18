#!/usr/bin/env python3
"""Build the pinned ORB-SLAM3 experiment on Apple Silicon, outside the product.

Requires CMake, Apple command-line tools and existing Homebrew Boost/OpenSSL.
All downloaded sources and newly built dependencies stay under --vendor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

PINS = {
    "ORB_SLAM3": ("https://github.com/UZ-SLAMLab/ORB_SLAM3.git", "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"),
    "opencv": ("https://github.com/opencv/opencv.git", "31b0eeea0b44b370fd0712312df4214d4ae1b158"),
    "eigen": ("https://gitlab.com/libeigen/eigen.git", "3147391d946bb4b6c68edd901f2add6ac1f31f8c"),
    "Pangolin": ("https://github.com/stevenlovegrove/Pangolin.git", "aff6883c83f3fd7e8268a9715e84266c42e2efe3"),
}


def write_changed(path, contents):
    if path.read_text() != contents:
        path.write_text(contents)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--cmake", default=shutil.which("cmake"), required=not shutil.which("cmake"))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--stage", choices=["deps", "orb", "runner", "all"], default="all")
    args = p.parse_args()
    v = args.vendor.resolve()
    prefix = v / "install"
    build = v / "build"
    build.mkdir(parents=True, exist_ok=True)
    history = build / "commands.jsonl"

    def run(cmd, name, timeout=None):
        cmd = [str(x) for x in cmd]
        with history.open("a") as f:
            f.write(json.dumps({"name": name, "argv": cmd}) + "\n")
        print(name, flush=True)
        with (build / (name + ".log")).open("a") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)

    def checkout(name):
        path = v / name
        url, ref = PINS[name]
        if not path.exists():
            run(["git", "clone", "--filter=blob:none", "--no-checkout", url, path], "clone-" + name)
            run(["git", "-C", path, "checkout", ref], "checkout-" + name)
        revision = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        expected = subprocess.check_output(["git", "-C", path, "rev-parse", ref + "^{commit}"], text=True).strip()
        if revision != expected:
            raise ValueError(f"{name}: expected {expected}, found {revision}; refusing to reset a checkout")
        return path

    common = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_STANDARD=17", "-DCMAKE_OSX_ARCHITECTURES=arm64",
              "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", f"-DCMAKE_INSTALL_PREFIX={prefix}", f"-DCMAKE_PREFIX_PATH={prefix}"]

    def configure(source, name, flags=()):
        run([args.cmake, "-S", source, "-B", build / name, *common, *flags], name + "-configure")

    def compile_(name, target=None, install=False):
        cmd = [args.cmake, "--build", build / name, "--parallel", args.jobs]
        if target:
            cmd += ["--target", target]
        run(cmd, name + "-build")
        if install:
            run([args.cmake, "--install", build / name], name + "-install")

    if args.stage in ("deps", "all"):
        eigen = checkout("eigen")
        configure(eigen, "eigen", ["-DBUILD_TESTING=OFF", "-DEIGEN_BUILD_DOC=OFF"])
        run([args.cmake, "--install", build / "eigen"], "eigen-install")
        archive = v / "glew-2.2.0.tgz"
        if not archive.exists():
            urllib.request.urlretrieve("https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.tgz", archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != "d4fc82893cfb00109578d0a1a2337fb8ca335b3ceccf97b97e5cc7f08e4353e1":
            raise ValueError("GLEW source archive checksum mismatch")
        if not (v / "glew-2.2.0").exists():
            run(["tar", "-xzf", archive, "-C", v], "glew-extract")
        configure(v / "glew-2.2.0/build/cmake", "glew", ["-DBUILD_UTILS=OFF"])
        compile_("glew", install=True)
        opencv = checkout("opencv")
        configure(opencv, "opencv", ["-DBUILD_LIST=core,imgproc,imgcodecs,features2d,calib3d,highgui",
            "-DBUILD_JPEG=ON", "-DBUILD_PNG=ON", *[f"-D{flag}=OFF" for flag in (
                "BUILD_TESTS", "BUILD_PERF_TESTS", "BUILD_EXAMPLES", "BUILD_DOCS", "BUILD_opencv_python2",
                "BUILD_opencv_python3", "BUILD_JAVA", "WITH_IPP", "WITH_OPENEXR", "WITH_OPENJPEG", "WITH_TIFF",
                "WITH_WEBP", "WITH_JASPER", "WITH_AVFOUNDATION", "WITH_FFMPEG", "WITH_GSTREAMER", "WITH_1394", "WITH_OPENCL")]])
        compile_("opencv", install=True)
        pangolin = checkout("Pangolin")
        configure(pangolin, "Pangolin", [f"-D{flag}=OFF" for flag in (
            "BUILD_TOOLS", "BUILD_EXAMPLES", "BUILD_TESTS", "BUILD_PANGOLIN_PYTHON", "BUILD_PANGOLIN_LIBPNG",
            "BUILD_PANGOLIN_LIBJPEG", "BUILD_PANGOLIN_LIBTIFF", "BUILD_PANGOLIN_LIBOPENEXR", "BUILD_PANGOLIN_LIBRAW",
            "BUILD_PANGOLIN_LZ4", "BUILD_PANGOLIN_ZSTD", "BUILD_PANGOLIN_LIBDC1394", "BUILD_PANGOLIN_FFMPEG",
            "BUILD_PANGOLIN_REALSENSE", "BUILD_PANGOLIN_REALSENSE2", "BUILD_PANGOLIN_OPENNI", "BUILD_PANGOLIN_OPENNI2", "BUILD_PANGOLIN_LIBUVC")])
        compile_("Pangolin", install=True)

    if args.stage in ("orb", "runner", "all"):
        orb = checkout("ORB_SLAM3")
        # Native build corrections only; no feature, matching, or optimizer change.
        for path in [orb / "Thirdparty/DBoW2/DBoW2/FORB.cpp", orb / "src/ORBmatcher.cc"]:
            write_changed(path, path.read_text().replace("stdint-gcc.h", "stdint.h"))
        for name in ("hyper_graph.h", "robust_kernel.h", "marginal_covariance_cholesky.h", "estimate_propagator.h", "sparse_block_matrix_ccs.h"):
            path = orb / "Thirdparty/g2o/g2o/core" / name
            write_changed(path, path.read_text().replace("tr1/unordered_map", "unordered_map").replace("tr1/memory", "memory").replace("std::tr1::", "std::"))
        # Upstream increments this generation counter; bool++ is illegal in C++17
        # and cannot distinguish successive cancellation generations in C++11.
        path = orb / "include/LoopClosing.h"
        write_changed(path, path.read_text().replace("bool mnFullBAIdx;", "unsigned long mnFullBAIdx;"))
        path = orb / "src/LoopClosing.cc"
        write_changed(path, path.read_text().replace("int idx =  mnFullBAIdx;", "unsigned long idx = mnFullBAIdx;"))
        # The upstream N*N variable-length array exhausts a macOS worker stack
        # at 366 observations. Keep the exact distance/median algorithm on heap.
        path = orb / "src/MapPoint.cc"
        text = path.read_text().replace("float Distances[N][N];", "std::vector<float> Distances(N * N);")
        text = text.replace("Distances[i][i]", "Distances[i*N+i]").replace("Distances[i][j]", "Distances[i*N+j]").replace("Distances[j][i]", "Distances[j*N+i]")
        text = text.replace("vector<int> vDists(Distances[i],Distances[i]+N);", "vector<int> vDists(Distances.begin()+i*N,Distances.begin()+(i+1)*N);")
        write_changed(path, text)
        cmake_file = orb / "CMakeLists.txt"
        cmake_text = cmake_file.read_text().replace("${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so", "${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2${CMAKE_SHARED_LIBRARY_SUFFIX}")
        cmake_text = cmake_text.replace("${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so", "g2o")
        if args.stage == "runner":
            runner = v / "runner/phase2_camera_runner.cc"
            if not runner.exists():
                raise FileNotFoundError(runner)
            for target in ("phase2_camera_runner", "phase2_camera_gba_check", "phase2_camera_mask_check"):
                source = v / "runner" / (target + ".cc")
                if source.exists() and f"add_executable({target} " not in cmake_text:
                    cmake_text += f'\nadd_executable({target} "{source}")\ntarget_link_libraries({target} ORB_SLAM3)\nset_target_properties({target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "{v / "runner"}")\n'
        write_changed(cmake_file, cmake_text)
        native = ["-DCMAKE_CXX_FLAGS=-I/opt/homebrew/opt/boost/include -I/opt/homebrew/opt/openssl@3/include",
            "-DCMAKE_SHARED_LINKER_FLAGS=-L/opt/homebrew/opt/boost/lib -L/opt/homebrew/opt/openssl@3/lib",
            "-DCMAKE_EXE_LINKER_FLAGS=-L/opt/homebrew/opt/boost/lib -L/opt/homebrew/opt/openssl@3/lib",
            f"-DCMAKE_BUILD_RPATH={prefix / 'lib'};/opt/homebrew/opt/boost/lib;/opt/homebrew/opt/openssl@3/lib",
            f"-DEIGEN3_INCLUDE_DIR={prefix / 'include/eigen3'}", f"-DG2O_EIGEN3_INCLUDE={prefix / 'include/eigen3'}"]
        configure(orb / "Thirdparty/DBoW2", "DBoW2", native)
        compile_("DBoW2")
        configure(orb, "ORB_SLAM3", native)
        compile_("ORB_SLAM3", "phase2_camera_runner" if args.stage == "runner" else "ORB_SLAM3")
        if args.stage == "runner":
            for check in ("phase2_camera_gba_check", "phase2_camera_mask_check"):
                if (v / "runner" / (check + ".cc")).exists():
                    compile_("ORB_SLAM3", check)
                    run([v / "runner" / check], check, timeout=30)
        (build / "orb-local-changes.patch").write_bytes(subprocess.check_output(["git", "-C", orb, "diff"]))
        vocab = orb / "Vocabulary/ORBvoc.txt"
        if not vocab.exists():
            run(["tar", "-xzf", orb / "Vocabulary/ORBvoc.txt.tar.gz", "-C", orb / "Vocabulary"], "vocabulary-extract")
    revisions = {name: subprocess.check_output(["git", "-C", v / name, "rev-parse", "HEAD"], text=True).strip()
                 for name in PINS if (v / name).exists()}
    archive = v / "glew-2.2.0.tgz"
    if archive.exists():
        revisions[archive.name + "_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (build / "source-revisions.json").write_text(json.dumps(revisions, indent=2) + "\n")


if __name__ == "__main__":
    main()

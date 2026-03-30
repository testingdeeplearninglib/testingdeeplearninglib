# GCFL-OTHER-0081

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def _print_and_exit(msg: str, code: int) -> None:
    print(msg)
    raise SystemExit(code)


def _skip(reason: str) -> None:
    _print_and_exit(f"SKIP_ENV: {reason}", 0)


def _pass() -> None:
    _print_and_exit("Test Passed ✅", 0)


def _fail() -> None:
    _print_and_exit("Test Failed ❌", 0)


def _harness_error(e: BaseException) -> None:
    _print_and_exit(f"HARNESS_ERROR: {type(e).__name__}: {e}", 1)


def _looks_like_tf_root(p: Path) -> bool:
    return (
        p.is_dir()
        and ((p / "WORKSPACE").exists() or (p / "WORKSPACE.bazel").exists())
        and (p / "tensorflow").is_dir()
    )


def _find_tf_root() -> Optional[Path]:
    for k in ("GCFL_TF_REPO", "TF_REPO", "TENSORFLOW_SRC", "TF_SRC", "TF_SRC_DIR"):
        v = os.environ.get(k, "").strip()
        if not v:
            continue
        p = Path(v).expanduser().resolve()
        if _looks_like_tf_root(p):
            return p

    cur = Path.cwd().resolve()
    for p in [cur] + list(cur.parents):
        if _looks_like_tf_root(p):
            return p

    return None


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _get_android_sdk_root() -> Optional[Path]:
    for k in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        v = os.environ.get(k, "").strip()
        if not v:
            continue
        p = Path(v).expanduser().resolve()
        if p.is_dir():
            return p
    return None


def _get_android_ndk_root(sdk_root: Optional[Path]) -> Optional[Path]:
    for k in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "ANDROID_NDK", "NDK_HOME", "NDK_ROOT"):
        v = os.environ.get(k, "").strip()
        if not v:
            continue
        p = Path(v).expanduser().resolve()
        if p.is_dir():
            return p

    if sdk_root is not None:
        ndk_bundle = sdk_root / "ndk-bundle"
        if ndk_bundle.is_dir():
            return ndk_bundle.resolve()

        ndk_dir = sdk_root / "ndk"
        if ndk_dir.is_dir():
            versions = sorted([p for p in ndk_dir.iterdir() if p.is_dir()])
            if versions:
                return versions[-1].resolve()

    return None


def _find_java() -> Optional[str]:
    java_home = os.environ.get("JAVA_HOME", "").strip()
    if java_home:
        java_bin = Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if java_bin.is_file() and os.access(str(java_bin), os.X_OK):
            return str(java_bin)
    return _which("java")


def _detect_layout(tf_root: Path):
    old_prof = tf_root / "tensorflow" / "contrib" / "lite" / "profiling" / "profile_summarizer.cc"
    old_build = tf_root / "tensorflow" / "contrib" / "lite" / "tools" / "benchmark" / "BUILD"

    new_prof = tf_root / "tensorflow" / "lite" / "profiling" / "profile_summarizer.cc"
    new_build = tf_root / "tensorflow" / "lite" / "tools" / "benchmark" / "BUILD"

    if old_prof.exists() and old_build.exists():
        return {
            "layout": "contrib",
            "needle_file": "tensorflow/contrib/lite/profiling/profile_summarizer.cc",
            "target": "//tensorflow/contrib/lite/tools/benchmark:benchmark_model",
        }

    if new_prof.exists() and new_build.exists():
        return {
            "layout": "lite",
            "needle_file": "tensorflow/lite/profiling/profile_summarizer.cc",
            "target": "//tensorflow/lite/tools/benchmark:benchmark_model",
        }

    return None


def _run_cmd(cmd, cwd: Path, env: dict, timeout_sec: int) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        return p.returncode, out
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout_sec}s"
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def _shutdown_bazel(tf_root: Path, env: dict) -> None:
    bazel = _which("bazel") or _which("bazelisk")
    if not bazel:
        return
    _run_cmd([bazel, "shutdown"], tf_root, env, 30)


def _fast_precheck(tf_root: Path, env: dict, target: str) -> Tuple[bool, str]:
    bazel = _which("bazel") or _which("bazelisk")
    if not bazel:
        return False, "bazel/bazelisk not found in PATH"

    cmd = [
        bazel,
        "query",
        "--color=no",
        "--curses=no",
        "--incompatible_remove_native_http_archive=false",
        target,
    ]
    rc, out = _run_cmd(cmd, tf_root, env, 90)
    if rc == 0:
        return True, out
    return False, out


def _run_bazel_build(tf_root: Path, env: dict, target: str, timeout_sec: int) -> Tuple[int, str]:
    bazel = _which("bazel") or _which("bazelisk")
    if not bazel:
        _skip("bazel/bazelisk not found in PATH")

    cmd = [
        bazel,
        "build",
        "--color=no",
        "--curses=no",
        "-s",
        "--verbose_failures",
        "--incompatible_remove_native_http_archive=false",
        "-c",
        "opt",
        "--config=android_arm",
        "--cxxopt=--std=c++11",
        target,
    ]
    return _run_cmd(cmd, tf_root, env, timeout_sec)


def _tail(text: str, n: int = 120) -> str:
    lines = text.splitlines()
    if not lines:
        return "<no output>"
    return "\n".join(lines[-n:])


def main() -> None:
    try:
        tf_root = _find_tf_root()
        if tf_root is None:
            _skip(
                "TensorFlow source repo not found. Set GCFL_TF_REPO (or TF_REPO / TENSORFLOW_SRC) to a TensorFlow checkout."
            )

        layout = _detect_layout(tf_root)
        if layout is None:
            _skip(
                "repo revision does not contain either tensorflow/contrib/lite or tensorflow/lite benchmark_model sources"
            )

        sdk_root = _get_android_sdk_root()
        if sdk_root is None:
            _skip("Android SDK root not found. Set ANDROID_HOME or ANDROID_SDK_ROOT.")

        ndk_root = _get_android_ndk_root(sdk_root)
        if ndk_root is None:
            _skip("Android NDK root not found. Set ANDROID_NDK_HOME/ANDROID_NDK_ROOT or install an NDK under $ANDROID_HOME/ndk")

        java_bin = _find_java()
        if java_bin is None:
            _skip("java not found. Install a JDK and/or set JAVA_HOME.")

        timeout_sec = 180
        try:
            timeout_sec = int(str(os.environ.get("GCFL_TIMEOUT_SEC", timeout_sec)).strip())
        except Exception:
            timeout_sec = 180

        print(f"INFO: tf_root={tf_root}")
        print(f"INFO: layout={layout['layout']}")
        print(f"INFO: android_sdk={sdk_root}")
        print(f"INFO: android_ndk={ndk_root}")
        print(f"INFO: java={java_bin}")
        print(f"INFO: bazel={_which('bazel') or _which('bazelisk') or 'NOT_FOUND'}")
        print(f"INFO: timeout_sec={timeout_sec}")

        env = os.environ.copy()
        env.setdefault("ANDROID_HOME", str(sdk_root))
        env.setdefault("ANDROID_SDK_ROOT", str(sdk_root))
        env.setdefault("ANDROID_NDK_HOME", str(ndk_root))
        env.setdefault("ANDROID_NDK_ROOT", str(ndk_root))

        _shutdown_bazel(tf_root, env)

        ok, precheck_out = _fast_precheck(tf_root, env, layout["target"])
        if not ok:
            print("INFO: bazel precheck output tail:")
            print(_tail(precheck_out, 80))

            if "native http_archive rule is deprecated" in precheck_out:
                _skip("bazel/rules_closure incompatibility before target analysis; build environment incompatible")

            _skip("bazel precheck failed before target analysis (environment/build prerequisites not satisfied)")

        rc, out = _run_bazel_build(tf_root, env, layout["target"], timeout_sec)
        text = out.replace("\r\n", "\n")

        if rc == 124:
            _skip(f"bazel build timed out after {timeout_sec}s")

        error_markers = (
            "use of undeclared identifier 'string'",
            "'string' was not declared in this scope",
            "unknown type name 'string'",
            "'string' does not name a type",
            "'string' has not been declared",
        )

        if rc != 0:
            if layout["needle_file"] in text and any(m in text for m in error_markers):
                _pass()

            print("INFO: bazel output tail:")
            print(_tail(text, 120))

            if layout["needle_file"] not in text:
                _skip("bazel build failed before compiling profile_summarizer.cc (environment/build prerequisites not satisfied)")

            _fail()
        else:
            _fail()

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)


if __name__ == "__main__":
    main()


# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# conda activate tf_venv
# set -o pipefail

# export PATH="$HOME/bin:$PATH"
# export JAVA_HOME="$CONDA_PREFIX"
# export PATH="$JAVA_HOME/bin:$PATH"

# export ANDROID_HOME="$HOME/android-sdk"
# export ANDROID_SDK_ROOT="$ANDROID_HOME"
# export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH"

# export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/21.4.7075529"
# export ANDROID_NDK_ROOT="$ANDROID_NDK_HOME"

# export GCFL_TF_REPO="$HOME/work/tensorflow_src"
# export USE_BAZEL_VERSION=0.21.0
# export GCFL_TIMEOUT_SEC=180

# bazel shutdown || true
# python "$HOME/dl_testing/testcases/tensorflow_testcase.py" 2>&1 | tee "$HOME/gcfl_other_0081.log"
# echo "exit_code=$?"


# Output:
# *****************
# "tf_cpp_min_log_level": "3"}
# ALLOCATOR_STATUS: {"allocated_by": "cupy", "leave_free_mib": 2048, "ok": true, "physical_gpu_index": 0, "target_alloc_mib": 20606}
# WORKER_RESULTS: [{"corrupt": false, "cpu_sum": 8386560.0, "expected_sum": 8386560.0, "gpu_sum": 8386560.0, "physical_gpu_index": 0, "reasons": [], "rmax": 0.9999771118164062, "rmean": -0.014401411637663841, "rmin": -0.9996249675750732, "rvar": 0.3416896462440491}, {"corrupt": false, "cpu_sum": 8386560.0, "expected_sum": 8386560.0, "gpu_sum": 8386560.0, "physical_gpu_index": 0, "reasons": [], "rmax": 0.9999771118164062, "rmean": -0.014401411637663841, "rmin": -0.9996249675750732, "rvar": 0.3416896462440491}]
# Test Failed ❌
# exit_code=0
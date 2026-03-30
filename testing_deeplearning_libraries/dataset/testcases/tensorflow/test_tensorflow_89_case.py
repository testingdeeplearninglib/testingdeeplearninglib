# GCFL-OTHER-0089

import os
import sys
import random
import argparse
import platform
import shutil


def _skip(reason: str):
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass():
    print("Test Passed ✅")
    sys.exit(0)


def _fail():
    print("Test Failed ❌")
    sys.exit(0)


def _harness_error(e: BaseException):
    print(f"HARNESS_ERROR: {e.__class__.__name__}: {e}")
    sys.exit(1)


def _find_tf_repo_root(start_dir: str, max_up: int = 8) -> str | None:
    d = os.path.abspath(start_dir)
    for _ in range(max_up + 1):
        script = os.path.join(d, "tensorflow", "lite", "lib_package", "create_ios_frameworks.sh")
        if os.path.isfile(script):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _resolve_repo_root(cli_tf_src: str | None) -> str | None:
    candidates = []

    if cli_tf_src:
        candidates.append(cli_tf_src)

    env_tf_src = os.environ.get("TF_SOURCE_ROOT")
    if env_tf_src:
        candidates.append(env_tf_src)

    here = os.path.abspath(os.path.dirname(__file__))
    candidates.append(here)
    candidates.append(os.getcwd())

    for c in candidates:
        root = _find_tf_repo_root(c)
        if root:
            return root
    return None


def main():
    random.seed(2021)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--tf-src",
        type=str,
        default=None,
        help="Path to TensorFlow source root containing tensorflow/lite/lib_package/create_ios_frameworks.sh",
    )
    args = parser.parse_args()

    # This testcase targets an iOS Metal packaging bug, not Linux CUDA runtime execution.
    if sys.platform != "darwin":
        _skip(
            f"requires macOS/iOS Metal build environment; current platform is {platform.system()} {platform.release()}"
        )

    # Xcode CLI tools are expected for iOS framework build flows.
    if shutil.which("xcrun") is None:
        _skip("xcrun not found; install Xcode command line tools")

    repo_root = _resolve_repo_root(args.tf_src)
    if not repo_root:
        _skip(
            "TensorFlow source tree not found "
            "(expected tensorflow/lite/lib_package/create_ios_frameworks.sh). "
            "Pass --tf-src /path/to/tensorflow"
        )

    script_path = os.path.join(
        repo_root, "tensorflow", "lite", "lib_package", "create_ios_frameworks.sh"
    )

    try:
        with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
            script_text = f.read()
    except Exception as e:
        _skip(f"cannot read create_ios_frameworks.sh ({e})")

    if "libmetal_delegate.a" not in script_text:
        _skip("script does not reference libmetal_delegate.a; not the affected version")

    # Best-effort sanity check to avoid trivial false positives on unrelated trees.
    make_tools_dir = os.path.join(repo_root, "tensorflow", "lite", "tools", "make")
    if not os.path.isdir(make_tools_dir):
        _skip("TensorFlow Lite make tools directory not found; unexpected source layout")

    candidate_rel_paths = [
        os.path.join("tensorflow", "lite", "delegates", "gpu", "libmetal_delegate.a"),
        os.path.join("tensorflow", "lite", "delegates", "gpu", "metal", "libmetal_delegate.a"),
        os.path.join("tensorflow", "lite", "delegates", "gpu", "metal_delegate", "libmetal_delegate.a"),
    ]

    existing = []
    for rel in candidate_rel_paths:
        p = os.path.join(repo_root, rel)
        if os.path.isfile(p):
            existing.append(p)

    # For the affected macOS source-tree version, missing libmetal_delegate.a is the bug signal.
    if not existing:
        _pass()
    else:
        _fail()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# conda activate tf_venv
# cd ~/dl_testing
# python testcase/tensorflow_testcase.py
# echo "exit_code=$?"


# Output:
# *****************
# SKIP_ENV: requires macOS/iOS Metal build environment; current platform is Linux 6.8.0-31-generic
# exit_code=0
# # Test Failed ❌
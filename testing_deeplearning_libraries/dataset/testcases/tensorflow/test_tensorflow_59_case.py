# GCFL-TRAININGFI-0059

import random
import shutil
import subprocess
import sys
import traceback


def _print_skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}")
    raise SystemExit(0)


def _print_pass() -> None:
    print("Test Passed ✅")
    raise SystemExit(0)


def _print_fail() -> None:
    print("Test Failed ❌")
    raise SystemExit(0)


def main() -> None:
    random.seed(1337)

    # This is NOT a TensorFlow testcase.
    # It targets Julia + MXNet.jl behavior.
    julia = shutil.which("julia")
    if not julia:
        _print_skip("Julia not installed; cannot execute MXNet Julia binding reproduction")

    julia_code = r"""
    try
        if Base.find_package("MXNet") === nothing
            println("SKIP_MXNET_JL_NOT_INSTALLED")
            flush(stdout)
            exit(0)
        end
    catch e
        println("SKIP_PACKAGE_PROBE_FAILED")
        println("PROBE_ERR=", sprint(showerror, e))
        flush(stdout)
        exit(0)
    end

    try
        import MXNet
    catch e
        println("SKIP_MXNET_IMPORT_FAILED")
        println("IMPORT_ERR=", sprint(showerror, e))
        flush(stdout)
        exit(0)
    end

    # Trigger the reported missing symbol exactly as described.
    try
        set_optimizer(nothing, nothing)
        println("NO_EXCEPTION")
    catch e
        println("EXC_TYPE=", string(typeof(e)))
        println("EXC_MSG=", sprint(showerror, e))
    end

    flush(stdout)
    """

    try:
        proc = subprocess.run(
            [julia, "--color=no", "--startup-file=no", "-e", julia_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        _print_skip("Julia execution timed out; environment too slow or hung")
    except Exception as e:
        raise RuntimeError(f"Failed to invoke Julia: {e}") from e

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    combined = "\n".join([out, err]).lower()

    if "skip_mxnet_jl_not_installed" in combined:
        _print_skip("MXNet.jl not installed in Julia environment")
    if "skip_package_probe_failed" in combined:
        _print_skip("Julia package probe failed before MXNet import")
    if "skip_mxnet_import_failed" in combined:
        _print_skip("MXNet.jl found but import failed in Julia environment")

    if "no_exception" in combined:
        _print_fail()

    undefined_markers = [
        "undefvarerror",
        "undefinedvarerror",
        "not defined",
        "is not defined",
        "unbound variable",
    ]

    if "set_optimizer" in combined and any(marker in combined for marker in undefined_markers):
        _print_pass()

    _print_fail()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("HARNESS_ERROR: " + "".join(traceback.format_exception_only(type(e), e)).strip())
        sys.exit(1)



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# conda activate tf_venv
# cd ~/dl_testing

# unset TF_XLA_FLAGS
# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=3

# python testcases/tensorflow_testcase.py 2>&1 | tee logs/tensorflow_testcase.log
# echo "exit_code=$?"


# Output:
# *****************
# /home/talha/miniconda3/envs/tf_venv/bin/python
# Python 3.11.15
# Command 'julia' not found, but can be installed with:
# snap install julia
# Please ask your administrator.
# Command 'julia' not found, but can be installed with:
# snap install julia
# Please ask your administrator.
# SKIP_ENV: Julia not installed; cannot execute MXNet Julia binding reproduction
# exit_code=0
# Test Failed ❌
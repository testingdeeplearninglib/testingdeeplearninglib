# GCFL-OTHER-0078

import os
import sys
import traceback
import subprocess


def _skip(reason: str):
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass():
    print("Test Passed ✅")
    sys.exit(0)


def _fail():
    print("Test Failed ❌")
    sys.exit(0)


def _harness_error(exc: BaseException):
    print(f"HARNESS_ERROR: {type(exc).__name__}: {exc}")
    sys.exit(1)


def _is_crash_returncode(rc: int) -> bool:
    # Unix: negative rc or 128+signal often indicates crash by signal.
    # Windows: access violation often shows as a large positive code.
    if rc is None:
        return False
    if rc < 0:
        return True
    if rc >= 128:
        return True
    if rc > 255:
        return True
    return False


def _extract_child_marker(stdout: str, stderr: str) -> str:
    markers = ("CHILD_SKIP:", "CHILD_OK", "CHILD_EXCEPTION:")
    for stream in (stdout or "", stderr or ""):
        lines = [line.strip() for line in stream.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith(markers):
                return line
    return ""


def _run_child() -> None:
    try:
        # Must be set before importing TensorFlow
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("PYTHONHASHSEED", "0")
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

        try:
            import numpy as np
        except Exception as e:
            print(f"CHILD_SKIP: numpy not available: {e}")
            sys.exit(0)

        try:
            import tensorflow as tf
        except Exception as e:
            print(f"CHILD_SKIP: tensorflow not available: {e}")
            sys.exit(0)

        try:
            np.random.seed(2021)
        except Exception:
            pass

        try:
            if hasattr(tf, "random") and hasattr(tf.random, "set_seed"):
                tf.random.set_seed(2021)
        except Exception:
            pass

        try:
            v1 = tf.compat.v1 if hasattr(tf, "compat") and hasattr(tf.compat, "v1") else tf
        except Exception as e:
            print(f"CHILD_SKIP: tf.compat.v1 not available: {e}")
            sys.exit(0)

        try:
            v1.disable_eager_execution()
        except Exception:
            pass

        try:
            v1.reset_default_graph()
        except Exception:
            pass

        # Require a visible GPU for this testcase
        try:
            gpus = tf.config.list_physical_devices("GPU")
        except Exception:
            gpus = []

        if not gpus:
            print("CHILD_SKIP: no GPU visible to TensorFlow")
            sys.exit(0)

        try:
            dynamic_rnn = v1.nn.dynamic_rnn
        except Exception as e:
            print(f"CHILD_SKIP: dynamic_rnn not available: {e}")
            sys.exit(0)

        try:
            BasicRNNCell = v1.nn.rnn_cell.BasicRNNCell
        except Exception as e:
            print(f"CHILD_SKIP: BasicRNNCell not available: {e}")
            sys.exit(0)

        with tf.device("/GPU:0"):
            x = tf.zeros((0, 1, 1), dtype=tf.float32)
            cell = BasicRNNCell(1)
            outputs, state = dynamic_rnn(cell, x, dtype=tf.float32)

        try:
            Session = v1.Session
            ConfigProto = v1.ConfigProto
            init = v1.global_variables_initializer()
        except Exception as e:
            print(f"CHILD_SKIP: Session/init not available: {e}")
            sys.exit(0)

        config = ConfigProto()
        config.allow_soft_placement = False
        config.log_device_placement = False

        with Session(config=config) as sess:
            sess.run(init)
            sess.run([outputs, state])

        print("CHILD_OK")
        sys.exit(0)

    except BaseException as e:
        msg = "".join(traceback.format_exception_only(type(e), e)).strip()
        print(f"CHILD_EXCEPTION: {msg}")
        sys.exit(0)


def _run_parent() -> None:
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("PYTHONHASHSEED", "0")
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

        try:
            import tensorflow  # noqa: F401
        except Exception as e:
            _skip(f"tensorflow not available: {e}")

        cmd = [sys.executable, os.path.abspath(__file__), "--child"]
        try:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                timeout=120,
                check=False,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            _fail()

        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        marker = _extract_child_marker(out, err)

        if p.returncode != 0:
            if _is_crash_returncode(p.returncode):
                _pass()
            raise RuntimeError(
                f"Child exited with code {p.returncode}. "
                f"marker={marker!r} stdout={out!r} stderr={err!r}"
            )

        if marker.startswith("CHILD_SKIP:"):
            _skip(marker[len("CHILD_SKIP:"):].strip())

        if marker == "CHILD_OK":
            _fail()

        if marker.startswith("CHILD_EXCEPTION:"):
            _fail()

        raise RuntimeError(f"No recognizable child marker found. stdout={out!r} stderr={err!r}")

    except SystemExit:
        raise
    except BaseException as e:
        _harness_error(e)


def main():
    try:
        if "--child" in sys.argv:
            _run_child()
            return
        _run_parent()
    except SystemExit:
        raise
    except BaseException as e:
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
# export LANG=C.UTF-8
# export LC_ALL=C.UTF-8
# export PYTHONIOENCODING=UTF-8
# export TF_USE_LEGACY_KERAS=1
# export CUDA_VISIBLE_DEVICES=0
# export PYTHONUNBUFFERED=1

# python testcase/tensorflow_testcase.py --child 2>&1 | tee gcfl_0072_child.log
# echo "exit_code=$?"


# Output:
# *****************

# 2026-03-11 15:56:28.138957: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
# 2026-03-11 15:56:28.164419: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
# 2026-03-11 15:56:28.164457: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
# 2026-03-11 15:56:28.180186: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2026-03-11 15:56:29.058005: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
# PYTHON: 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0]
# TF: 2.16.2

# Test Failed ❌
# GCFL-OTHER-0073

import os
import sys

# Force CPU path before importing TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _print_and_exit(msg: str, code: int) -> None:
    print(msg)
    sys.exit(code)


def _skip(reason: str) -> None:
    _print_and_exit(f"SKIP_ENV: {reason}", 0)


def _pass() -> None:
    _print_and_exit("Test Passed ✅", 0)


def _fail() -> None:
    _print_and_exit("Test Failed ❌", 0)


def _harness_error(exc: BaseException) -> None:
    _print_and_exit(f"HARNESS_ERROR: {type(exc).__name__}: {exc}", 1)


def _np_import():
    try:
        import numpy as np  # noqa: F401
        return np
    except Exception as e:
        _skip(f"missing numpy ({e})")


def _tf_import():
    try:
        import tensorflow as tf  # noqa: F401
        return tf
    except Exception as e:
        _skip(f"missing tensorflow ({e})")


def _disable_gpu_best_effort(tf):
    try:
        if hasattr(tf, "config") and hasattr(tf.config, "set_visible_devices"):
            tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def _ensure_tf1_graph_mode(tf):
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception as e:
        _skip(f"cannot disable eager execution / no tf.compat.v1 ({e})")


def _categorical_op(tf):
    try:
        if hasattr(tf, "compat") and hasattr(tf.compat, "v1") and hasattr(tf.compat.v1, "multinomial"):
            return "tf.compat.v1.multinomial", tf.compat.v1.multinomial
    except Exception:
        pass

    if hasattr(tf, "multinomial"):
        return "tf.multinomial", tf.multinomial

    try:
        if hasattr(tf, "random") and hasattr(tf.random, "categorical"):
            return "tf.random.categorical", tf.random.categorical
    except Exception:
        pass

    try:
        if (
            hasattr(tf, "compat")
            and hasattr(tf.compat, "v1")
            and hasattr(tf.compat.v1, "random")
            and hasattr(tf.compat.v1.random, "categorical")
        ):
            return "tf.compat.v1.random.categorical", tf.compat.v1.random.categorical
    except Exception:
        pass

    _skip("no multinomial/categorical op available in this TensorFlow build")


def _build_sampling_ops(tf, op_name, op_fn, logits, logits_ls, num_samples):
    if "multinomial" in op_name:
        return (
            op_fn(logits, num_samples, output_dtype=tf.int64),
            op_fn(logits_ls, num_samples, output_dtype=tf.int64),
        )
    return (
        op_fn(logits, num_samples, dtype=tf.int64),
        op_fn(logits_ls, num_samples, dtype=tf.int64),
    )


def _run_case(np, tf, op_name, op_fn, logits_value, num_samples=10):
    num_classes = int(np.asarray(logits_value).shape[1])

    g = tf.Graph()
    with g.as_default():
        try:
            tf.compat.v1.set_random_seed(2021)
        except Exception:
            pass

        with tf.device("/CPU:0"):
            logits = tf.constant(np.asarray(logits_value, dtype=np.float32), dtype=tf.float32)
            logits_ls = tf.nn.log_softmax(logits)

            try:
                samp_base, samp_work = _build_sampling_ops(
                    tf, op_name, op_fn, logits, logits_ls, num_samples
                )
            except Exception as e:
                _skip(f"failed to build sampling op ({op_name}): {e}")

    try:
        config = tf.compat.v1.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True)
    except Exception:
        config = None

    try:
        sess_ctor = tf.compat.v1.Session
    except Exception as e:
        _skip(f"no tf.compat.v1.Session available ({e})")

    try:
        with sess_ctor(graph=g, config=config) as sess:
            base_np, work_np = sess.run([samp_base, samp_work])
    except Exception as e:
        _skip(f"session run failed ({op_name}): {e}")

    try:
        base_np = np.asarray(base_np, dtype=np.int64)
        work_np = np.asarray(work_np, dtype=np.int64)
    except Exception as e:
        _skip(f"unexpected sample output types: {e}")

    if base_np.ndim != 2 or base_np.shape[0] != 1 or base_np.shape[1] != num_samples:
        _skip(f"unexpected base sample shape {base_np.shape} (expected (1,{num_samples}))")
    if work_np.ndim != 2 or work_np.shape[0] != 1 or work_np.shape[1] != num_samples:
        _skip(f"unexpected workaround sample shape {work_np.shape} (expected (1,{num_samples}))")

    return base_np, work_np, num_classes


def _has_out_of_range(samples_np, num_classes: int) -> bool:
    return bool(((samples_np < 0) | (samples_np >= num_classes)).any())


def _all_in_range(samples_np, num_classes: int) -> bool:
    return bool(((samples_np >= 0) & (samples_np < num_classes)).all())


def main():
    np = _np_import()
    tf = _tf_import()

    _disable_gpu_best_effort(tf)
    _ensure_tf1_graph_mode(tf)

    try:
        np.random.seed(2021)
    except Exception:
        pass
    try:
        tf.compat.v1.set_random_seed(2021)
    except Exception:
        pass

    op_name, op_fn = _categorical_op(tf)

    cases = [
        ("pos_extreme", np.array([[1000.0] * 5], dtype=np.float32)),
        ("neg_extreme", np.array([[-1000.0] * 5], dtype=np.float32)),
    ]

    reproduced = False
    for _, logits_val in cases:
        base_np, work_np, num_classes = _run_case(np, tf, op_name, op_fn, logits_val, num_samples=10)
        base_bad = _has_out_of_range(base_np, num_classes)
        work_good = _all_in_range(work_np, num_classes)
        if base_bad and work_good:
            reproduced = True
            break

    if reproduced:
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

# python -m pip show tf-keras
# python -m pip uninstall -y tf-keras
# python -m pip check

# python - <<'PY'
# import tensorflow as tf

# print("TF version:", tf.__version__)
# print("has tf.compat.v1:", hasattr(tf, "compat") and hasattr(tf.compat, "v1"))
# print("has disable_eager_execution:", hasattr(tf.compat.v1, "disable_eager_execution"))
# print("has Session:", hasattr(tf.compat.v1, "Session"))
# print("has tf.compat.v1.multinomial:", hasattr(tf.compat.v1, "multinomial"))
# print("has tf.random.categorical:", hasattr(tf.random, "categorical"))
# PY

# mkdir -p logs
# python testcase/tensorflow_testcase.py 2>&1 | tee logs/GCFL-OTHER-0073.log
# echo "exit_code=$?"


# Output:
# *****************
# Name: tf_keras
# Version: 2.16.0
# Summary: Deep learning for humans.
# Home-page: https://keras.io/
# Author: Keras team
# Author-email: keras-users@googlegroups.com
# License: Apache 2.0
# Location: /home/talha/miniconda3/envs/tf_venv/lib/python3.10/site-packages
# Requires: tensorflow
# Required-by:

# Found existing installation: tf_keras 2.16.0
# Uninstalling tf_keras-2.16.0:
#   Successfully uninstalled tf_keras-2.16.0

# No broken requirements found.

# 2026-03-12 09:10:12.766330: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# TF version: 2.20.0
# has tf.compat.v1: True
# has disable_eager_execution: True
# has Session: True
# has tf.compat.v1.multinomial: True
# has tf.random.categorical: True

# 2026-03-12 09:10:30.407221: E external/local_xla/xla/stream_executor/cuda/cuda_platform.cc:51] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# WARNING:tensorflow:From /home/talha/miniconda3/envs/tf_venv/lib/python3.10/site-packages/tensorflow/python/util/dispatch.py:1264: multinomial (from tensorflow.python.ops.random_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use `tf.random.categorical` instead.
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1773277830.418150 2526449 mlir_graph_optimization_pass.cc:437] MLIR V1 optimization pass is not enabled
# Test Failed ❌
# exit_code=0
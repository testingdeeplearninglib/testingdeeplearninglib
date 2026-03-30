# GCFL-OTHER-0101

import os
import sys
import json
import random


def _print(msg: str):
    print(msg, flush=True)


def _exit(msg: str, code: int):
    _print(msg)
    sys.exit(code)


def _skip(reason: str):
    _exit(f"SKIP_ENV: {reason}", 0)


def _pass():
    _exit("Test Passed ✅", 0)


def _fail():
    _exit("Test Failed ❌", 0)


def _harness_error(e: BaseException):
    _exit(f"HARNESS_ERROR: {type(e).__name__}: {e}", 1)


def _force_eval(x):
    try:
        if hasattr(x, "numpy"):
            _ = x.numpy()
        elif isinstance(x, (list, tuple)):
            for item in x:
                if hasattr(item, "numpy"):
                    _ = item.numpy()
    except Exception:
        pass


def _env(tf):
    return {
        "python": sys.version.split()[0],
        "tensorflow": getattr(tf, "__version__", "unknown"),
        "eager": bool(tf.executing_eagerly()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "visible_gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
    }


def _run_case(label, fn):
    try:
        out = fn()
        _force_eval(out)
        shape = getattr(out, "shape", None)
        dtype = str(getattr(out, "dtype", None))
        return {
            "label": label,
            "status": "NO_EXCEPTION",
            "shape": str(shape),
            "dtype": dtype,
            "error_type": None,
            "error": None,
        }
    except Exception as e:
        return {
            "label": label,
            "status": "RAISED",
            "shape": None,
            "dtype": None,
            "error_type": type(e).__name__,
            "error": str(e),
        }


def main():
    try:
        try:
            import numpy as np
        except Exception as e:
            _skip(f"missing numpy ({e})")

        try:
            import tensorflow as tf
        except Exception as e:
            _skip(f"missing tensorflow ({e})")

        seed = 2021
        random.seed(seed)
        np.random.seed(seed)
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass

        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        _print("ENV: " + json.dumps(_env(tf), sort_keys=True))

        required_tf = [
            ("tf.math", "cumulative_logsumexp"),
            ("tf.math", "cumprod"),
            ("tf.math", "reduce_mean"),
            ("tf.math", "reduce_sum"),
            ("tf.math", "reduce_max"),
            ("tf.math", "reduce_min"),
            ("tf.math", "reduce_prod"),
        ]
        missing = []
        for owner_name, attr in required_tf:
            owner = getattr(tf, owner_name.split(".")[1], None) if owner_name.startswith("tf.") else None
            if owner is None or not hasattr(owner, attr):
                missing.append(f"{owner_name}.{attr}")
        if missing:
            _skip(f"missing required TensorFlow APIs: {missing}")

        x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        vec = tf.constant([0.5, 1.0, 2.0, 4.0], dtype=tf.float32)

        invalid_keepdims_values = [-1, 1, 2]
        valid_keepdims_values = [False, True]

        reduce_ops = [
            ("reduce_mean", tf.math.reduce_mean),
            ("reduce_sum", tf.math.reduce_sum),
            ("reduce_max", tf.math.reduce_max),
            ("reduce_min", tf.math.reduce_min),
            ("reduce_prod", tf.math.reduce_prod),
        ]

        results = []

        # Baseline sanity: valid booleans should work.
        for op_name, op in reduce_ops:
            for keepdims_value in valid_keepdims_values:
                label = f"{op_name}(keepdims={keepdims_value!r})"
                results.append(
                    _run_case(
                        label,
                        lambda op=op, keepdims_value=keepdims_value: op(
                            x, axis=1, keepdims=keepdims_value
                        ),
                    )
                )

        # Target: invalid integer values for a boolean parameter.
        for op_name, op in reduce_ops:
            for keepdims_value in invalid_keepdims_values:
                label = f"{op_name}(keepdims={keepdims_value!r})"
                results.append(
                    _run_case(
                        label,
                        lambda op=op, keepdims_value=keepdims_value: op(
                            x, axis=1, keepdims=keepdims_value
                        ),
                    )
                )

        # Strict-control cases: these should raise on invalid bool-like ints.
        strict_controls = [
            (
                "cumulative_logsumexp(exclusive=-1, reverse=0)",
                lambda: tf.math.cumulative_logsumexp(
                    vec, axis=-1, exclusive=-1, reverse=0
                ),
            ),
            (
                "cumprod(exclusive=-1, reverse=0)",
                lambda: tf.math.cumprod(
                    vec, axis=-1, exclusive=-1, reverse=0
                ),
            ),
            (
                "cumulative_logsumexp(exclusive=1, reverse=2)",
                lambda: tf.math.cumulative_logsumexp(
                    vec, axis=-1, exclusive=1, reverse=2
                ),
            ),
            (
                "cumprod(exclusive=1, reverse=2)",
                lambda: tf.math.cumprod(
                    vec, axis=-1, exclusive=1, reverse=2
                ),
            ),
        ]

        for label, fn in strict_controls:
            results.append(_run_case(label, fn))

        for r in results:
            if r["status"] == "NO_EXCEPTION":
                _print(
                    f"CASE: {r['label']} -> NO_EXCEPTION | shape={r['shape']} dtype={r['dtype']}"
                )
            else:
                _print(
                    f"CASE: {r['label']} -> RAISED {r['error_type']}: {r['error']}"
                )

        valid_bool_failures = [
            r for r in results
            if "keepdims=False" in r["label"] or "keepdims=True" in r["label"]
            if r["status"] != "NO_EXCEPTION"
        ]
        if valid_bool_failures:
            _skip(
                "baseline valid boolean keepdims calls failed unexpectedly; environment or API behavior is abnormal"
            )

        keepdims_invalid_accepted = [
            r for r in results
            if "keepdims=" in r["label"]
            and any(f"keepdims={v!r}" in r["label"] for v in invalid_keepdims_values)
            and r["status"] == "NO_EXCEPTION"
        ]

        strict_controls_rejected = [
            r for r in results
            if (
                "cumulative_logsumexp(" in r["label"]
                or "cumprod(" in r["label"]
            )
            and r["status"] == "RAISED"
        ]

        summary = {
            "invalid_keepdims_accepted_count": len(keepdims_invalid_accepted),
            "strict_controls_rejected_count": len(strict_controls_rejected),
            "accepted_invalid_keepdims_cases": [r["label"] for r in keepdims_invalid_accepted],
        }
        _print("SUMMARY: " + json.dumps(summary, sort_keys=True))

        # Stronger oracle:
        # Reproduce only if invalid keepdims values are accepted while strict bool controls reject.
        if keepdims_invalid_accepted and strict_controls_rejected:
            _pass()
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
# GPU run
# cd ~/dl_testing
# conda activate tf_venv

# export CUDA_VISIBLE_DEVICES=0
# export PYTHONUNBUFFERED=1
# unset TF_XLA_FLAGS

# set -o pipefail
# python testcase/tensorflow_testcase.py 2>&1 | tee logs/GCFL-OTHER-0101_gpu_v3.log
# echo "exit_code=$?"

# CPU run
# cd ~/dl_testing
# conda activate tf_venv

# export CUDA_VISIBLE_DEVICES=""
# export PYTHONUNBUFFERED=1
# unset TF_XLA_FLAGS

# set -o pipefail
# python testcase/tensorflow_testcase.py 2>&1 | tee logs/GCFL-OTHER-0101_cpu_v3.log
# echo "exit_code=$?"


# Output:
# *****************
# GPU output:
# ENV: {"cuda_visible_devices": "0", "eager": true, "python": "3.11.15", "tensorflow": "2.21.0", "visible_gpus": ["/physical_device:GPU:0"]}
# CASE: reduce_mean(keepdims=False) -> NO_EXCEPTION | shape=(2,) dtype=<dtype: 'float32'>
# CASE: reduce_mean(keepdims=True) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_sum(keepdims=False) -> NO_EXCEPTION | shape=(2,) dtype=<dtype: 'float32'>
# CASE: reduce_sum(keepdims=True) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_max(keepdims=False) -> NO_EXCEPTION | shape=(2,) dtype=<dtype: 'float32'>
# CASE: reduce_max(keepdims=True) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_min(keepdims=False) -> NO_EXCEPTION | shape=(2,) dtype=<dtype: 'float32'>
# CASE: reduce_min(keepdims=True) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_prod(keepdims=False) -> NO_EXCEPTION | shape=(2,) dtype=<dtype: 'float32'>
# CASE: reduce_prod(keepdims=True) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_mean(keepdims=-1) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_mean(keepdims=1) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_mean(keepdims=2) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_sum(keepdims=-1) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_sum(keepdims=1) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_sum(keepdims=2) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_max(keepdims=-1) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_max(keepdims=1) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_max(keepdims=2) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_min(keepdims=-1) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_min(keepdims=1) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_min(keepdims=2) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_prod(keepdims=-1) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_prod(keepdims=1) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: reduce_prod(keepdims=2) -> NO_EXCEPTION | shape=(2, 1) dtype=<dtype: 'float32'>
# CASE: cumulative_logsumexp(exclusive=-1, reverse=0) -> RAISED TypeError: Expected bool for argument 'exclusive' not -1.
# CASE: cumprod(exclusive=-1, reverse=0) -> RAISED TypeError: Expected bool for argument 'exclusive' not -1.
# CASE: cumulative_logsumexp(exclusive=1, reverse=2) -> RAISED TypeError: Expected bool for argument 'exclusive' not 1.
# CASE: cumprod(exclusive=1, reverse=2) -> RAISED TypeError: Expected bool for argument 'exclusive' not 1.
# SUMMARY: {"accepted_invalid_keepdims_cases": ["reduce_mean(keepdims=-1)", "reduce_mean(keepdims=1)", "reduce_mean(keepdims=2)", "reduce_sum(keepdims=-1)", "reduce_sum(keepdims=1)", "reduce_sum(keepdims=2)", "reduce_max(keepdims=-1)", "reduce_max(keepdims=1)", "reduce_max(keepdims=2)", "reduce_min(keepdims=-1)", "reduce_min(keepdims=1)", "reduce_min(keepdims=2)", "reduce_prod(keepdims=-1)", "reduce_prod(keepdims=1)", "reduce_prod(keepdims=2)"], "invalid_keepdims_accepted_count": 15, "strict_controls_rejected_count": 4}
# Test Passed ✅
# exit_code=0




# CPU output:
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# E0000 00:00:1773956431.129562 1309071 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# ENV: {"cuda_visible_devices": "", "eager": true, "python": "3.11.15", "tensorflow": "2.21.0", "visible_gpus": []}
# CASE: tf.math.cumulative_logsumexp(exclusive=-1, reverse=0) -> RAISED TypeError: Expected bool for argument 'exclusive' not -1.
# CASE: tf.math.cumprod(exclusive=-1, reverse=0) -> RAISED TypeError: Expected bool for argument 'exclusive' not -1.
# CASE: tf.math.reduce_mean(keepdims=-1) -> NO_EXCEPTION | shape=(1, 1, 2) dtype=<dtype: 'float32'>
# Test Passed ✅
# exit_code=0

# **************************** Reported ✅ ****************************
# Link: 
# https://github.com/tensorflow/tensorflow/issues/112808

# GCFL-TRAININGFI-0045

import inspect
import os
import sys
import warnings


def _print_and_exit(msg: str, code: int) -> None:
    try:
        print(msg)
    finally:
        sys.exit(code)


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("KMP_WARNINGS", "0")
    warnings.filterwarnings("ignore")

    try:
        import tensorflow as tf
    except Exception as e:
        _print_and_exit(f"SKIP_ENV: tensorflow not available ({type(e).__name__}: {e})", 0)

    try:
        try:
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass

        try:
            from absl import logging as absl_logging
            absl_logging.set_verbosity(absl_logging.ERROR)
        except Exception:
            pass

        try:
            tf.random.set_seed(1234)
        except Exception:
            pass

        if not hasattr(tf, "keras") or not hasattr(tf.keras, "losses") or not hasattr(tf.keras.losses, "Dice"):
            _print_and_exit("SKIP_ENV: tf.keras.losses.Dice not available in this TensorFlow/Keras build", 0)

        axes = (1, 2, 3)

        y_true = tf.constant(
            [[[[1.0], [1.0]], [[0.0], [0.0]]],
             [[[1.0], [1.0]], [[0.0], [0.0]]]],
            dtype=tf.float32,
        )
        y_pred = tf.constant(
            [[[[0.0], [1.0]], [[0.0], [1.0]]],
             [[[0.4], [0.0]], [[0.0], [0.9]]]],
            dtype=tf.float32,
        )

        inter = tf.reduce_sum(y_true * y_pred, axis=axes)
        denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
        per_sample_ref = 1.0 - (2.0 * inter) / denom

        if per_sample_ref.shape.rank != 1 or per_sample_ref.shape[0] != 2:
            _print_and_exit("HARNESS_ERROR: reference computation did not produce expected shape (2,)", 1)

        try:
            sig = inspect.signature(tf.keras.losses.Dice.__init__)
            has_axis = "axis" in sig.parameters
        except Exception:
            has_axis = True

        if not has_axis:
            out = tf.keras.losses.Dice()(y_true, y_pred)
            if out.shape.rank == 0:
                _print_and_exit("Test Passed ✅", 0)
            _print_and_exit("Test Failed ❌", 0)

        out = tf.keras.losses.Dice(axis=axes, reduction=None)(y_true, y_pred)

        if out.shape.rank == 0:
            _print_and_exit("Test Passed ✅", 0)

        if out.shape.rank != 1 or out.shape[0] != 2:
            _print_and_exit("Test Passed ✅", 0)

        out = tf.cast(out, tf.float32)
        per_sample_ref = tf.cast(per_sample_ref, tf.float32)

        max_abs_err = tf.reduce_max(tf.abs(out - per_sample_ref))
        if bool(max_abs_err.numpy() > 1e-6):
            _print_and_exit("Test Passed ✅", 0)

        default_out = tf.keras.losses.Dice()(y_true, y_pred)
        if default_out.shape.rank != 0:
            _print_and_exit("HARNESS_ERROR: unexpected default Dice() output shape; expected scalar", 1)

        _print_and_exit("Test Failed ❌", 0)

    except SystemExit:
        raise
    except Exception as e:
        _print_and_exit(f"HARNESS_ERROR: {type(e).__name__}: {e}", 1)


if __name__ == "__main__":
    main()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# conda activate tf_venv
# cd ~/dl_testing

# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=3
# export PYTHONUNBUFFERED=1
# unset TF_XLA_FLAGS

# set -o pipefail
# python testcases/tensorflow_testcase.py 2>&1 | tee logs/tensorflow_gcfl_trainingfi_0045.log
# echo "exit_code=$?"


# Re-run confirmation commands
# cd ~/dl_testing

# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=3
# export PYTHONUNBUFFERED=1
# unset TF_XLA_FLAGS

# set -o pipefail
# python testcases/tensorflow_testcase.py 2>&1 | tee logs/tensorflow_gcfl_trainingfi_0045_rerun.log
# echo "exit_code=$?"



# Output:
# *****************
# Testcase ID: GCFL-TRAININGFI-0045
# Reference issue: keras-team/keras #19637
# Environment: tf_venv
# TensorFlow version: 2.21.0
# GPU used: GPU:0 (NVIDIA GeForce RTX 3090)
# Observed result: Test Failed ❌
# Exit code: 0
# Conclusion: historical issue not reproduced on the tested stack
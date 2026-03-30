# GCFL-TRAININGFI-0103

# bug no: 56132
# GCFL-TRAININGFI-0103
# Phenomenon: SGD with momentum optimizer update fails for variables with dynamic shape

import os
import sys
import traceback
import random


def _print_and_exit(msg: str, code: int) -> None:
    print(msg)
    raise SystemExit(code)


def skip(reason: str) -> None:
    _print_and_exit(f"SKIP_ENV: {reason}", 0)


def passed() -> None:
    _print_and_exit("Test Passed ✅", 0)


def failed() -> None:
    _print_and_exit("Test Failed ❌", 0)


def harness_error(e: BaseException) -> None:
    _print_and_exit(f"HARNESS_ERROR: {e}", 1)


def _set_determinism() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    random.seed(0)


def _run_case(tf, momentum: float, use_tf_function: bool) -> None:
    """
    Runs a single apply_gradients step on a dynamically-shaped variable.
    Should succeed without momentum; historically fails with momentum due to slot creation.
    """
    tf.random.set_seed(0)

    x = tf.Variable(tf.random.normal((32, 3)), shape=[None, 3])
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=momentum)

    def body():
        with tf.GradientTape() as tape:
            # Change runtime shape along the dynamic dimension.
            x.assign(tf.random.normal((20, 3)))
            y = tf.reduce_sum(x)
        grads = tape.gradient(y, x)
        if grads is None:
            raise RuntimeError("Gradient is None")
        opt.apply_gradients([(grads, x)])

    if use_tf_function:
        tf.function(body)()
    else:
        body()


def main() -> None:
    _set_determinism()

    try:
        import tensorflow as tf  # noqa: F401
    except Exception as e:
        skip(f"tensorflow not available: {e}")

    try:
        import tensorflow as tf
        try:
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass

        # Try both eager and tf.function to maximize reproducibility across TF versions.
        modes = [("eager", False), ("tf_function", True)]

        for _, use_tf_function in modes:
            # Baseline: no momentum should succeed
            try:
                _run_case(tf, momentum=0.0, use_tf_function=use_tf_function)
            except Exception:
                # If baseline fails, this does not match the phenomenon; try next mode.
                continue

            # Bug trigger: momentum should raise an exception (historically ValueError about partial TensorShape)
            try:
                _run_case(tf, momentum=0.9, use_tf_function=use_tf_function)
            except Exception:
                passed()

        failed()

    except SystemExit:
        raise
    except Exception as e:
        harness_error(e)


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

# export TF_CPP_MIN_LOG_LEVEL=3
# export KERAS_BACKEND=tensorflow
# export CUDA_VISIBLE_DEVICES=""

# python testcases/tensorflow_testcase.py \
#   > logs/56132_cpu_stdout.log \
#   2> logs/56132_cpu_stderr.log
  
  


# Output:
# *****************
# Standard Output
# $(cat logs/56132_cpu_stdout.log)
# Standard Error
# $(cat logs/56132_cpu_stderr.log)

# TF version: 2.21.0
# Visible GPUs: []
# RUN momentum=0.0 step1_rows=32
# RUN momentum=0.0 step2_rows=20
# BASELINE_OK momentum=0.0
# RUN momentum=0.9 step1_rows=32
# TRIGGER_EXCEPTION: ValueError: Shapes used to initialize variables must be fully-defined (no `None` dimensions). Received: shape=(None, 3) for variable path='SGD/Variable_0_momentum'

# Test Passed ✅


# Already Reported

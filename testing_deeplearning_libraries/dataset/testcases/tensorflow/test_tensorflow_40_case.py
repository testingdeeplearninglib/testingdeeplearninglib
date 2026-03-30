# GCFL-TRAININGFI-0040

import os
import sys
import random


def _print_and_exit(msg: str, code: int) -> None:
    try:
        print(msg)
    finally:
        sys.exit(code)


def skip(reason: str) -> None:
    _print_and_exit(f"SKIP_ENV: {reason}", 0)


def passed() -> None:
    _print_and_exit("Test Passed ✅", 0)


def failed() -> None:
    _print_and_exit("Test Failed ❌", 0)


def harness_error(exc: BaseException) -> None:
    _print_and_exit(f"HARNESS_ERROR: {type(exc).__name__}: {exc}", 1)


def iter_exception_chain(e: BaseException):
    seen = set()
    stack = [e]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        obj_id = id(cur)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield cur
        cause = getattr(cur, "__cause__", None)
        context = getattr(cur, "__context__", None)
        if cause is not None:
            stack.append(cause)
        if context is not None:
            stack.append(context)


def is_expected_operator_not_allowed(e: BaseException) -> bool:
    target_substrings = [
        "OperatorNotAllowedInGraphError",
        "Using a symbolic `tf.Tensor` as a Python `bool` is not allowed",
        "Using a symbolic tf.Tensor as a Python bool is not allowed",
    ]
    for ex in iter_exception_chain(e):
        name = type(ex).__name__
        msg = str(ex)
        if name == "OperatorNotAllowedInGraphError":
            return True
        for s in target_substrings:
            if s in msg:
                return True
    return False


def exception_summary(e: BaseException) -> str:
    parts = []
    for ex in iter_exception_chain(e):
        parts.append(f"{type(ex).__name__}: {str(ex)}")
    return " | ".join(parts)


def get_pkg_version(pkg_name: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return "unknown"


def main() -> None:
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    seed = 123
    random.seed(seed)

    try:
        import numpy as np
    except Exception:
        skip("numpy is required")

    np.random.seed(seed)

    try:
        import tensorflow as tf
    except Exception as e:
        skip(f"tensorflow import failed: {type(e).__name__}: {e}")

    try:
        import keras
    except Exception as e:
        skip(f"keras import failed: {type(e).__name__}: {e}")

    try:
        if hasattr(keras.utils, "set_random_seed"):
            keras.utils.set_random_seed(seed)
        else:
            tf.random.set_seed(seed)
    except Exception:
        pass

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    backend_name = None
    try:
        if hasattr(keras, "backend") and hasattr(keras.backend, "backend"):
            backend_name = keras.backend.backend()
        elif hasattr(keras, "config") and hasattr(keras.config, "backend"):
            backend_name = keras.config.backend()
    except Exception:
        backend_name = None

    if backend_name != "tensorflow":
        skip(f"TensorFlow backend required (current backend: {backend_name})")

    try:
        gpus = tf.config.list_physical_devices("GPU")
        gpu_names = [getattr(g, "name", str(g)) for g in gpus]
    except Exception:
        gpu_names = []

    print(
        "ENV:",
        {
            "python": sys.version.split()[0],
            "tensorflow": getattr(tf, "__version__", "unknown"),
            "keras": get_pkg_version("keras"),
            "backend": backend_name,
            "gpu_count_visible_to_tf": len(gpu_names),
            "gpu_names": gpu_names,
            "seed": seed,
        },
    )

    n_samples = 8
    n_features = 3
    n_targets = 2

    rng = np.random.RandomState(seed)
    train_x = rng.random((n_samples, n_features)).astype("float32")
    train_y = rng.random((n_samples, n_targets)).astype("float32")

    def build_model():
        try:
            keras.backend.clear_session()
        except Exception:
            pass

        inputs = keras.layers.Input(shape=(n_features,))
        outputs = keras.layers.Dense(
            n_targets,
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
            bias_initializer="zeros",
        )(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    model_control = build_model()
    try:
        model_control.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer="adam",
            run_eagerly=False,
            jit_compile=False,
        )
        model_control.fit(
            train_x,
            train_y,
            epochs=1,
            batch_size=4,
            shuffle=False,
            verbose=0,
        )
        print("CONTROL: instantiated loss path completed successfully")
    except Exception as e:
        print("CONTROL_EXCEPTION:", exception_summary(e))
        skip(f"control path failed (instantiated loss): {type(e).__name__}")

    model_bug = build_model()
    try:
        model_bug.compile(
            loss="MeanSquaredError",
            optimizer="adam",
            run_eagerly=False,
            jit_compile=False,
        )
        model_bug.fit(
            train_x,
            train_y,
            epochs=1,
            batch_size=4,
            shuffle=False,
            verbose=0,
        )
        print("BUG_PATH: no exception raised")
        failed()
    except Exception as e:
        print("BUG_EXCEPTION:", exception_summary(e))
        if is_expected_operator_not_allowed(e):
            passed()
        else:
            failed()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        harness_error(e)



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# conda activate tf_venv
# cd ~/dl_testing

# export CUDA_VISIBLE_DEVICES=0
# export KERAS_BACKEND=tensorflow
# export TF_CPP_MIN_LOG_LEVEL=2

# python testcases/tensorflow_testcase.py 2>&1 | tee logs/issue_19333/run.log
# echo "exit_code=$?"

# grep -E "ENV:|CONTROL:|CONTROL_EXCEPTION:|BUG_PATH:|BUG_EXCEPTION:|Test Passed|Test Failed|SKIP_ENV|HARNESS_ERROR" logs/issue_19333/run.log


# Output:
# *****************
# ENV: {'python': '3.11.15', 'tensorflow': '2.21.0', 'keras': '3.13.2', 'backend': 'tensorflow', 'gpu_count_visible_to_tf': 1, 'gpu_names': ['/physical_device:GPU:0'], 'seed': 123}
# CONTROL: instantiated loss path completed successfully
# BUG_PATH: no exception raised
# Test Failed ❌
# exit_code=0
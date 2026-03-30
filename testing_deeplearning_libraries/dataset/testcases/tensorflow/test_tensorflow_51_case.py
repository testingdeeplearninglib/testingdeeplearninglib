# GCFL-OTHER-0051

import os
import sys
import traceback
import random

# Prefer TF backend unless caller explicitly sets otherwise.
# Must happen BEFORE importing keras.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

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
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

def _is_keyerror_zero(e: BaseException) -> bool:
    if not isinstance(e, KeyError):
        return False
    # Most robust: KeyError keeps the missing key in args
    if getattr(e, "args", None):
        return e.args == (0,) or e.args == ("0",)
    s = str(e).strip()
    return s in ("0", "'0'")

def _get_keras_backend_name(keras_mod):
    # Best-effort across versions / backends.
    try:
        return keras_mod.backend.backend()
    except Exception:
        pass
    try:
        fn = getattr(getattr(keras_mod, "backend", None), "backend", None)
        return fn() if callable(fn) else None
    except Exception:
        return None

def main():
    try:
        seed = 1337
        random.seed(seed)

        try:
            import numpy as np
        except Exception as e:
            _skip(f"numpy not available: {e}")
        np.random.seed(seed)

        try:
            import tensorflow as tf
        except Exception as e:
            _skip(f"tensorflow not available: {e}")

        try:
            import keras
        except Exception as e:
            _skip(f"keras not available: {e}")

        # If ops.map is missing for any reason, skip rather than harness-crash
        if not hasattr(keras, "ops") or not hasattr(keras.ops, "map"):
            _skip("keras.ops.map not available in this keras build")

        resolved_backend = _get_keras_backend_name(keras)

        print(f"PYTHON: {sys.version.split()[0]}")
        print(f"KERAS_VERSION: {getattr(keras, '__version__', 'unknown')}")
        print(f"TF_VERSION: {getattr(tf, '__version__', 'unknown')}")
        print(f"KERAS_BACKEND_ENV: {os.environ.get('KERAS_BACKEND', '')}")
        print(f"KERAS_BACKEND_RESOLVED: {resolved_backend}")
        try:
            print(f"TF_GPUS: {tf.config.list_physical_devices('GPU')}")
        except Exception:
            print("TF_GPUS: <unavailable>")

        # Basic sanity: ensure backend is tensorflow (best-effort)
        try:
            if resolved_backend and str(resolved_backend).lower() not in ("tensorflow", "tf"):
                _skip(f"KERAS_BACKEND not tensorflow (got {resolved_backend})")
        except Exception:
            pass

        # Define fn that returns nested output structure (dict), as in excerpt.
        def my_fn(inputs):
            outputs = dict(inputs)
            outputs["x"] = inputs["x"][:, 0]
            outputs["y"] = inputs["y"] + 1
            return outputs

        # Nested xs that is NOT indexable by int at top-level -> xs[0] KeyError
        xs = {
            "x": tf.convert_to_tensor(np.random.rand(4, 100, 3), dtype=tf.float32),
            "y": tf.convert_to_tensor(np.random.rand(4, 5), dtype=tf.float32),
        }

        # Bug reproduction oracle: exception (KeyError: 0) during signature inference
        try:
            out = keras.ops.map(my_fn, xs)

            # If no exception, bug did not reproduce
            # (Attempt a light materialization if it's a dict-like output)
            try:
                if hasattr(out, "items"):
                    _ = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in out.items()}
            except Exception:
                pass

            _fail()

        except Exception as e:
            # Print traceback to help you paste into the GitHub issue
            traceback.print_exc()

            if _is_keyerror_zero(e):
                _pass()

            # Sometimes wrapped; be conservative but still allow clear "KeyError: 0"
            msg = f"{type(e).__name__}: {e}"
            if ("KeyError" in msg) and (": 0" in msg or " 0" in msg or "'0'" in msg):
                _pass()

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

# Keras


# Commands
# *****************
# KERAS_BACKEND=tensorflow TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES="" \
# python testcases/keras_testcase.py 2>&1 | tee testcases/keras_testcase.repro.log



# Output:
# *****************
# Traceback (most recent call last):
#   File "/home/talha/dl_testing/testcases/keras_testcase.py", line 109, in main
#     out = keras.ops.map(my_fn, xs)
#   File "/home/talha/miniconda3/envs/keras_venv/lib/python3.10/site-packages/keras/src/ops/core.py", line 91, in map
#     return backend.core.map(f, xs)
#   File "/home/talha/miniconda3/envs/keras_venv/lib/python3.10/site-packages/keras/src/backend/tensorflow/core.py", line 228, in map
#     fn_output_signature = get_fn_output_signature(xs[0])
# KeyError: 0

# PYTHON: 3.10.19
# KERAS_VERSION: 3.4.1
# TF_VERSION: 2.17.0
# KERAS_BACKEND_ENV: tensorflow
# KERAS_BACKEND_RESOLVED: tensorflow
# TF_GPUS: []
# Test Passed ✅

# already posted
# Test Failed
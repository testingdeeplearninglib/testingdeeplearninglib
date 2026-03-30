# GCFL-OTHER-0041

import sys
import os
import traceback
import random
import math


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
    sys.exit(1)


def _set_global_seeds(seed: int):
    random.seed(seed)
    try:
        import numpy as np  # noqa
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf  # noqa
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass
    except Exception:
        pass


def _to_numpy(keras, out):
    """
    Backend-robust conversion:
    - TF: EagerTensor.numpy()
    - Torch: detach().cpu().numpy() (works even if CUDA tensor)
    - Keras ops: keras.ops.convert_to_numpy
    - Fallback: np.asarray
    """
    # 1) TensorFlow-style
    try:
        if hasattr(out, "numpy"):
            return out.numpy()
    except Exception:
        pass

    # 2) PyTorch-style (important on GPU: .numpy() fails on CUDA tensors)
    try:
        import torch  # type: ignore
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().numpy()
    except Exception:
        pass

    # 3) Keras ops conversion (Keras 3)
    try:
        return keras.ops.convert_to_numpy(out)
    except Exception:
        pass

    # 4) Final fallback
    try:
        import numpy as np
        return np.asarray(out)
    except Exception as e:
        raise TypeError(f"Unable to convert initializer output to numpy: {e}")


def _sample_initializer_outputs(keras, initializer_name: str, seed_value, n_runs: int = 3):
    """
    Create the initializer n_runs times with the same seed_value and return flattened outputs.
    Uses a fixed shape to compare results across runs.
    """
    outputs = []
    init_mod = keras.initializers

    if not hasattr(init_mod, initializer_name):
        raise AttributeError(f"keras.initializers has no attribute '{initializer_name}'")

    init_cls = getattr(init_mod, initializer_name)
    shape = (4, 4)

    for _ in range(n_runs):
        init = init_cls(seed=seed_value)
        out = init(shape=shape, dtype="float32")

        arr = _to_numpy(keras, out)

        import numpy as np
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)

        # Convert to stable python floats
        outputs.append([float(x) for x in arr])

    return outputs


def _all_equal(a, b, tol=0.0):
    if len(a) != len(b):
        return False

    if tol == 0.0:
        for x, y in zip(a, b):
            # avoid rare false “nondeterministic” on NaNs (NaN != NaN)
            if (math.isnan(x) and math.isnan(y)):
                continue
            if x != y:
                return False
        return True

    for x, y in zip(a, b):
        if math.isnan(x) and math.isnan(y):
            continue
        if abs(x - y) > tol:
            return False
    return True


def main():
    try:
        # IMPORTANT (Keras 3): backend must be set BEFORE importing keras (do it in shell via KERAS_BACKEND).
        # https://keras.io/getting_started/  :contentReference[oaicite:0]{index=0}

        try:
            import keras  # type: ignore
        except Exception as e_keras:
            try:
                import tensorflow as tf  # type: ignore
                keras = tf.keras
            except Exception as e_tf:
                _skip(f"keras not available (keras import: {e_keras}; tf.keras import: {e_tf})")

        _set_global_seeds(12345)

        # Target: seed=0 treated as falsy => ignored, random seed chosen => nondeterministic
        seed_value = 0

        candidates = ["RandomNormal", "TruncatedNormal", "RandomUniform"]

        reproduced = False
        details = []

        for name in candidates:
            try:
                outs = _sample_initializer_outputs(keras, name, seed_value, n_runs=3)
            except Exception as e:
                details.append((name, f"SKIP_INIT: {type(e).__name__}: {e}"))
                continue

            eq01 = _all_equal(outs[0], outs[1], tol=0.0)
            eq02 = _all_equal(outs[0], outs[2], tol=0.0)

            if not (eq01 and eq02):
                reproduced = True
                details.append((name, "NONDETERMINISTIC_WITH_SEED_0"))
            else:
                details.append((name, "DETERMINISTIC_WITH_SEED_0"))

        if os.environ.get("GCFL_VERBOSE") == "1":
            try:
                import keras as _k
                print("DEBUG: keras.__version__ =", getattr(_k, "__version__", "unknown"))
                print("DEBUG: KERAS_BACKEND =", os.environ.get("KERAS_BACKEND", ""))
            except Exception:
                pass
            try:
                import tensorflow as _tf
                print("DEBUG: tf.__version__ =", _tf.__version__)
                print("DEBUG: GPUs =", _tf.config.list_physical_devices("GPU"))
            except Exception:
                pass
            print("DEBUG: details =", details)

        testable = [d for d in details if not str(d[1]).startswith("SKIP_INIT")]
        if len(testable) == 0:
            _skip("No testable keras random initializers found or runnable in this environment")

        if reproduced:
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

# Keras


# Commands
# *****************
# conda activate tf216_k311
# cd /home/talha/dl_testing

# export KERAS_BACKEND="tensorflow"
# export CUDA_VISIBLE_DEVICES=""
# export GCFL_VERBOSE=1

# python -u /home/talha/dl_testing/testcases/keras_testcase.py 2>&1 | tee /home/talha/dl_testing/run_tf216_k311.txt


# Output:
# *****************
# WARNING:... An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
# DEBUG: keras.__version__ = 3.1.1
# DEBUG: KERAS_BACKEND = jax
# DEBUG: details = [('RandomNormal', 'NONDETERMINISTIC_WITH_SEED_0'), ('TruncatedNormal', 'NONDETERMINISTIC_WITH_SEED_0'), ('RandomUniform', 'NONDETERMINISTIC_WITH_SEED_0')]
# Test Passed


# Test passed but this is already been reported to Github so
# Test Failed
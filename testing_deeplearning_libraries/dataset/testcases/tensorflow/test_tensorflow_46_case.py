# GCFL-OTHER-0046

import os
import sys
import traceback

GCFL_ID = "GCFL-OTHER-0046"


def _skip(reason: str):
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass():
    print("Test Passed ✅")
    sys.exit(0)


def _fail():
    print("Test Failed ❌")
    sys.exit(0)


def _is_div0_exception(e: BaseException) -> bool:
    msg = (str(e) + " " + repr(e)).lower()
    if isinstance(e, ZeroDivisionError):
        return True
    needles = [
        "division by zero",
        "divide by zero",
        "zero division",
        "zerodivision",
        "div by zero",
        "float division by zero",
    ]
    return any(n in msg for n in needles)


def main():
    # Force backend selection BEFORE importing keras.
    # Do NOT use setdefault here: deterministic even if the shell exported KERAS_BACKEND before.
    os.environ["KERAS_BACKEND"] = "jax"

    # Keep execution reproducible and avoid GPU dependency (your current jaxlib is CPU-only anyway).
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    # Reduce JAX preallocation surprises on shared machines (harmless on CPU).
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    seed = 2021

    try:
        import numpy as np
    except Exception as e:
        _skip(f"numpy not available: {e}")

    try:
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import jax
        import jax.numpy as jnp
        import jaxlib
    except Exception as e:
        _skip(f"jax not available: {e}")

    try:
        import keras
    except Exception as e:
        _skip(f"keras not available: {e}")

    # Best-effort deterministic setup
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass

    try:
        try:
            backend = keras.backend.backend()
        except Exception:
            backend = os.environ.get("KERAS_BACKEND", "unknown")

        print(f"GCFL_ID: {GCFL_ID}")
        print(f"PYTHON: {sys.version.split()[0]}")
        print(f"KERAS_VERSION: {getattr(keras, '__version__', 'unknown')}")
        print(f"KERAS_FILE: {getattr(keras, '__file__', 'unknown')}")
        print(f"KERAS_BACKEND_RESOLVED: {backend}")
        print(f"KERAS_BACKEND_ENV: {os.environ.get('KERAS_BACKEND', '')}")
        print(f"JAX_VERSION: {getattr(jax, '__version__', 'unknown')}")
        print(f"JAXLIB_VERSION: {getattr(jaxlib, '__version__', 'unknown')}")
        print(f"JAX_PLATFORM_NAME: {os.environ.get('JAX_PLATFORM_NAME', '')}")
        try:
            print(f"JAX_DEVICES: {jax.devices()}")
        except Exception:
            pass

        if str(backend).lower() != "jax":
            _skip(f"KERAS_BACKEND is not jax (resolved={backend})")

        # Stress input (NHWC): extremely skinny image.
        x = jnp.asarray(np.ones((1, 1, 1000, 3), dtype=np.float32))

        # Baseline sanity: a normal shape should work.
        try:
            layer_sanity = keras.layers.Resizing(
                height=16, width=16, crop_to_aspect_ratio=False
            )
            _ = layer_sanity(jnp.asarray(np.ones((1, 224, 224, 3), dtype=np.float32)))
        except Exception as e:
            print(f"DEBUG_BASELINE_SANITY_EXCEPTION: {type(e).__name__}: {e}")
            _fail()

        # Baseline 1: small target + crop enabled should not crash.
        try:
            layer_small = keras.layers.Resizing(
                height=8, width=8, crop_to_aspect_ratio=True
            )
            _ = layer_small(x)
        except Exception as e:
            print(f"DEBUG_BASELINE_SMALL_EXCEPTION: {type(e).__name__}: {e}")
            _fail()

        # Differential baseline:
        # Edge-case width=0 with crop_to_aspect_ratio=False should NOT crash (observed in probe).
        try:
            layer_no_crop = keras.layers.Resizing(
                height=2000, width=0, crop_to_aspect_ratio=False
            )
            _ = layer_no_crop(x)
        except Exception as e:
            print(f"DEBUG_BASELINE_NO_CROP_EXCEPTION: {type(e).__name__}: {e}")
            _fail()

        # Target:
        # Edge-case width=0 with crop_to_aspect_ratio=True triggers ZeroDivisionError
        # (instead of a clean validation error).
        try:
            layer_bug = keras.layers.Resizing(
                height=2000, width=0, crop_to_aspect_ratio=True
            )
            _ = layer_bug(x)
        except Exception as e:
            print(f"DEBUG_TARGET_EXCEPTION: {type(e).__name__}: {e}")
            if _is_div0_exception(e):
                _pass()
            _fail()

        _fail()

    except SystemExit:
        raise
    except Exception as e:
        print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Keras


# Commands
# *****************
# conda activate keras_venv
# unset KERAS_BACKEND

# KERAS_BACKEND=jax python -c "import keras; print(keras.__version__, keras.backend.backend())"

# python testcases/keras_testcase.py



# Output:
# *****************
# PYTHON: 3.10.19
# KERAS_VERSION: 3.12.1
# KERAS_FILE: /home/talha/miniconda3/envs/keras_venv/lib/python3.10/site-packages/keras/__init__.py
# KERAS_BACKEND_RESOLVED: jax
# KERAS_BACKEND_ENV: jax
# JAX_VERSION: 0.6.2
# JAXLIB_VERSION: 0.6.2
# JAX_PLATFORM_NAME: cpu
# JAX_DEVICES: [CpuDevice(id=0)]
# DEBUG_TARGET_EXCEPTION: ZeroDivisionError: Exception encountered when calling Resizing.call().

# float division by zero

# Arguments received by Resizing.call():
#   • data=jnp.ndarray(shape=(1, 1, 1000, 3), dtype=float32)
#   • training=True
# Test Passed ✅ --- but the bug was already reported so
# # Test Failed ❌

# crop_to_aspect_ratio=False

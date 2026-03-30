# GCFL-OTHER-0052

# CONTROL TEST for GCFL-OTHER-0052
# Purpose: Prove that a proper keras.constraints.Constraint works
# under Keras 3 + torch backend.
#
# Note: Some Keras builds do not expose a reliable "active backend" getter.
# In that case, we do not fail early; we infer backend from runtime behavior.

import os
import sys
import traceback

# Force torch backend BEFORE importing keras
os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _fail(msg):
    print(f"FAIL: {msg}")
    print("Test Failed ❌")
    sys.exit(0)


def _pass():
    print("CONTROL_OK")
    sys.exit(0)


def _harness_error(e: BaseException):
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)


def main():
    try:
        import torch
    except Exception as e:
        _fail(f"torch not available: {e}")

    try:
        import keras
    except Exception as e:
        _fail(f"keras not available: {e}")

    from keras import ops

    print(f"PYTHON: {sys.version.split()[0]}")
    print(f"KERAS_VERSION: {getattr(keras, '__version__', 'unknown')}")
    print(f"TORCH_VERSION: {getattr(torch, '__version__', 'unknown')}")
    print(f"CUDA_AVAILABLE: {torch.cuda.is_available()}")
    print(f"KERAS_BACKEND_ENV: {os.environ.get('KERAS_BACKEND')!r}")

    # Best-effort backend check (do not fail if API not available)
    backend_name = None
    backend_check_note = "unavailable"
    try:
        cfg = getattr(keras.backend, "config", None)
        if cfg and hasattr(cfg, "backend"):
            backend_name = cfg.backend()
            backend_check_note = "keras.backend.config.backend()"
    except Exception:
        backend_name = None

    if backend_name is None:
        try:
            if hasattr(keras.backend, "backend"):
                backend_name = keras.backend.backend()
                backend_check_note = "keras.backend.backend()"
        except Exception:
            backend_name = None

    print(f"KERAS_BACKEND_ACTIVE: {backend_name!r} (check: {backend_check_note})")

    # Runtime inference: what type does ops.ones produce?
    try:
        probe = ops.ones((1, 2), dtype="float32")
        probe_type = type(probe).__name__
        probe_mod = type(probe).__module__
        print(f"OPS_ONES_TYPE: {probe_mod}.{probe_type}")
    except Exception as e:
        _fail(f"Failed to create ops.ones probe: {e}")

    # Proper Constraint subclass
    class MulMask(keras.constraints.Constraint):
        def __init__(self, mask):
            self.mask = mask

        def __call__(self, w):
            return w * self.mask

    class L(keras.layers.Layer):
        def build(self, input_shape):
            # Use torch tensor to avoid type-mismatch issues
            mask = torch.ones((2, 2), dtype=torch.float32, device="cpu")

            self.w = self.add_weight(
                name="w",
                shape=(2, 2),
                initializer="ones",
                trainable=True,
                constraint=MulMask(mask),
            )

        def call(self, inputs):
            return inputs

    try:
        x = ops.ones((1, 2), dtype="float32")
        layer = L()
        _ = layer(x)
    except Exception as e:
        _fail(f"Unexpected exception during layer call: {type(e).__name__}: {e}")

    _pass()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        _harness_error(e)


# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Keras


# Commands
# *****************
# KERAS_BACKEND=torch python testcases/keras_testcase.py




# Output:
# *****************

# PYTHON: 3.10.19
# KERAS_VERSION: 3.12.1
# TORCH_VERSION: 2.10.0+cu128
# CUDA_AVAILABLE: True
# KERAS_BACKEND_ENV: 'torch'
# KERAS_BACKEND_ACTIVE: 'torch'
# EXCEPTION_TYPE: ValueError
# EXCEPTION_MSG: Invalid value for attribute `constraint`. Expected an instance of `keras.constraints.Constraint`, or `None`. Received: constraint=<function ...>
# Test Passed ✅

# PYTHON: 3.10.19
# KERAS_VERSION: 3.12.1
# TORCH_VERSION: 2.10.0+cu128
# CUDA_AVAILABLE: True
# KERAS_BACKEND_ENV: 'torch'
# KERAS_BACKEND_ACTIVE: 'torch'
# OPS_ONES_TYPE: torch.Tensor
# CONTROL_OK





# **************************** Reported ✅ ****************************

# Link: 
# https://github.com/keras-team/keras/issues/22221

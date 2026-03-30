# GCFL-OTHER-0048

# GCFL-OTHER-0048

import os
import sys
import traceback

# Must be set BEFORE importing keras (Keras 3 multi-backend).
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def _skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass() -> None:
    print("Test Passed ✅")
    sys.exit(0)


def _fail() -> None:
    print("Test Failed ❌")
    sys.exit(0)


def _harness_error(e: BaseException) -> None:
    print(f"HARNESS_ERROR: {repr(e)}")
    traceback.print_exc()
    sys.exit(1)


def main() -> None:
    try:
        # -------------------------
        # Imports (skip if missing)
        # -------------------------
        try:
            import numpy as np
        except Exception as e:
            _skip(f"numpy not available: {e}")

        try:
            import keras
            from keras import layers, ops
            from keras import random as k_random
        except Exception as e:
            _skip(f"keras not available or failed to import: {e}")

        # Basic environment prints (helps you debug remote server fast)
        try:
            print(f"KERAS_BACKEND: {os.environ.get('KERAS_BACKEND', 'unset')}")
            print(f"KERAS_VERSION: {getattr(keras, '__version__', 'unknown')}")
        except Exception:
            pass

        # If TF backend, print TF + GPU visibility (non-fatal)
        if os.environ.get("KERAS_BACKEND") == "tensorflow":
            try:
                import tensorflow as tf  # noqa: F401
                print(f"TF_VERSION: {getattr(tf, '__version__', 'unknown')}")
                try:
                    gpus = tf.config.list_physical_devices("GPU")
                    print(f"TF_GPUS: {[d.name for d in gpus]}")
                except Exception as _e:
                    print(f"TF_GPUS: <unavailable> ({_e})")
            except Exception as _e:
                print(f"TF_IMPORT: failed ({_e})")

        # -------------------------
        # Deterministic seeds
        # -------------------------
        seed = 2021
        try:
            import random as py_random
            py_random.seed(seed)
        except Exception:
            pass

        try:
            np.random.seed(seed)
        except Exception:
            pass

        try:
            from keras.utils import set_random_seed
            set_random_seed(seed)
        except Exception:
            pass

        # -------------------------
        # Helpers for (get/set) mask
        # -------------------------
        def _resolve_masking_fns():
            # Return (set_fn, get_fn) or (None, None)
            # Try public-ish locations first, then internal.
            set_fn = None
            get_fn = None

            # Internal path observed in Keras 3 sources
            try:
                from keras.backend.common.masking import set_keras_mask as _set  # type: ignore
                from keras.backend.common.masking import get_keras_mask as _get  # type: ignore
                set_fn, get_fn = _set, _get
                return set_fn, get_fn
            except Exception:
                pass

            # Fallback: sometimes exposed via backend module (version-dependent)
            try:
                from keras import backend as K  # type: ignore
                if hasattr(K, "set_keras_mask") and hasattr(K, "get_keras_mask"):
                    set_fn, get_fn = K.set_keras_mask, K.get_keras_mask  # type: ignore
                    return set_fn, get_fn
            except Exception:
                pass

            return None, None

        _set_keras_mask_fn, _get_keras_mask_fn = _resolve_masking_fns()

        def _try_set_mask(t, m) -> bool:
            # First try direct attribute (works on some tensor objects)
            try:
                setattr(t, "_keras_mask", m)
                return True
            except Exception:
                pass
            # Then try helper fn (more reliable for Keras internals)
            if _set_keras_mask_fn is not None:
                try:
                    _set_keras_mask_fn(t, m)
                    return True
                except Exception:
                    return False
            return False

        def _try_get_mask(t):
            if hasattr(t, "_keras_mask"):
                try:
                    return getattr(t, "_keras_mask")
                except Exception:
                    return None
            if _get_keras_mask_fn is not None:
                try:
                    return _get_keras_mask_fn(t)
                except Exception:
                    return None
            return None

        def _to_numpy(x):
            if x is None:
                return None
            try:
                return ops.convert_to_numpy(x)
            except Exception:
                try:
                    return np.asarray(x)
                except Exception:
                    return None

        # -------------------------
        # Repro (from evidence excerpt)
        # -------------------------
        try:
            a = k_random.uniform([1, 2, 256], dtype="float32")
            b = k_random.uniform([1, 2, 256], dtype="float32")
        except Exception as e:
            _skip(f"unable to create tensors via keras.random.uniform: {e}")

        try:
            mask_a = ops.convert_to_tensor([[True, False]])
            mask_b = ops.convert_to_tensor([[True, False]])
        except Exception as e:
            _skip(f"unable to create masks via keras.ops.convert_to_tensor: {e}")

        if not _try_set_mask(a, mask_a) or not _try_set_mask(b, mask_b):
            _skip("cannot attach _keras_mask to input tensors in this environment/backend")

        add_op = layers.Add()

        # Call merge layer with list inputs (the problematic path)
        x = add_op([a, b])

        # Ground-truth expected mask computed explicitly
        try:
            expected_mask = add_op.compute_mask([a, b], [mask_a, mask_b])
        except TypeError:
            # Some versions use compute_mask(inputs, mask=None)
            try:
                expected_mask = add_op.compute_mask([a, b], mask=[mask_a, mask_b])
            except Exception as e:
                _skip(f"compute_mask failed: {e}")
        except Exception as e:
            _skip(f"compute_mask failed: {e}")

        if expected_mask is None:
            _skip("compute_mask returned None; cannot evaluate mask propagation for this backend/version")

        out_mask = _try_get_mask(x)

        # -------------------------
        # Oracle: output mismatch / missing mask
        # Bug reproduces if output mask is missing/None OR differs from expected_mask.
        # -------------------------
        if out_mask is None:
            _pass()

        out_np = _to_numpy(out_mask)
        exp_np = _to_numpy(expected_mask)

        if out_np is None or exp_np is None:
            _skip("unable to convert masks to numpy for comparison in this environment")

        try:
            out_np = np.asarray(out_np).astype(bool)
            exp_np = np.asarray(exp_np).astype(bool)
        except Exception as e:
            _skip(f"mask normalization failed: {e}")

        if out_np.shape != exp_np.shape or not np.array_equal(out_np, exp_np):
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
# conda activate keras_venv
# export KERAS_BACKEND=tensorflow

# python -c "import keras, os; print('KERAS', keras.__version__); print('KERAS_BACKEND', os.environ.get('KERAS_BACKEND')); import tensorflow as tf; print('TF', tf.__version__)"

# python -c "import tensorflow as tf; print([d.name for d in tf.config.list_physical_devices('GPU')])"

# KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python testcases/keras_testcase.py

# export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
# KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python testcases/keras_testcase.py

# export XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/cuda"
# KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python testcases/keras_testcase.py


# Output:
# *****************
# 2026-02-17 19:11:44.767786: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# KERAS_BACKEND: tensorflow
# KERAS_VERSION: 3.12.1
# TF_VERSION: 2.20.0
# TF_GPUS: ['/physical_device:GPU:0']
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1771326707.215260 1776426 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22446 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:02:00.0, compute capability: 8.6
# 2026-02-17 19:11:47.887239: W external/local_xla/xla/service/gpu/llvm_gpu_backend/default/nvptx_libdevice_path.cc:41] Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice. This may result in compilation or runtime failures, if the program we try to run uses routines from libdevice.
# Searched for CUDA in the following directories:
#   ./cuda_sdk_lib
#   testcases/keras_testcase.py.runfiles/cuda_nvcc
#   testcases/keras_testcase.py.runfiles/cuda_nvdisasm
#   testcases/keras_testcase.py.runfiles/nvidia_nvshmem
#   testcas/cuda_nvcc
#   testcas/cuda_nvdisasm
#   testcas/nvidia_nvshmem
  
#   /usr/local/cuda
#   /opt/cuda
#   /home/talha/miniconda3/envs/keras_venv/lib/python3.10/site-packages/tensorflow/python/platform/../../../nvidia/cuda_nvcc
#   /home/talha/miniconda3/envs/keras_venv/lib/python3.10/site-packages/tensorflow/python/platform/../../../../nvidia/cuda_nvcc
#   /home/talha/miniconda3/envs/keras_venv/lib/python3.10/site-packages/tensorflow/python/platform/../../cuda
#   /home/talha/miniconda3/envs/keras_venv/lib/python3.10/site-packages/tensorflow/python/platform/../../../../../..
#   /home/talha/miniconda3/envs/keras_venv/lib/python3.10/site-packages/tensorflow/python/platform/../../../../../../..
#   .
# You can choose the search directory by setting xla_gpu_cuda_data_dir in HloModule's DebugOptions.  For most apps, setting the environment variable XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda will work.
# Test Failed ❌

# GCFL-OTHER-0050

import os
import sys
import random
import traceback
import glob
import shutil
import subprocess


def _early_reexec():
    """
    Two modes:
      - TRY_RET_CALL=0 (default): keep it CPU-only and QUIET.
      - TRY_RET_CALL=1: allow GPU + XLA and apply libdevice workaround.
    """
    try_ret_call = os.environ.get("TRY_RET_CALL", "0") == "1"

    # avoid infinite recursion
    if os.environ.get("_GCFL_REEXECED", "0") == "1":
        return

    if not try_ret_call:
        # CPU-only + quieter TF logs (your TF_CPP_MIN_LOG_LEVEL=0 forces spam)
        new_env = os.environ.copy()
        new_env["_GCFL_REEXECED"] = "1"

        # Hide GPUs from CUDA
        new_env["CUDA_VISIBLE_DEVICES"] = ""

        # Force TF to be quiet unless user insists; if they set it to 0, we override
        # because the entire point of this mode is "don't spam logs".
        new_env["TF_CPP_MIN_LOG_LEVEL"] = "3"

        # Run again
        cmd = [sys.executable] + sys.argv
        proc = subprocess.run(cmd, env=new_env)
        raise SystemExit(proc.returncode)

    # TRY_RET_CALL=1: no re-exec; we want user's env as-is (GPU allowed)
    return


_early_reexec()


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


def _set_seeds(seed: int = 2021):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _matches_expected_valueerror(e: BaseException) -> bool:
    return isinstance(e, ValueError) and ("undefined shapes are not supported" in str(e).lower())


def _is_env_issue(e: BaseException) -> bool:
    msg = str(e).lower()
    if isinstance(e, (ModuleNotFoundError, ImportError)):
        return True
    needles = [
        "cuda", "cudnn", "cublas", "dlopen",
        "libdevice", "jit compilation failed", "xla", "generating device code failed",
        "tensorflow is not installed", "requires tensorflow",
    ]
    return any(n in msg for n in needles)


def _find_libdevice_bc() -> str | None:
    candidates = []
    for root in ["/usr/local/cuda", "/opt/cuda"] + glob.glob("/usr/local/cuda-*"):
        candidates.extend(glob.glob(os.path.join(root, "nvvm", "libdevice", "libdevice*.bc")))

    sp = os.path.join(sys.prefix, "lib", "python3.10", "site-packages")
    candidates.extend(glob.glob(os.path.join(sp, "**", "libdevice*.bc"), recursive=True))

    uniq = []
    seen = set()
    for p in candidates:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isfile(ap):
            uniq.append(ap)
            seen.add(ap)

    if not uniq:
        return None

    for p in uniq:
        if os.path.basename(p) == "libdevice.10.bc":
            return p
    return uniq[0]


def _force_xla_cuda_data_dir_using_libdevice(libdevice_path: str) -> str:
    base = "/tmp/gcfl_xla_cuda_data"
    target_dir = os.path.join(base, "nvvm", "libdevice")
    os.makedirs(target_dir, exist_ok=True)

    target_file = os.path.join(target_dir, "libdevice.10.bc")
    if not os.path.isfile(target_file):
        try:
            os.symlink(libdevice_path, target_file)
            print(f"LIBDEVICE_SYMLINKED: {libdevice_path} -> {target_file}")
        except Exception:
            shutil.copyfile(libdevice_path, target_file)
            print(f"LIBDEVICE_COPIED: {libdevice_path} -> {target_file}")
    else:
        print(f"LIBDEVICE_EXISTS: {target_file}")

    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={base}"
    print(f"XLA_FLAGS_FORCED: {os.environ['XLA_FLAGS']}")
    return base


def _try_tf_gpu_safety():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass


def _build_models(keras):
    ops = keras.ops

    def euclidean_distance(vects):
        x, y = vects
        sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
        eps = getattr(keras.backend, "epsilon", lambda: 1e-7)()
        return ops.sqrt(ops.maximum(sum_square, eps))

    try:
        inputs = keras.layers.Input(shape=(128, 128, 3))
    except TypeError:
        inputs = keras.layers.Input((128, 128, 3))

    preprocess_fn = keras.applications.efficientnet.preprocess_input
    try:
        x_in = keras.layers.Lambda(lambda z: preprocess_fn(z), name="effnet_preprocess")(inputs)
    except Exception:
        x_in = inputs

    base = keras.applications.EfficientNetB0(
        include_top=False,
        input_tensor=x_in,
        pooling="max",
        weights=None,
    )

    head = base.output
    x = keras.layers.Dense(256, activation="relu")(head)
    x = keras.layers.Dense(32)(x)
    embedding_network = keras.Model(inputs, x)

    try:
        input_1 = keras.layers.Input(shape=(128, 128, 3), name="input_layer_base_r")
        input_2 = keras.layers.Input(shape=(128, 128, 3), name="input_layer_base_l")
    except TypeError:
        input_1 = keras.layers.Input((128, 128, 3), name="input_layer_base_r")
        input_2 = keras.layers.Input((128, 128, 3), name="input_layer_base_l")

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = keras.layers.Lambda(euclidean_distance, output_shape=(1,), name="euclid")([tower_1, tower_2])
    output_layer = keras.layers.Dense(1, activation="sigmoid")(merge_layer)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

    class SiameseWrapper(keras.Model):
        def __init__(self, inner_model):
            super().__init__()
            try:
                self.attr_input_1 = keras.layers.Input(shape=(128, 128, 3), name="attr_input_layer_r")
                self.attr_input_2 = keras.layers.Input(shape=(128, 128, 3), name="attr_input_layer_l")
            except TypeError:
                self.attr_input_1 = keras.layers.Input((128, 128, 3), name="attr_input_layer_r")
                self.attr_input_2 = keras.layers.Input((128, 128, 3), name="attr_input_layer_l")
            self.inner = inner_model

        def call(self, inputs, training=None):
            return self.inner

    return siamese, SiameseWrapper(siamese)


def main():
    try:
        os.environ.setdefault("KERAS_BACKEND", os.environ.get("KERAS_BACKEND", "tensorflow"))

        _set_seeds(2021)
        try_ret_call = os.environ.get("TRY_RET_CALL", "0") == "1"

        if try_ret_call:
            libdevice = _find_libdevice_bc()
            if not libdevice:
                _skip("TRY_RET_CALL=1 requested but no libdevice*.bc found.")
            print(f"LIBDEVICE_FOUND: {libdevice}")
            _force_xla_cuda_data_dir_using_libdevice(libdevice)

        import numpy as np
        import keras

        if try_ret_call:
            _try_tf_gpu_safety()

        try:
            if hasattr(keras, "utils") and hasattr(keras.utils, "set_random_seed"):
                keras.utils.set_random_seed(2021)
        except Exception:
            pass

        try:
            backend_name = None
            try:
                backend_name = keras.backend.backend()
            except Exception:
                backend_name = os.environ.get("KERAS_BACKEND", "unknown")
            print(f"KERAS_VERSION: {getattr(keras, '__version__', 'unknown')}")
            print(f"KERAS_BACKEND_RESOLVED: {backend_name}")
        except Exception:
            pass

        siamese, wrapper = _build_models(keras)

        try:
            opt = keras.optimizers.Adam()
            loss = keras.losses.BinaryCrossentropy()
            siamese.compile(optimizer=opt, loss=loss)
            wrapper.compile(optimizer=opt, loss=loss)
        except Exception:
            pass

        try:
            siamese.summary()
        except Exception as e:
            if _matches_expected_valueerror(e):
                _pass()
            if _is_env_issue(e):
                _skip(f"env issue during siamese.summary(): {type(e).__name__}: {e}")

        x1 = np.zeros((1, 128, 128, 3), dtype="float32")
        x2 = np.zeros((1, 128, 128, 3), dtype="float32")

        ret = wrapper([x1, x2])

        print("WRAPPER_CALL_RETURN_TYPE:", type(ret))
        print("WRAPPER_CALL_RET_IS_MODEL:", hasattr(ret, "layers") and hasattr(ret, "summary"))

        try:
            if hasattr(ret, "summary"):
                ret.summary()
        except Exception as e:
            if _matches_expected_valueerror(e):
                _pass()
            if _is_env_issue(e):
                _skip(f"env issue during ret.summary(): {type(e).__name__}: {e}")

        if try_ret_call:
            try:
                if callable(ret):
                    y = ret([x1, x2])
                    print(f"RET_EXEC_OK: output_shape={getattr(y, 'shape', None)}")
            except Exception as e:
                if _matches_expected_valueerror(e):
                    _pass()
                if _is_env_issue(e):
                    _skip(f"env issue during ret([x1,x2]) execution: {type(e).__name__}: {e}")
                print("RET_CALL_ERROR:", type(e).__name__, str(e))

        else:
            print("RET_CALL_SKIPPED: set TRY_RET_CALL=1 to enable execution path")

        try:
            wrapper.summary()
        except Exception as e:
            if _matches_expected_valueerror(e):
                _pass()
            if _is_env_issue(e):
                _skip(f"env issue during wrapper.summary(): {type(e).__name__}: {e}")

        print("NOTE: Oracle not observed (no ValueError: 'Undefined shapes are not supported').")
        _fail()

    except SystemExit:
        raise
    except BaseException as e:
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
# unset TRY_RET_CALL
# unset CUDA_VISIBLE_DEVICES
# python testcases/keras_testcase.py


# conda activate keras_venv
# export KERAS_BACKEND=tensorflow
# export TRY_RET_CALL=1
# unset CUDA_VISIBLE_DEVICES
# python testcases/keras_testcase.py



# Output:
# *****************

# (keras_venv) talha@bitse-SYS-7048GR-TR:~/dl_testing$ conda activate keras_venv
# export KERAS_BACKEND=tensorflow
# unset TRY_RET_CALL
# unset CUDA_VISIBLE_DEVICES
# python testcases/keras_testcase.py
# KERAS_VERSION: 3.12.1
# KERAS_BACKEND_RESOLVED: tensorflow
# Model: "functional_1"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ input_layer_base_r            │ (None, 128, 128, 3)       │               0 │ -                          │
# │ (InputLayer)                  │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ input_layer_base_l            │ (None, 128, 128, 3)       │               0 │ -                          │
# │ (InputLayer)                  │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ functional (Functional)       │ (None, 32)                │       4,385,731 │ input_layer_base_r[0][0],  │
# │                               │                           │                 │ input_layer_base_l[0][0]   │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ euclid (Lambda)               │ (None, 1)                 │               0 │ functional[0][0],          │
# │                               │                           │                 │ functional[1][0]           │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dense_2 (Dense)               │ (None, 1)                 │               2 │ euclid[0][0]               │
# └───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
#  Total params: 4,385,733 (16.73 MB)
#  Trainable params: 4,343,710 (16.57 MB)
#  Non-trainable params: 42,023 (164.16 KB)
# WRAPPER_CALL_RETURN_TYPE: <class 'keras.src.models.functional.Functional'>
# WRAPPER_CALL_RET_IS_MODEL: True
# Model: "functional_1"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ input_layer_base_r            │ (None, 128, 128, 3)       │               0 │ -                          │
# │ (InputLayer)                  │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ input_layer_base_l            │ (None, 128, 128, 3)       │               0 │ -                          │
# │ (InputLayer)                  │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ functional (Functional)       │ (None, 32)                │       4,385,731 │ input_layer_base_r[0][0],  │
# │                               │                           │                 │ input_layer_base_l[0][0]   │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ euclid (Lambda)               │ (None, 1)                 │               0 │ functional[0][0],          │
# │                               │                           │                 │ functional[1][0]           │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dense_2 (Dense)               │ (None, 1)                 │               2 │ euclid[0][0]               │
# └───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
#  Total params: 4,385,733 (16.73 MB)
#  Trainable params: 4,343,710 (16.57 MB)
#  Non-trainable params: 42,023 (164.16 KB)
# RET_CALL_SKIPPED: set TRY_RET_CALL=1 to enable execution path
# Model: "siamese_wrapper"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ functional_1 (Functional)            │ (None, 1)                   │       4,385,733 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 4,385,733 (16.73 MB)
#  Trainable params: 4,343,710 (16.57 MB)
#  Non-trainable params: 42,023 (164.16 KB)
# NOTE: Oracle not observed (no ValueError: 'Undefined shapes are not supported').
# Test Failed ❌
# (keras_venv) talha@bitse-SYS-7048GR-TR:~/dl_testing$ conda activate keras_venv
# export KERAS_BACKEND=tensorflow
# export TRY_RET_CALL=1
# unset CUDA_VISIBLE_DEVICES     # IMPORTANT
# python testcases/keras_testcase.py
# LIBDEVICE_FOUND: /home/talha/miniconda3/envs/keras_venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/libdevice.10.bc
# LIBDEVICE_EXISTS: /tmp/gcfl_xla_cuda_data/nvvm/libdevice/libdevice.10.bc
# XLA_FLAGS_FORCED: --xla_gpu_cuda_data_dir=/tmp/gcfl_xla_cuda_data
# 2026-02-19 21:57:25.884405: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# KERAS_VERSION: 3.12.1
# KERAS_BACKEND_RESOLVED: tensorflow
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1771509448.890125 3968434 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22446 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:02:00.0, compute capability: 8.6
# I0000 00:00:1771509448.890971 3968434 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 22446 MB memory:  -> device: 1, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:03:00.0, compute capability: 8.6
# I0000 00:00:1771509448.891692 3968434 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:2 with 22446 MB memory:  -> device: 2, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:82:00.0, compute capability: 8.6
# I0000 00:00:1771509448.892330 3968434 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:3 with 22446 MB memory:  -> device: 3, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:83:00.0, compute capability: 8.6
# Model: "functional_1"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ input_layer_base_r            │ (None, 128, 128, 3)       │               0 │ -                          │
# │ (InputLayer)                  │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ input_layer_base_l            │ (None, 128, 128, 3)       │               0 │ -                          │
# │ (InputLayer)                  │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ functional (Functional)       │ (None, 32)                │       4,385,731 │ input_layer_base_r[0][0],  │
# │                               │                           │                 │ input_layer_base_l[0][0]   │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ euclid (Lambda)               │ (None, 1)                 │               0 │ functional[0][0],          │
# │                               │                           │                 │ functional[1][0]           │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dense_2 (Dense)               │ (None, 1)                 │               2 │ euclid[0][0]               │
# └───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
#  Total params: 4,385,733 (16.73 MB)
#  Trainable params: 4,343,710 (16.57 MB)
#  Non-trainable params: 42,023 (164.16 KB)
# WRAPPER_CALL_RETURN_TYPE: <class 'keras.src.models.functional.Functional'>
# WRAPPER_CALL_RET_IS_MODEL: True
# Model: "functional_1"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ input_layer_base_r            │ (None, 128, 128, 3)       │               0 │ -                          │
# │ (InputLayer)                  │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ input_layer_base_l            │ (None, 128, 128, 3)       │               0 │ -                          │
# │ (InputLayer)                  │                           │                 │                            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ functional (Functional)       │ (None, 32)                │       4,385,731 │ input_layer_base_r[0][0],  │
# │                               │                           │                 │ input_layer_base_l[0][0]   │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ euclid (Lambda)               │ (None, 1)                 │               0 │ functional[0][0],          │
# │                               │                           │                 │ functional[1][0]           │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dense_2 (Dense)               │ (None, 1)                 │               2 │ euclid[0][0]               │
# └───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
#  Total params: 4,385,733 (16.73 MB)
#  Trainable params: 4,343,710 (16.57 MB)
#  Non-trainable params: 42,023 (164.16 KB)
# 2026-02-19 21:57:31.629122: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:473] Loaded cuDNN version 91002
# RET_EXEC_OK: output_shape=(1, 1)
# Model: "siamese_wrapper"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ functional_1 (Functional)            │ (None, 1)                   │       4,385,733 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 4,385,733 (16.73 MB)
#  Trainable params: 4,343,710 (16.57 MB)
#  Non-trainable params: 42,023 (164.16 KB)
# NOTE: Oracle not observed (no ValueError: 'Undefined shapes are not supported').
# Test Failed ❌
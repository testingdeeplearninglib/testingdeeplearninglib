# GCFL-AUTOGRADBA-0102

# GCFL-AUTOGRADBA-0102 (MODERN oracle for TF 2.x)
# Pass (Test Passed ✅) == suspicious behavior observed.

import os
import sys
import traceback


def _skip(reason: str):
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass(reason: str):
    print(f"SUSPICIOUS: {reason}")
    print("Test Passed ✅")
    sys.exit(0)


def _fail():
    print("Test Failed ❌")
    sys.exit(0)


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, default)).strip())
    except Exception:
        return default


def main():
    try:
        import numpy as np
    except Exception as e:
        _skip(f"numpy not available: {e}")

    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"tensorflow not available: {e}")

    print(f"TF_VERSION: {getattr(tf, '__version__', 'unknown')}")

    seed = 2021
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    gpus = []
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:
        pass
    print(f"TF_GPUS: {gpus}")

    if not gpus:
        _skip("No GPU visible; this oracle needs GPU vs CPU comparison")

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"NOTE: set_memory_growth failed: {e}")

    # Use a single GPU by default to reduce noise
    gpu_index = _env_int("GCFL_GPU_INDEX", 0)
    gpu_device = f"/GPU:{gpu_index}"

    N = _env_int("GCFL_NUM_SCALARS", 16384)
    ITERS = _env_int("GCFL_ITERS", 50)
    ATOL = _env_float("GCFL_ATOL", 1e-5)
    RTOL = _env_float("GCFL_RTOL", 1e-4)
    USE_TF_FUNCTION = _env_int("GCFL_TF_FUNCTION", 1)
    USE_XLA = _env_int("GCFL_XLA", 0)

    print(f"GCFL_GPU_INDEX: {gpu_index}")
    print(f"GCFL_NUM_SCALARS: {N}")
    print(f"GCFL_ITERS: {ITERS}")
    print(f"GCFL_ATOL: {ATOL}")
    print(f"GCFL_RTOL: {RTOL}")
    print(f"GCFL_TF_FUNCTION: {USE_TF_FUNCTION}")
    print(f"GCFL_XLA: {USE_XLA}")

    class MyModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.d1 = tf.keras.layers.Dense(256, activation="relu")
            self.d2 = tf.keras.layers.Dense(1)

        def call(self, inputs):
            return self.d2(self.d1(inputs))

    cpu_model = MyModel()
    gpu_model = MyModel()

    x = tf.ones([1, 256], dtype=tf.float32)

    with tf.device("/CPU:0"):
        _ = cpu_model(x)
    with tf.device(gpu_device):
        _ = gpu_model(x)

    # sync weights
    try:
        gpu_model.set_weights(cpu_model.get_weights())
    except Exception as e:
        _skip(f"failed to sync weights CPU->GPU: {e}")

    def compute_grads(model, device_str):
        with tf.device(device_str):
            with tf.GradientTape() as tape:
                y = model(x)
                s = y[0, 0]  # rank-0 scalar from indexing inside tape (feature)
                one = tf.reshape(s, [1])  # make concat legal
                loss_vec = tf.concat([one] * N, axis=0)
                loss = tf.reduce_sum(loss_vec)
            grads = tape.gradient(loss, model.trainable_variables)
        return loss, grads

    if USE_TF_FUNCTION:
        compute_grads = tf.function(compute_grads, jit_compile=bool(USE_XLA))

    for it in range(ITERS):
        # GPU
        try:
            _, ggrads = compute_grads(gpu_model, gpu_device)
        except Exception as e:
            _pass(f"GPU exception at iter {it}: {type(e).__name__}: {e}")

        # CPU
        try:
            _, cgrads = compute_grads(cpu_model, "/CPU:0")
        except Exception as e:
            msg = str(e)
            # Known XLA limitation: variable on different device
            if ("Trying to access resource" in msg) and (
                "xla/known_issues#tfvariable_on_a_different_device" in msg
            ):
                _fail()
            _pass(f"CPU exception at iter {it}: {type(e).__name__}: {e}")

        # None gradients
        if ggrads is None or any(g is None for g in ggrads):
            _pass(f"GPU returned None gradient at iter {it}")
        if cgrads is None or any(g is None for g in cgrads):
            _pass(f"CPU returned None gradient at iter {it}")

        # NaN/Inf
        try:
            for i, g in enumerate(ggrads):
                a = g.numpy()
                if not np.all(np.isfinite(a)):
                    _pass(f"GPU grad[{i}] NaN/Inf at iter {it}")
            for i, g in enumerate(cgrads):
                a = g.numpy()
                if not np.all(np.isfinite(a)):
                    _pass(f"CPU grad[{i}] NaN/Inf at iter {it}")
        except Exception as e:
            _pass(f"grad materialization failed at iter {it}: {type(e).__name__}: {e}")

        # Compare
        try:
            for i, (cg, gg) in enumerate(zip(cgrads, ggrads)):
                ca = cg.numpy()
                ga = gg.numpy()
                diff = np.max(np.abs(ca - ga))
                denom = np.maximum(np.max(np.abs(ca)), np.max(np.abs(ga)))
                rel = diff / (denom + 1e-12)
                if diff > ATOL and rel > RTOL:
                    _pass(
                        f"CPU↔GPU grad mismatch at iter {it}, var {i}: abs={diff:.6g}, rel={rel:.6g}"
                    )
        except Exception as e:
            _pass(f"CPU↔GPU compare failed at iter {it}: {type(e).__name__}: {e}")

    _fail()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    
    
# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# source ~/.venvs/dl_testing/bin/activate
# unset CUDA_VISIBLE_DEVICES
# unset NVIDIA_VISIBLE_DEVICES

# export GCFL_GPU_INDEX=0
# export GCFL_NUM_SCALARS=16384
# export GCFL_ITERS=50
# export GCFL_ATOL=1e-6
# export GCFL_RTOL=1e-5
# export GCFL_TF_FUNCTION=1
# export GCFL_XLA=1

# python gcfl_autogradba_0102_modern.py



# Output:
# *****************
# 2026-01-21 16:29:57.997192: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# TF_VERSION: 2.20.0
# TF_GPUS: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU')]
# GCFL_GPU_INDEX: 0
# GCFL_NUM_SCALARS: 16384
# GCFL_ITERS: 50
# GCFL_ATOL: 1e-06
# GCFL_RTOL: 1e-05
# GCFL_TF_FUNCTION: 1
# GCFL_XLA: 1
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1768984200.696954 1212002 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22446 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:02:00.0, compute capability: 8.6
# I0000 00:00:1768984200.697781 1212002 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 22446 MB memory:  -> device: 1, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:03:00.0, compute capability: 8.6
# I0000 00:00:1768984200.698426 1212002 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:2 with 22446 MB memory:  -> device: 2, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:82:00.0, compute capability: 8.6
# I0000 00:00:1768984200.699091 1212002 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:3 with 22446 MB memory:  -> device: 3, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:83:00.0, compute capability: 8.6
# 2026-01-21 16:30:10.278033: I external/local_xla/xla/service/service.cc:163] XLA service 0x121a70b0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
# 2026-01-21 16:30:10.278114: I external/local_xla/xla/service/service.cc:171]   StreamExecutor device (0): NVIDIA GeForce RTX 3090, Compute Capability 8.6
# 2026-01-21 16:30:10.278125: I external/local_xla/xla/service/service.cc:171]   StreamExecutor device (1): NVIDIA GeForce RTX 3090, Compute Capability 8.6
# 2026-01-21 16:30:10.278150: I external/local_xla/xla/service/service.cc:171]   StreamExecutor device (2): NVIDIA GeForce RTX 3090, Compute Capability 8.6
# 2026-01-21 16:30:10.278173: I external/local_xla/xla/service/service.cc:171]   StreamExecutor device (3): NVIDIA GeForce RTX 3090, Compute Capability 8.6
# 2026-01-21 16:30:11.111502: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
# 2026-01-21 16:30:11.145359: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:473] Loaded cuDNN version 91800
# I0000 00:00:1768984211.341893 1212002 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
# 2026-01-21 16:30:20.066983: W tensorflow/core/framework/op_kernel.cc:1855] OP_REQUIRES failed at xla_ops.cc:528 : INVALID_ARGUMENT: Trying to access resource my_model/kernel/0 (defined @ /home/talha/.venvs/dl_testing/lib/python3.12/site-packages/keras/src/backend/tensorflow/core.py:42) located in device /job:localhost/replica:0/task:0/device:CPU:0 from device /job:localhost/replica:0/task:0/device:GPU:0
#  Cf. https://www.tensorflow.org/xla/known_issues#tfvariable_on_a_different_device
# Test Failed ❌

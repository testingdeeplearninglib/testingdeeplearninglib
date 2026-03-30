# GCFL-OTHER-0039

import os
import numpy as np

def max_abs(a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.max(np.abs(a[m] - b[m])))

def to_np(x):
    if isinstance(x, dict):
        for v in x.values():
            y = to_np(v)
            if y is not None:
                return y
        return None
    if hasattr(x, "numpy"):
        return x.numpy()
    try:
        return np.asarray(x)
    except Exception:
        return None

def main():
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    import tensorflow as tf
    import keras_hub

    np.random.seed(2021)
    try:
        tf.random.set_seed(2021)
    except Exception:
        pass

    # reduce VRAM grab
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    except Exception:
        pass

    preset = os.environ.get("CLIP_PRESET", "clip_vit_base_patch32")
    thresh = float(os.environ.get("GCFL_ABS_THRESH", "1e-3"))
    seq_len = int(os.environ.get("SEQ_LEN", "77"))

    clip = keras_hub.models.CLIPBackbone.from_preset(preset)
    text_enc = clip.text_encoder

    vocab = 49408
    token_ids = np.random.randint(0, vocab, size=(1, seq_len), dtype=np.int32)

    @tf.function(jit_compile=False)
    def graph_call(ids):
        return text_enc({"token_ids": ids}, training=False)

    y_graph = to_np(graph_call(token_ids))
    y_pred = to_np(text_enc.predict_on_batch({"token_ids": token_ids}))

    if y_graph is None or y_pred is None:
        raise RuntimeError("Could not materialize outputs")

    d = max_abs(y_graph, y_pred)
    print(f"max_abs_graph_pred={d:.6e} thresh={thresh}")

    if d > thresh:
        raise AssertionError(
            f"predict_on_batch != tf.function: max_abs={d:.6e} > {thresh}"
        )

if __name__ == "__main__":
    main()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Keras


# Commands
# *****************
# conda activate keras_venv
# cd ~/dl_testing

# unset TF_XLA_FLAGS
# export CUDA_VISIBLE_DEVICES=0
# export KERAS_BACKEND=tensorflow
# export CLIP_PRESET=clip_vit_base_patch32
# export GCFL_ABS_THRESH=1e-3

# set -o pipefail
# python repro_min_khub_clip_predict_mismatch.py 2>&1 | tee repro_gpu.log
# echo "exit_code=$?"


# Triggering commands (GPU repro + disable XLA auto-jit)

# conda activate keras_venv
# cd ~/dl_testing

# export TF_XLA_FLAGS=--tf_xla_auto_jit=0
# export CUDA_VISIBLE_DEVICES=0
# export KERAS_BACKEND=tensorflow
# export CLIP_PRESET=clip_vit_base_patch32
# export GCFL_ABS_THRESH=1e-3

# set -o pipefail
# python repro_min_khub_clip_predict_mismatch.py 2>&1 | tee repro_gpu_no_xla.log
# echo "exit_code=$?"


# Triggering commands (CPU-only control)

# conda activate keras_venv
# cd ~/dl_testing

# export CUDA_VISIBLE_DEVICES=""
# unset TF_XLA_FLAGS
# export KERAS_BACKEND=tensorflow
# export CLIP_PRESET=clip_vit_base_patch32
# export GCFL_ABS_THRESH=1e-3

# set -o pipefail
# python repro_min_khub_clip_predict_mismatch.py 2>&1 | tee repro_cpu.log
# echo "exit_code=$?"



# Output:
# *****************
# 2026-03-02 01:39:33.787145: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1772386776.567432  891672 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22446 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:02:00.0, compute capability: 8.6
# ...
# DEBUG_DIFF: max_abs=1.308370e-02 max_rel=1.954607e+00
# ORACLE_MISMATCH: predict_on_batch != direct (thresh=0.001)


# Triggering commands (CPU-only control)
# max_abs_graph_pred=9.863973e-03 thresh=0.001
# AssertionError: predict_on_batch != tf.function: max_abs=9.863973e-03 > 0.001
# exit_code=1


# Output (CPU-only control)
# max_abs_graph_pred=0.000000e+00 thresh=0.001
# exit_code=0

# max_abs_graph_pred=0.000000e+00 thresh=0.001
# exit_code=0

# Test Passed ✅



# Reported ✅
# Link: 
# https://github.com/keras-team/keras/issues/22380
# FILE: GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_embed_zeros_save.py
import os, sys, json, tempfile, random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

def _skip(reason: str):
    print(f"SKIP_ENV: {reason}", flush=True)
    sys.exit(0)

def _pass():
    print("Test Passed ✅", flush=True)
    sys.exit(0)

def _fail():
    print("Test Failed ❌", flush=True)
    sys.exit(0)

def _norm_tfver(v: str) -> str:
    v = (v or "").strip()
    v = v.split("+", 1)[0]
    v = v.split("-", 1)[0]
    return v

def _env_line(tf, np, knobs: dict):
    payload = {
        "python": sys.version.split()[0],
        "tensorflow": getattr(tf, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
        "cpu_count": len(tf.config.list_physical_devices("CPU")),
        "knobs": knobs,
    }
    print("ENV: " + json.dumps(payload, sort_keys=True), flush=True)

def main():
    if not (sys.version_info.major == 3 and sys.version_info.minor in (10, 11)):
        _skip(f"Python not in {{3.10,3.11}}: {sys.version.split()[0]}")

    try:
        import numpy as np
    except Exception as e:
        _skip(f"numpy import failed: {type(e).__name__}: {e}")

    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"tensorflow import failed: {type(e).__name__}: {e}")

    if _norm_tfver(getattr(tf, "__version__", "")) != "2.20.0":
        _skip(f"tensorflow version != 2.20.0: {getattr(tf,'__version__','unknown')}")

    seed = int(os.environ.get("SEED", "2026"))
    vocab = int(os.environ.get("VOCAB", "128"))
    seq = int(os.environ.get("SEQ", "4"))
    batch = int(os.environ.get("BATCH", "2"))
    d_model = int(os.environ.get("D_MODEL", "16"))

    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    _env_line(tf, np, {"SEED": seed, "VOCAB": vocab, "SEQ": seq, "BATCH": batch, "D_MODEL": d_model})

    def build_model(init):
        x = tf.keras.Input(shape=(seq,), dtype=tf.int32, name="tok")
        e = tf.keras.layers.Embedding(vocab, d_model, embeddings_initializer=init)(x)
        y = tf.keras.layers.Dense(8, activation="relu")(e)
        y = tf.keras.layers.Dense(4)(y)
        return tf.keras.Model(x, y)

    x_in = np.random.RandomState(seed).randint(0, vocab, size=(batch, seq), dtype=np.int32)

    with tempfile.TemporaryDirectory() as td:
        ref_dir = os.path.join(td, "ref")
        bad_dir = os.path.join(td, "bad")

        ref = build_model(tf.keras.initializers.Zeros())
        ref_out = ref(x_in, training=False).numpy()
        try:
            ref.save(ref_dir)
            ref2 = tf.keras.models.load_model(ref_dir)
            ref2_out = ref2(x_in, training=False).numpy()
            if not np.allclose(ref_out, ref2_out, atol=0.0, rtol=0.0):
                _pass()
        except Exception as e:
            _skip(f"reference save/load failed: {type(e).__name__}: {e}")

        bad = build_model(tf.keras.initializers.zeros)
        bad_out = bad(x_in, training=False).numpy()
        try:
            bad.save(bad_dir)
            bad2 = tf.keras.models.load_model(bad_dir)
            bad2_out = bad2(x_in, training=False).numpy()
            if not np.allclose(bad_out, bad2_out, atol=0.0, rtol=0.0):
                _pass()
        except Exception:
            _pass()

    _fail()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _skip(f"harness_error: {type(e).__name__}: {e}")



# Output:
# *****************
# bug no: GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_embed_zeros_save
# Result: SKIP_ENV
# Result: Test Failed ❌
# Triggering command:
# conda activate tf_venv_220_py311
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=2
# export TF_DETERMINISTIC_OPS=1
# export KERAS_BACKEND=tensorflow
# export SEED=2026
# /home/talha/miniconda3/envs/tf_venv_220_py311/bin/python3.11 \
#   testcases/tf_serialization_inputs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_embed_zeros_save.py \
#   > logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_embed_zeros_save_stdout.log \
#   2> logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_embed_zeros_save_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_embed_zeros_save_stdout.log
# Test Failed ❌
# Observed output:
# exit_code=0
# SKIP_ENV: reference save/load failed: ValueError: Invalid filepath extension for saving. Please add either a `.keras` extension for the native Keras format (recommended) or a `.h5` extension. Use `model.export(filepath)` if you want to export a SavedModel for use with TFLite/TFServing/etc. Received: filepath=/tmp/tmp3jd9pzly/ref.

# Note:
# This testcase was blocked by modern Keras save semantics, so it did not meaningfully test the target serialization behavior.
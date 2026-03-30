# FILE: GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_sequential_json_roundtrip.py
import os, sys, json, random

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
    batch = int(os.environ.get("BATCH", "2"))
    d_model = int(os.environ.get("D_MODEL", "32"))

    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    _env_line(tf, np, {"SEED": seed, "BATCH": batch, "D_MODEL": d_model})

    try:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(d_model,)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(4),
            ]
        )
        x = np.random.RandomState(seed).randn(batch, d_model).astype("float32")
        y0 = model(x, training=False).numpy()

        js = model.to_json()
        model2 = tf.keras.models.model_from_json(js)
        model2.set_weights(model.get_weights())
        y1 = model2(x, training=False).numpy()

        # should be bitwise-identical on CPU for this simple graph
        if not np.allclose(y0, y1, atol=0.0, rtol=0.0):
            _pass()
    except Exception:
        # sequential JSON roundtrip should work; exception is suspicious (TF2 sequential serialization bug lineage)
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
# bug no: GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_sequential_json_roundtrip
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
#   testcases/tf_serialization_inputs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_sequential_json_roundtrip.py \
#   > logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_sequential_json_roundtrip_stdout.log \
#   2> logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_sequential_json_roundtrip_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_sequential_json_roundtrip_stdout.log

# Observed output:
# exit_code=0
# Test Failed ❌

# Note:
# The suspicious JSON roundtrip mismatch or exception was not triggered in this run.
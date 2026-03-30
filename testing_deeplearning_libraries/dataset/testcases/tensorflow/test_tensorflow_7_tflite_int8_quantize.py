# FILE: GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_tflite_int8_quantize.py
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
    batch = int(os.environ.get("BATCH", "1"))
    d_model = int(os.environ.get("D_MODEL", "8"))
    iters = int(os.environ.get("ITERS", "8"))

    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    _env_line(tf, np, {"SEED": seed, "BATCH": batch, "D_MODEL": d_model, "ITERS": iters})

    # Tiny model: should be trivially quantizable
    inp = tf.keras.Input(shape=(d_model,), name="x")
    x = tf.keras.layers.Dense(8, activation="relu")(inp)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)

    # representative dataset generator
    def rep_data():
        for i in range(iters):
            x0 = np.random.RandomState(seed + i).randn(batch, d_model).astype("float32")
            yield [x0]

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
    except Exception:
        # simple model int8 quantization should not crash; if it does, suspicious
        _pass()

    # Optional quick interpreter smoke test
    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        in0 = interpreter.get_input_details()[0]
        out0 = interpreter.get_output_details()[0]

        x_fp = np.random.RandomState(seed).randn(batch, d_model).astype("float32")
        scale, zero_point = in0["quantization"]
        if scale == 0:
            _pass()
        x_q = np.clip(np.round(x_fp / scale + zero_point), -128, 127).astype("int8")
        interpreter.set_tensor(in0["index"], x_q)
        interpreter.invoke()
        y_q = interpreter.get_tensor(out0["index"])
        if y_q is None or getattr(y_q, "shape", None) is None:
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
# bug no: GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_tflite_int8_quantize
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
#   testcases/tf_serialization_inputs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_tflite_int8_quantize.py \
#   > logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_tflite_int8_quantize_stdout.log \
#   2> logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_tflite_int8_quantize_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-SERIALIZATION_CHECKPOINTING-0007_tf_tflite_int8_quantize_stdout.log

# Observed output:
# exit_code=0
# Test Failed ❌

# Note:
# The tiny-model int8 quantization probe did not trigger a conversion or interpreter failure in this run.
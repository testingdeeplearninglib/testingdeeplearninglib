# GCFL-OTHER-0002_tf_tflite_float_dense_mul.py
import os, sys, json, random, traceback
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"))

def _final(line: str, code: int = 0):
    print(line, flush=True)
    sys.exit(code)

def _skip(r: str): _final(f"SKIP_ENV: {r}", 0)
def _pass(): _final("Test Passed ✅", 0)
def _fail(): _final("Test Failed ❌", 0)
def _harness_error(e: BaseException): _final(f"HARNESS_ERROR: {type(e).__name__}: {e}", 1)

def _get_int(name: str, d: int) -> int:
    v = os.environ.get(name, str(d))
    try: return int(v)
    except Exception: return d

def _env_line(tf, np, knobs: dict):
    payload = {
        "python": sys.version,
        "tensorflow": getattr(tf, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "eager": bool(tf.executing_eagerly()),
        "gpus": len(tf.config.list_physical_devices("GPU")),
        "cpus": len(tf.config.list_physical_devices("CPU")),
        "knobs": knobs,
    }
    print("ENV: " + json.dumps(payload, sort_keys=True), flush=True)

def _require_versions(tf):
    if not (sys.version_info.major == 3 and sys.version_info.minor in (10, 11)):
        _skip(f"python {sys.version_info.major}.{sys.version_info.minor} not in {{3.10,3.11}}")
    if str(getattr(tf, "__version__", "")) != "2.20.0":
        _skip(f"tensorflow version != 2.20.0 (got {getattr(tf,'__version__','?')})")

def _tflite_run(model_content: bytes, x):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_content=model_content)
    in0 = interp.get_input_details()[0]
    out0 = interp.get_output_details()[0]
    # Resize to match x
    interp.resize_tensor_input(in0["index"], list(x.shape), strict=True)
    interp.allocate_tensors()
    interp.set_tensor(in0["index"], x)
    interp.invoke()
    return interp.get_tensor(out0["index"])

def main():
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as e:
        _skip(f"import failed: {type(e).__name__}: {e}")

    seed = _get_int("SEED", 2026)
    iters = _get_int("ITERS", 10)
    batch = _get_int("BATCH", 1)
    d_model = _get_int("D_MODEL", 128)

    knobs = {"SEED": seed, "ITERS": iters, "BATCH": batch, "D_MODEL": d_model}
    _env_line(tf, np, knobs)
    _require_versions(tf)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Simple model that should be convertible and numerically stable in TFLite float
    inp = tf.keras.Input(shape=(d_model,), dtype=tf.float32, name="x")
    y = tf.keras.layers.Dense(32, use_bias=True)(inp)
    y = tf.keras.layers.Lambda(lambda t: t * 0.5)(y)
    y = tf.keras.layers.Dense(8, use_bias=False)(y)
    model = tf.keras.Model(inp, y)

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite = converter.convert()
    except Exception as e:
        # Conversion failure for this tiny model is suspicious
        _pass()

    # Differential check: TF vs TFLite should match closely for float model
    for i in range(max(1, iters)):
        x = np.random.RandomState(seed + i).randn(batch, d_model).astype(np.float32)
        try:
            tf_out = model(x, training=False).numpy()
            tfl_out = _tflite_run(tflite, x)
        except Exception:
            _pass()

        if tf_out.shape != tfl_out.shape:
            _pass()

        diff = np.max(np.abs(tf_out - tfl_out))
        if not np.isfinite(diff):
            _pass()

        # float TFLite should be very close
        if diff > 1e-4:
            _pass()

    _fail()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)



# Output:
# *****************
# bug no: GCFL-OTHER-0002_tf_tflite_float_dense_mul
# Result: Test Failed ❌
#
# Triggering command:
# conda activate tf_venv_220_py311
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=2
# export TF_DETERMINISTIC_OPS=1
# export KERAS_BACKEND=tensorflow
# export SEED=2026
# /home/talha/miniconda3/envs/tf_venv_220_py311/bin/python3.11 \
#   testcases/tf_batch_inputs_gcfl_other_0002/GCFL-OTHER-0002_tf_tflite_float_dense_mul.py \
#   > logs/GCFL-OTHER-0002_tf_tflite_float_dense_mul_stdout.log \
#   2> logs/GCFL-OTHER-0002_tf_tflite_float_dense_mul_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0002_tf_tflite_float_dense_mul_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# The probe did not detect suspicious behavior for the float dense-mul TFLite path.
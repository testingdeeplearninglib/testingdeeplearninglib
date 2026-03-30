# GCFL-OTHER-0002_tf_tflite_float_dynamic_reshape.py
import os, sys, json, random
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"))

def _final(line: str, code: int = 0):
    print(line, flush=True); sys.exit(code)
def _skip(r: str): _final(f"SKIP_ENV: {r}", 0)
def _pass(): _final("Test Passed ✅", 0)
def _fail(): _final("Test Failed ❌", 0)
def _harness_error(e: BaseException): _final(f"HARNESS_ERROR: {type(e).__name__}: {e}", 1)

def _get_int(k: str, d: int) -> int:
    try: return int(os.environ.get(k, str(d)))
    except Exception: return d

def _env_line(tf, np, knobs):
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

def _require(tf):
    if not (sys.version_info.major == 3 and sys.version_info.minor in (10, 11)):
        _skip("unsupported python")
    if str(getattr(tf, "__version__", "")) != "2.20.0":
        _skip(f"tensorflow version != 2.20.0 (got {getattr(tf,'__version__','?')})")

def _tflite_run(tf, tflite: bytes, x):
    interp = tf.lite.Interpreter(model_content=tflite)
    in0 = interp.get_input_details()[0]
    out0 = interp.get_output_details()[0]
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
    seq = _get_int("SEQ", 4)
    d_model = _get_int("D_MODEL", 16)

    knobs = {"SEED": seed, "ITERS": iters, "BATCH": batch, "SEQ": seq, "D_MODEL": d_model}
    _env_line(tf, np, knobs)
    _require(tf)

    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)

    # Reshape using dynamic shape ops: should still be convertible.
    x_in = tf.keras.Input(shape=(seq, d_model), dtype=tf.float32, name="x")
    y = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1]))(x_in)
    y = tf.keras.layers.Dense(8)(y)
    model = tf.keras.Model(x_in, y)

    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite = conv.convert()
    except Exception:
        _pass()

    for i in range(max(1, iters)):
        x = np.random.RandomState(seed + i).randn(batch, seq, d_model).astype(np.float32)
        try:
            tf_out = model(x, training=False).numpy()
            tfl_out = _tflite_run(tf, tflite, x)
        except Exception:
            _pass()
        if tf_out.shape != tfl_out.shape:
            _pass()
        diff = float(np.max(np.abs(tf_out - tfl_out)))
        if not np.isfinite(diff) or diff > 1e-4:
            _pass()

    _fail()

if __name__ == "__main__":
    try: main()
    except SystemExit: raise
    except Exception as e: _harness_error(e)




# Output:
# *****************
# bug no: GCFL-OTHER-0002_tf_tflite_float_dynamic_reshape
# Result: HARNESS_ERROR: ValueError: Shapes used to initialize variables must be fully-defined (no `None` dimensions). Received: shape=(None, 8) for variable path='dense/kernel'
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
#   testcases/tf_batch_inputs_gcfl_other_0002/GCFL-OTHER-0002_tf_tflite_float_dynamic_reshape.py \
#   > logs/GCFL-OTHER-0002_tf_tflite_float_dynamic_reshape_stdout.log \
#   2> logs/GCFL-OTHER-0002_tf_tflite_float_dynamic_reshape_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0002_tf_tflite_float_dynamic_reshape_stdout.log
#
# Observed output:
# exit_code=1
# HARNESS_ERROR: ValueError: Shapes used to initialize variables must be fully-defined (no `None` dimensions). Received: shape=(None, 8) for variable path='dense/kernel'

# Test Failed ❌
#
# Note:
# This is a testcase design error, not solid TFLite bug evidence. The model construction leaves the Dense input dimension undefined.
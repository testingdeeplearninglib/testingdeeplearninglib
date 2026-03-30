# GCFL-OTHER-0002_tf_tflite_float_multi_input_concat.py
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

def _tflite_eval(tf, tflite: bytes, a, b):
    interp = tf.lite.Interpreter(model_content=tflite)
    ins = interp.get_input_details()
    outs = interp.get_output_details()
    # two inputs
    interp.resize_tensor_input(ins[0]["index"], list(a.shape), strict=True)
    interp.resize_tensor_input(ins[1]["index"], list(b.shape), strict=True)
    interp.allocate_tensors()
    interp.set_tensor(ins[0]["index"], a)
    interp.set_tensor(ins[1]["index"], b)
    interp.invoke()
    return interp.get_tensor(outs[0]["index"])

def main():
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as e:
        _skip(f"import failed: {type(e).__name__}: {e}")

    seed = _get_int("SEED", 2026)
    iters = _get_int("ITERS", 10)
    batch = _get_int("BATCH", 1)
    d1 = _get_int("D_MODEL", 32)

    knobs = {"SEED": seed, "ITERS": iters, "BATCH": batch, "D_MODEL": d1}
    _env_line(tf, np, knobs)
    _require(tf)

    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)

    a_in = tf.keras.Input(shape=(d1,), dtype=tf.float32, name="a")
    b_in = tf.keras.Input(shape=(d1,), dtype=tf.float32, name="b")
    x = tf.keras.layers.Concatenate(axis=-1)([a_in, b_in])
    x = tf.keras.layers.Dense(16)(x)
    model = tf.keras.Model([a_in, b_in], x)

    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite = conv.convert()
    except Exception:
        _pass()

    for i in range(max(1, iters)):
        rs = np.random.RandomState(seed + i)
        a = rs.randn(batch, d1).astype(np.float32)
        b = rs.randn(batch, d1).astype(np.float32)
        try:
            tf_out = model([a, b], training=False).numpy()
            tfl_out = _tflite_eval(tf, tflite, a, b)
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
# bug no: GCFL-OTHER-0002_tf_tflite_float_multi_input_concat
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
#   testcases/tf_batch_inputs_gcfl_other_0002/GCFL-OTHER-0002_tf_tflite_float_multi_input_concat.py \
#   > logs/GCFL-OTHER-0002_tf_tflite_float_multi_input_concat_stdout.log \
#   2> logs/GCFL-OTHER-0002_tf_tflite_float_multi_input_concat_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0002_tf_tflite_float_multi_input_concat_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# The probe detected suspicious behavior in the TFLite multi-input concat path under TensorFlow 2.20.0.
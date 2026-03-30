# GCFL-OTHER-0002_tf_tflite_int8_mul_minmax_probe.py
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

def main():
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as e:
        _skip(f"import failed: {type(e).__name__}: {e}")

    seed = _get_int("SEED", 2026)
    batch = _get_int("BATCH", 1)
    d_model = _get_int("D_MODEL", 32)

    knobs = {"SEED": seed, "BATCH": batch, "D_MODEL": d_model}
    _env_line(tf, np, knobs)
    _require(tf)

    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)

    # MUL-heavy graph: sometimes quantization fails with "Unable to quantize buffer/min/max..."
    a_in = tf.keras.Input(shape=(d_model,), dtype=tf.float32, name="a")
    b_in = tf.keras.Input(shape=(d_model,), dtype=tf.float32, name="b")
    y = tf.keras.layers.Lambda(lambda xs: xs[0] * xs[1])([a_in, b_in])
    y = tf.keras.layers.Lambda(lambda t: t * 0.25 + 0.1)(y)
    y = tf.keras.layers.Dense(8)(y)
    model = tf.keras.Model([a_in, b_in], y)

    rs = np.random.RandomState(seed)
    repA = rs.randn(32, d_model).astype(np.float32)
    repB = rs.randn(32, d_model).astype(np.float32)

    def rep_gen():
        for i in range(16):
            yield [repA[i:i+1], repB[i:i+1]]

    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        _ = conv.convert()
    except Exception as e:
        msg = str(e).lower()
        # Treat "unable to quantize buffer/min/max" (cluster evidence) as suspicious
        if ("unable to quantize" in msg) or ("min/max" in msg) or ("min max" in msg):
            _pass()
        # Internal error types are suspicious too
        if isinstance(e, (UnboundLocalError, AttributeError, KeyError)):
            _pass()
        # Otherwise likely a normal conversion limitation; not a bug signal
        _fail()

    # If conversion succeeds, we did NOT reproduce that failure pattern.
    _fail()

if __name__ == "__main__":
    try: main()
    except SystemExit: raise
    except Exception as e: _harness_error(e)




# Output:
# *****************
# bug no: GCFL-OTHER-0002_tf_tflite_int8_mul_minmax_probe
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
#   testcases/tf_batch_inputs_gcfl_other_0002/GCFL-OTHER-0002_tf_tflite_int8_mul_minmax_probe.py \
#   > logs/GCFL-OTHER-0002_tf_tflite_int8_mul_minmax_probe_stdout.log \
#   2> logs/GCFL-OTHER-0002_tf_tflite_int8_mul_minmax_probe_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0002_tf_tflite_int8_mul_minmax_probe_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# The min/max quantization probe did not reproduce the targeted conversion failure signature in this run.
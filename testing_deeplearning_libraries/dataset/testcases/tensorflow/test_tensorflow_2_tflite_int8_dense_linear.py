# GCFL-OTHER-0002_tf_tflite_int8_dense_linear.py
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
    iters = _get_int("ITERS", 5)
    batch = _get_int("BATCH", 1)
    d_model = _get_int("D_MODEL", 32)

    knobs = {"SEED": seed, "ITERS": iters, "BATCH": batch, "D_MODEL": d_model}
    _env_line(tf, np, knobs)
    _require(tf)

    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)

    inp = tf.keras.Input(shape=(d_model,), dtype=tf.float32, name="x")
    y = tf.keras.layers.Dense(8, use_bias=True)(inp)  # linear only
    model = tf.keras.Model(inp, y)

    rs = np.random.RandomState(seed)
    rep_data = rs.randn(32, d_model).astype(np.float32)

    def rep_gen():
        for i in range(16):
            yield [rep_data[i:i+1]]

    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        tflite = conv.convert()
    except Exception as e:
        # For a linear dense, INT8 conversion should generally succeed.
        _pass()

    # Run interpreter on a few inputs
    try:
        interp = tf.lite.Interpreter(model_content=tflite)
        in0 = interp.get_input_details()[0]
        out0 = interp.get_output_details()[0]
        interp.resize_tensor_input(in0["index"], [batch, d_model], strict=True)
        interp.allocate_tensors()
        in_scale, in_zp = in0.get("quantization", (0.0, 0))
        out_scale, out_zp = out0.get("quantization", (0.0, 0))
        if not (in_scale and out_scale):
            _pass()
    except Exception:
        _pass()

    for i in range(max(1, iters)):
        x = np.random.RandomState(seed + i).randn(batch, d_model).astype(np.float32)
        try:
            tf_out = model(x, training=False).numpy()
            qx = (x / in_scale + in_zp).round().astype(np.int8)
            interp.set_tensor(in0["index"], qx)
            interp.invoke()
            qy = interp.get_tensor(out0["index"])
            y = (qy.astype(np.float32) - float(out_zp)) * float(out_scale)
        except Exception:
            _pass()

        if not (y.shape == tf_out.shape):
            _pass()

        # Loose sanity: finite outputs; quant error shouldn't be insane for linear model
        if not (np.isfinite(y).all() and np.isfinite(tf_out).all()):
            _pass()
        max_abs = float(np.max(np.abs(tf_out)))
        err = float(np.max(np.abs(y - tf_out)))
        if max_abs < 1e-6:
            # avoid divide-by-small
            if err > 1e-2:
                _pass()
        else:
            if (err / (max_abs + 1e-6)) > 0.35:
                _pass()

    _fail()

if __name__ == "__main__":
    try: main()
    except SystemExit: raise
    except Exception as e: _harness_error(e)



# Output:
# *****************
# bug no: GCFL-OTHER-0002_tf_tflite_int8_dense_linear
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
#   testcases/tf_batch_inputs_gcfl_other_0002/GCFL-OTHER-0002_tf_tflite_int8_dense_linear.py \
#   > logs/GCFL-OTHER-0002_tf_tflite_int8_dense_linear_stdout.log \
#   2> logs/GCFL-OTHER-0002_tf_tflite_int8_dense_linear_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0002_tf_tflite_int8_dense_linear_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# The INT8 dense linear quantization probe did not reproduce suspicious behavior under this environment.
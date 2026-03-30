# FILE: GCFL-OTHER-0001_tf_case10_tflite_int8_quantize_converter_crash_proxy.py
import os
import sys
import json
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")


def _skip(r): print(f"SKIP_ENV: {r}"); sys.exit(0)
def _pass_(): print("Test Passed ✅"); sys.exit(0)
def _fail_(): print("Test Failed ❌"); sys.exit(0)
def _herr(e): print(f"HARNESS_ERROR: {type(e).__name__}: {e}"); sys.exit(1)


def _env_int(k, d):
    v = os.environ.get(k, "").strip()
    if not v:
        return d
    try:
        return int(v)
    except Exception:
        return d


def main():
    try:
        import numpy as np
    except Exception as e:
        _skip(f"numpy_import_failed:{type(e).__name__}:{e}")
    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"tf_import_failed:{type(e).__name__}:{e}")

    if not (sys.version_info.major == 3 and sys.version_info.minor in (10, 11)):
        _skip(f"python_not_supported:{sys.version_info.major}.{sys.version_info.minor}")
    if tf.__version__ != "2.20.0":
        _skip(f"tf_version_mismatch:{tf.__version__}")
    if not hasattr(tf, "lite"):
        _skip("tf_lite_missing")

    seed = _env_int("SEED", 2026)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    env_payload = {
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "testcase": "tflite_int8_quantize_converter_crash_proxy",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    keras = tf.keras

    # tiny model with ops that sometimes trip quantization paths (mul + tanh)
    inp = keras.Input(shape=(8,), dtype=tf.float32)
    x = keras.layers.Dense(8, use_bias=False)(inp)
    x = tf.math.tanh(x)
    x = tf.math.multiply(x, x)
    model = keras.Model(inp, x)

    def rep_data():
        rs = np.random.RandomState(seed)
        for _ in range(32):
            yield [rs.randn(1, 8).astype("float32")]

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        _ = converter.convert()
        # success means no suspicious crash/exception
        _fail_()
    except Exception as e:
        msg = str(e).lower()
        # Only treat "unexpected" converter failures as suspicious; skip known unsupported-op messages
        if "unsupported" in msg or "not supported" in msg or "requires" in msg:
            _skip(f"tflite_known_unsupported:{type(e).__name__}")
        # otherwise, suspicious converter exception
        _pass_()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _herr(e)



# Output:
# *****************
# bug no: GCFL-OTHER-0001-10
# Result: HARNESS_ERROR
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
#   testcases/tf_other_0001_inputs/GCFL-OTHER-0001-10_tflite_int8_quantize_converter_crash_proxy.py \
#   > logs/GCFL-OTHER-0001-10_stdout.log \
#   2> logs/GCFL-OTHER-0001-10_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0001-10_stdout.log
#
# Observed output:
# exit_code=1
# HARNESS_ERROR: ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.ops`). You are likely doing something like:
#Test Failed ❌
#
# Note:
# This is a testcase construction error, not valid bug evidence yet. The script mixes raw TensorFlow ops
# directly with Keras symbolic tensors in a way Keras 3 rejects before the TFLite converter path is even reached.
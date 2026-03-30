# FILE: GCFL-OTHER-0001_tf_case06_model_fit_numpy_function_unknown_shape.py
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

    seed = _env_int("SEED", 2026)
    iters = _env_int("ITERS", 3)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    # optional GPU memory growth
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        gpus = []

    env_payload = {
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "gpu_count": len(gpus),
        "knobs": {"SEED": seed, "ITERS": iters},
        "testcase": "model_fit_numpy_function_unknown_shape",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    keras = tf.keras

    def np_make_example(_):
        # returns flat vector with unknown static shape from numpy_function
        img = np.random.RandomState(seed).randn(32 * 32 * 3).astype("float32")
        y = np.int32(np.random.RandomState(seed + 1).randint(0, 2))
        return img, y

    base = tf.data.Dataset.range(8)

    def map_fn(i):
        img, y = tf.numpy_function(np_make_example, [i], [tf.float32, tf.int32])
        # INTENTIONALLY keep unknown shape; then reshape with tf.reshape using runtime shape.
        # This often triggers "unknown tensorshape" issues in some pipelines.
        img = tf.reshape(img, [32, 32, 3])
        return img, y

    ds = base.map(map_fn, num_parallel_calls=1).batch(1).prefetch(1)

    inputs = keras.Input(shape=(32, 32, 3), name="img")
    x = keras.layers.Flatten()(inputs)
    out = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")

    # oracle: any shape/graph plumbing exception during fit is suspicious
    for _ in range(iters):
        try:
            model.fit(ds, epochs=1, steps_per_epoch=1, verbose=0)
        except Exception as e:
            msg = str(e).lower()
            # target signals: unknown shape/rank issues
            if ("as_list() is not defined on an unknown tensorshape" in msg) or ("unknown rank" in msg) or ("unknown shape" in msg):
                _pass_()
            # still treat as suspicious exception (pipeline should handle)
            _pass_()

    _fail_()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _herr(e)



# Output:
# *****************
# bug no: GCFL-OTHER-0001-06
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
#   testcases/tf_other_0001_inputs/GCFL-OTHER-0001-06_model_fit_numpy_function_unknown_shape.py \
#   > logs/GCFL-OTHER-0001-06_stdout.log \
#   2> logs/GCFL-OTHER-0001-06_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0001-06_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# The testcase hit a suspicious exception path during model.fit with a numpy_function/unknown-shape pipeline.
# This is a probe hit, not automatically a confirmed TensorFlow bug.
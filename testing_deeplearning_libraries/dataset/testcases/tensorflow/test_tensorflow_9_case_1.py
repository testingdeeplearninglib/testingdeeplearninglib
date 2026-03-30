# GCFL-DATA_PIPELINE-0009-02.py
# Target: TF 2.20.* | Oracle: exception from model.fit when Dataset elements have unknown rank (tf.py_function variant)

import os, sys, json, traceback
import random
import numpy as np

_DONE = False
def _final(line: str, code: int = 0):
    global _DONE
    if _DONE: return
    _DONE = True
    print(line, flush=True)
    raise SystemExit(code)

def _skip(r): _final(f"SKIP_ENV: {r}", 0)
def _pass(): _final("Test Passed ✅", 0)
def _fail(): _final("Test Failed ❌", 0)
def _herr(msg): _final(f"HARNESS_ERROR: {msg}", 1)

def _get_int(k, d):
    v = os.environ.get(k, "").strip()
    if not v: return d
    try: return int(v)
    except Exception: return d

def main():
    if not (sys.version_info.major == 3 and sys.version_info.minor in (10, 11)):
        _skip(f"python {sys.version_info.major}.{sys.version_info.minor} not in {{3.10,3.11}}")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"))

    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"tensorflow import failed: {type(e).__name__}: {e}")

    tfv = str(getattr(tf, "__version__", ""))
    if not tfv.startswith("2.20."):
        _skip(f"tensorflow version {tfv} != 2.20.*")

    seed = _get_int("SEED", 2026)
    batch = _get_int("BATCH", 1)

    random.seed(seed)
    np.random.seed(seed)
    try: tf.random.set_seed(seed)
    except Exception: pass

    env_payload = {
        "test_id": "GCFL-DATA_PIPELINE-0009-02",
        "python": sys.version.split()[0],
        "tensorflow": tfv,
        "numpy": np.__version__,
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
        "cpu_count": len(tf.config.list_physical_devices("CPU")),
        "knobs": {"SEED": seed, "BATCH": batch},
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True), flush=True)

    keras = tf.keras

    def py_make(_):
        x = np.random.RandomState(seed).randn(32 * 32 * 3).astype(np.float32)  # flat, shape unknown to TF
        y = np.int32(1)
        return x, y

    ds = tf.data.Dataset.range(8)

    def map_fn(i):
        x, y = tf.py_function(py_make, [i], [tf.float32, tf.int32])
        # intentionally leave shape unspecified
        return x, y

    ds = ds.map(map_fn, num_parallel_calls=1).batch(max(1, batch)).prefetch(1)

    inp = keras.Input(shape=(32, 32, 3))
    out = keras.layers.Dense(1)(keras.layers.Flatten()(inp))
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")

    try:
        model.fit(ds, epochs=1, steps_per_epoch=1, verbose=0)
        _fail()
    except Exception as e:
        m = str(e).lower()
        if isinstance(e, (ValueError, TypeError)) and ("unknown rank" in m or "unknown tensorshape" in m):
            _pass()
        _fail()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        _herr(traceback.format_exc().strip())




# Output:
# *****************


# Triggering command:
# conda activate tf_venv_220_py311
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=2
# export TF_DETERMINISTIC_OPS=1
# export KERAS_BACKEND=tensorflow
# export SEED=2026
# /home/talha/miniconda3/envs/tf_venv_220_py311/bin/python3.11 \
#   testcases/tf_batch_dp0009_inputs/GCFL-DATA_PIPELINE-0009-02.py \
#   > logs/GCFL-DATA_PIPELINE-0009-02_stdout.log \
#   2> logs/GCFL-DATA_PIPELINE-0009-02_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-DATA_PIPELINE-0009-02_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
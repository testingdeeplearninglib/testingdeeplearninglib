# GCFL-DATA_PIPELINE-0009-07.py
# Target: TF 2.20.* | Oracle: hang detection with from_generator + map(py_function) + prefetch

import os, sys, json, traceback, threading
import numpy as np
import random

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
def _herr(m): _final(f"HARNESS_ERROR: {m}", 1)

def _get_int(k, d):
    v = os.environ.get(k, "").strip()
    if not v: return d
    try: return int(v)
    except Exception: return d

def main():
    if not (sys.version_info.major == 3 and sys.version_info.minor in (10, 11)):
        _skip("python not in {3.10,3.11}")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"))

    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"tensorflow import failed: {type(e).__name__}: {e}")

    tfv = str(getattr(tf, "__version__", ""))
    if not tfv.startswith("2.20."):
        _skip(f"tensorflow version {tfv} != 2.20.*")

    seed = _get_int("SEED", 2026)
    timeout_s = _get_int("TIMEOUT_S", 20)
    steps = _get_int("STEPS", 3)

    random.seed(seed)
    np.random.seed(seed)
    try: tf.random.set_seed(seed)
    except Exception: pass

    env_payload = {
        "test_id": "GCFL-DATA_PIPELINE-0009-07",
        "python": sys.version.split()[0],
        "tensorflow": tfv,
        "numpy": np.__version__,
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
        "cpu_count": len(tf.config.list_physical_devices("CPU")),
        "knobs": {"SEED": seed, "TIMEOUT_S": timeout_s, "STEPS": steps},
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True), flush=True)

    keras = tf.keras

    def gen():
        rs = np.random.RandomState(seed)
        while True:
            x = rs.randn(32, 32, 3).astype(np.float32)
            y = np.int32(rs.randint(0, 2))
            yield x, y

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    def py_aug(x, y):
        # minimal python-side transform (could stress pipeline crossing)
        return x * np.float32(1.0), y

    def map_fn(x, y):
        x2, y2 = tf.py_function(py_aug, [x, y], [tf.float32, tf.int32])
        x2.set_shape((32, 32, 3))
        y2.set_shape(())
        return x2, y2

    ds = ds.map(map_fn, num_parallel_calls=1).prefetch(2)

    inp = keras.Input(shape=(32, 32, 3))
    out = keras.layers.Dense(1)(keras.layers.Flatten()(inp))
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")

    res = {"done": False}

    def _run():
        try:
            model.fit(ds, epochs=1, steps_per_epoch=steps, verbose=0)
        finally:
            res["done"] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=float(timeout_s))

    if t.is_alive():
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
#   testcases/tf_batch_dp0009_inputs/GCFL-DATA_PIPELINE-0009-07.py \
#   > logs/GCFL-DATA_PIPELINE-0009-07_stdout.log \
#   2> logs/GCFL-DATA_PIPELINE-0009-07_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-DATA_PIPELINE-0009-07_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
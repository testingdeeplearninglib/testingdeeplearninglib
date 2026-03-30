# GCFL-DATA_PIPELINE-0009-10.py
# Target: TF 2.20.* | Oracle: hang detection in dataset pipeline iteration (iterator stall) + bounded timeout

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
    iters = _get_int("ITERS", 50)
    timeout_s = _get_int("TIMEOUT_S", 15)

    random.seed(seed)
    np.random.seed(seed)
    try: tf.random.set_seed(seed)
    except Exception: pass

    env_payload = {
        "test_id": "GCFL-DATA_PIPELINE-0009-10",
        "python": sys.version.split()[0],
        "tensorflow": tfv,
        "numpy": np.__version__,
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
        "cpu_count": len(tf.config.list_physical_devices("CPU")),
        "knobs": {"SEED": seed, "ITERS": iters, "TIMEOUT_S": timeout_s},
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True), flush=True)

    # Create a pipeline with python boundary + prefetch; then iterate in a thread and detect stall.
    def gen():
        rs = np.random.RandomState(seed)
        for _ in range(iters):
            x = rs.randn(16).astype(np.float32)
            yield x

    ds = tf.data.Dataset.from_generator(gen, output_signature=tf.TensorSpec(shape=(16,), dtype=tf.float32))

    def py_map(x):
        # tiny python transform (forces crossing runtime boundary)
        arr = x.numpy()
        return np.asarray(arr * 1.0, dtype=np.float32)

    def map_fn(x):
        y = tf.py_function(py_map, [x], Tout=tf.float32)
        y.set_shape((16,))
        return y

    ds = ds.map(map_fn, num_parallel_calls=1).prefetch(2)

    res = {"count": 0, "done": False}

    def _consume():
        try:
            for _ in ds.as_numpy_iterator():
                res["count"] += 1
        finally:
            res["done"] = True

    t = threading.Thread(target=_consume, daemon=True)
    t.start()
    t.join(timeout=float(timeout_s))

    if t.is_alive():
        # iterator stalled / deadlocked
        _pass()
    # If it finishes but consumed far fewer than expected, also suspicious
    if res["count"] < max(1, iters // 4):
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
#   testcases/tf_batch_dp0009_inputs/GCFL-DATA_PIPELINE-0009-10.py \
#   > logs/GCFL-DATA_PIPELINE-0009-10_stdout.log \
#   2> logs/GCFL-DATA_PIPELINE-0009-10_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-DATA_PIPELINE-0009-10_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
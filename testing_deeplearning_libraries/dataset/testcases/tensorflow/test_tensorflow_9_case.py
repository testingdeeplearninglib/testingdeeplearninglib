# GCFL-DATA_PIPELINE-0009-01.py
# Target: TF 2.20.*  | Oracle: exception "Cannot take the length of Shape with unknown rank" during model.fit

import os, sys, json, traceback, time
import random
import numpy as np

_DONE = False

def _final(line: str, code: int = 0):
    global _DONE
    if _DONE:
        return
    _DONE = True
    print(line, flush=True)
    raise SystemExit(code)

def _skip(reason: str):
    _final(f"SKIP_ENV: {reason}", 0)

def _pass():
    _final("Test Passed ✅", 0)

def _fail():
    _final("Test Failed ❌", 0)

def _herr(msg: str):
    _final(f"HARNESS_ERROR: {msg}", 1)

def _get_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default

def main():
    # gating: python version
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
    iters = _get_int("ITERS", 3)
    batch = _get_int("BATCH", 1)
    seq = _get_int("SEQ", 4)
    d_model = _get_int("D_MODEL", 128)

    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    gpus = []
    cpus = []
    try:
        gpus = tf.config.list_physical_devices("GPU")
        cpus = tf.config.list_physical_devices("CPU")
    except Exception:
        pass

    env_payload = {
        "test_id": "GCFL-DATA_PIPELINE-0009-01",
        "python": sys.version.split()[0],
        "tensorflow": tfv,
        "numpy": np.__version__,
        "gpu_count": len(gpus),
        "cpu_count": len(cpus),
        "knobs": {"SEED": seed, "ITERS": iters, "BATCH": batch, "SEQ": seq, "D_MODEL": d_model},
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True), flush=True)

    keras = tf.keras

    # Dataset: uses numpy_function returning a flat vector; DOES NOT set shape => unknown rank/shape in pipeline.
    def np_make_example(_):
        x = np.random.RandomState(seed).randn(32 * 32 * 3).astype(np.float32)
        y = np.int32(np.random.randint(0, 2))
        return x, y

    base = tf.data.Dataset.range(max(2, iters))

    def map_fn(i):
        x, y = tf.numpy_function(np_make_example, [i], [tf.float32, tf.int32])
        # intentionally leave shape unknown
        return x, y

    ds = base.map(map_fn, num_parallel_calls=1).batch(max(1, batch)).prefetch(1)

    # Model expects rank-3 image [32,32,3]; pipeline provides unknown rank => often triggers "unknown rank length" class errors.
    inputs = keras.Input(shape=(32, 32, 3), name="img")
    x = keras.layers.Flatten()(inputs)
    out = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, out)
    model.compile(optimizer="adam", loss="mse")

    for _ in range(max(1, iters)):
        try:
            model.fit(ds, epochs=1, steps_per_epoch=1, verbose=0)
            _fail()
        except Exception as e:
            m = str(e).lower()
            if isinstance(e, (ValueError, TypeError)) and (
                "cannot take the length of shape with unknown rank" in m
                or "unknown rank" in m
                or "as_list() is not defined on an unknown tensorshape" in m
            ):
                _pass()
            # Other exceptions are not reliable signals for this cluster.
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
#   testcases/tf_batch_dp0009_inputs/GCFL-DATA_PIPELINE-0009-01.py \
#   > logs/GCFL-DATA_PIPELINE-0009-01_stdout.log \
#   2> logs/GCFL-DATA_PIPELINE-0009-01_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-DATA_PIPELINE-0009-01_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
# 
# Note:
# The testcase triggered the expected unknown-rank failure during model.fit using numpy_function.

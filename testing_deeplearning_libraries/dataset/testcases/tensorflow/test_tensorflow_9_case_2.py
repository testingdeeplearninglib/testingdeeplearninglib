# GCFL-DATA_PIPELINE-0009-03.py
# Target: TF 2.20.* | Oracle: TFRecordDataset cross-feature corruption (integrity check under shuffle/map/prefetch)

import os, sys, json, tempfile, traceback
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
    iters = _get_int("ITERS", 20)
    n_examples = max(10, iters)

    random.seed(seed)
    np.random.seed(seed)
    try: tf.random.set_seed(seed)
    except Exception: pass

    env_payload = {
        "test_id": "GCFL-DATA_PIPELINE-0009-03",
        "python": sys.version.split()[0],
        "tensorflow": tfv,
        "numpy": np.__version__,
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
        "cpu_count": len(tf.config.list_physical_devices("CPU")),
        "knobs": {"SEED": seed, "ITERS": iters, "N_EX": n_examples},
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True), flush=True)

    def _make_example(i: int) -> bytes:
        # payload is deterministic function of id: payload[j] = id + j
        feat = {
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
            "payload": tf.train.Feature(float_list=tf.train.FloatList(value=[float(i), float(i+1), float(i+2)])),
        }
        ex = tf.train.Example(features=tf.train.Features(feature=feat))
        return ex.SerializeToString()

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "a.tfrecord")
        with tf.io.TFRecordWriter(p) as w:
            for i in range(n_examples):
                w.write(_make_example(i))

        spec = {
            "id": tf.io.FixedLenFeature([], tf.int64),
            "payload": tf.io.FixedLenFeature([3], tf.float32),
        }

        def parse_fn(s):
            x = tf.io.parse_single_example(s, spec)
            return x["id"], x["payload"]

        ds = tf.data.TFRecordDataset([p])
        ds = ds.shuffle(buffer_size=min(64, n_examples), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(parse_fn, num_parallel_calls=2).prefetch(2)

        # Integrity check: payload must match id within same element
        k = 0
        for (idv, pay) in ds.take(iters).as_numpy_iterator():
            k += 1
            id_int = int(idv)
            exp = np.array([id_int, id_int + 1, id_int + 2], dtype=np.float32)
            if pay.shape != (3,) or not np.allclose(pay, exp, atol=0.0, rtol=0.0):
                _pass()  # suspicious: cross-example feature swap/corruption
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
#   testcases/tf_batch_dp0009_inputs/GCFL-DATA_PIPELINE-0009-03.py \
#   > logs/GCFL-DATA_PIPELINE-0009-03_stdout.log \
#   2> logs/GCFL-DATA_PIPELINE-0009-03_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-DATA_PIPELINE-0009-03_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌

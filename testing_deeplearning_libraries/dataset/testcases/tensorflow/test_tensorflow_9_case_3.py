# GCFL-DATA_PIPELINE-0009-04.py
# Target: TF 2.20.* | Oracle: TFRecord interleave/shuffle corruption check across 2 files

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
    iters = _get_int("ITERS", 30)
    n = max(12, iters)

    random.seed(seed)
    np.random.seed(seed)
    try: tf.random.set_seed(seed)
    except Exception: pass

    env_payload = {
        "test_id": "GCFL-DATA_PIPELINE-0009-04",
        "python": sys.version.split()[0],
        "tensorflow": tfv,
        "numpy": np.__version__,
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
        "cpu_count": len(tf.config.list_physical_devices("CPU")),
        "knobs": {"SEED": seed, "ITERS": iters, "N_EX": n},
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True), flush=True)

    def make_ser(i: int, shard: int) -> bytes:
        # encode shard into payload for stronger consistency check
        # payload = [shard, id, id^2]
        feat = {
            "shard": tf.train.Feature(int64_list=tf.train.Int64List(value=[shard])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
            "payload": tf.train.Feature(float_list=tf.train.FloatList(value=[float(shard), float(i), float(i*i)])),
        }
        ex = tf.train.Example(features=tf.train.Features(feature=feat))
        return ex.SerializeToString()

    with tempfile.TemporaryDirectory() as td:
        p0 = os.path.join(td, "s0.tfrecord")
        p1 = os.path.join(td, "s1.tfrecord")
        with tf.io.TFRecordWriter(p0) as w:
            for i in range(n):
                w.write(make_ser(i, 0))
        with tf.io.TFRecordWriter(p1) as w:
            for i in range(n):
                w.write(make_ser(i, 1))

        spec = {
            "shard": tf.io.FixedLenFeature([], tf.int64),
            "id": tf.io.FixedLenFeature([], tf.int64),
            "payload": tf.io.FixedLenFeature([3], tf.float32),
        }

        def parse_fn(s):
            x = tf.io.parse_single_example(s, spec)
            return x["shard"], x["id"], x["payload"]

        files = tf.data.Dataset.from_tensor_slices([p0, p1])
        ds = files.interleave(lambda fn: tf.data.TFRecordDataset([fn]),
                              cycle_length=2, num_parallel_calls=2, deterministic=False)
        ds = ds.shuffle(buffer_size=64, seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(parse_fn, num_parallel_calls=2).prefetch(2)

        for (sh, iid, pay) in ds.take(iters).as_numpy_iterator():
            sh = int(sh); iid = int(iid)
            exp = np.array([float(sh), float(iid), float(iid*iid)], dtype=np.float32)
            if pay.shape != (3,) or not np.allclose(pay, exp, atol=0.0, rtol=0.0):
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
#   testcases/tf_batch_dp0009_inputs/GCFL-DATA_PIPELINE-0009-04.py \
#   > logs/GCFL-DATA_PIPELINE-0009-04_stdout.log \
#   2> logs/GCFL-DATA_PIPELINE-0009-04_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-DATA_PIPELINE-0009-04_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
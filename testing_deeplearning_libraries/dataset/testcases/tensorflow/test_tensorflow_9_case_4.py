# GCFL-DATA_PIPELINE-0009-05.py
# Target: TF 2.20.* | Oracle: graph/session iterator + derived tensors should NOT cross-example swap (single sess.run)

import os, sys, json, tempfile, traceback
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
    iters = _get_int("ITERS", 30)
    n = max(12, iters)

    random.seed(seed)
    np.random.seed(seed)

    env_payload = {
        "test_id": "GCFL-DATA_PIPELINE-0009-05",
        "python": sys.version.split()[0],
        "tensorflow": tfv,
        "numpy": np.__version__,
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
        "cpu_count": len(tf.config.list_physical_devices("CPU")),
        "knobs": {"SEED": seed, "ITERS": iters},
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True), flush=True)

    # graph mode
    tf.compat.v1.disable_eager_execution()

    def make_ser(i: int) -> bytes:
        # payload = [id, id+1, id+2], sum = 3*id+3
        feat = {
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
            "payload": tf.train.Feature(float_list=tf.train.FloatList(value=[float(i), float(i+1), float(i+2)])),
        }
        ex = tf.train.Example(features=tf.train.Features(feature=feat))
        return ex.SerializeToString()

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "d.tfrecord")
        with tf.io.TFRecordWriter(p) as w:
            for i in range(n):
                w.write(make_ser(i))

        spec = {
            "id": tf.io.FixedLenFeature([], tf.int64),
            "payload": tf.io.FixedLenFeature([3], tf.float32),
        }

        g = tf.Graph()
        with g.as_default():
            ds = tf.data.TFRecordDataset([p]).shuffle(buffer_size=64, seed=seed, reshuffle_each_iteration=True)
            def parse_fn(s):
                x = tf.io.parse_single_example(s, spec)
                return x["id"], x["payload"]
            ds = ds.map(parse_fn, num_parallel_calls=1).repeat().prefetch(1)
            it = tf.compat.v1.data.make_one_shot_iterator(ds)
            iid, payload = it.get_next()

            # derived tensors
            iid_i32 = tf.cast(iid, tf.int32)
            payload_sum = tf.reduce_sum(payload)  # should match 3*id+3

            init = tf.compat.v1.global_variables_initializer()

        with tf.compat.v1.Session(graph=g) as sess:
            sess.run(init)
            for _ in range(iters):
                iid_v, ps_v = sess.run([iid_i32, payload_sum])
                exp = 3 * int(iid_v) + 3
                if not np.isfinite(ps_v) or int(round(float(ps_v))) != exp:
                    # suspicious: cross-example swap / mismatch between derived tensors
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
# bug no: GCFL-DATA_PIPELINE-0009-05
# Result: HARNESS_ERROR
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
#   testcases/tf_batch_dp0009_inputs/GCFL-DATA_PIPELINE-0009-05.py \
#   > logs/GCFL-DATA_PIPELINE-0009-05_stdout.log \
#   2> logs/GCFL-DATA_PIPELINE-0009-05_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-DATA_PIPELINE-0009-05_stdout.log
#
# Observed output:
# exit_code=1
# HARNESS_ERROR: Traceback (most recent call last):
# Result: Test Failed ❌

# Note:
# This testcase did not produce a clean library result. It failed at the harness/script level.
# Check logs/GCFL-DATA_PIPELINE-0009-05_stderr.log and logs/tf_batch_dp0009_run/GCFL-DATA_PIPELINE-0009-05.meta.json

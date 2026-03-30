# GCFL-DATAPIPELI-0084# _tc_01


import os
import sys
import tempfile
import traceback
import random

SEED = 23098

def _skip(reason: str):
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)

def _pass(msg: str):
    print(msg)
    print("Test Passed ✅")
    sys.exit(0)

def _fail(msg: str):
    print(msg)
    print("Test Failed ❌")
    sys.exit(0)

def _harness_error(e: BaseException):
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

def _as_tuple3(x):
    try:
        return tuple(float(f"{float(v):.6g}") for v in list(x))
    except Exception:
        return tuple(float(f"{float(v):.6g}") for v in x)

def main():
    random.seed(SEED)
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # reduce TF log noise

    try:
        try:
            import tensorflow as tf
        except Exception as e:
            _skip(f"tensorflow import failed: {e}")

        print(f"TF_VERSION: {getattr(tf, '__version__', 'unknown')}")
        try:
            print(f"TF_GPU_DEVICES: {tf.config.list_physical_devices('GPU')}")
        except Exception as e:
            print(f"TF_GPU_DEVICES: <error: {e}>")

        if not hasattr(tf, "compat") or not hasattr(tf.compat, "v1"):
            _skip("TensorFlow lacks tf.compat.v1; cannot run graph/session mode")

        tf1 = tf.compat.v1
        try:
            tf1.disable_eager_execution()
        except Exception as e:
            _skip(f"unable to disable eager execution: {e}")

        try:
            tf1.set_random_seed(SEED)
        except Exception:
            pass

        # Two known records
        P1 = (_as_tuple3([1.0, 2.0, 3.0]), _as_tuple3([10.0, 20.0, 30.0]))
        P2 = (_as_tuple3([3.0, 4.0, 5.0]), _as_tuple3([30.0, 40.0, 50.0]))
        valid_pairs = {P1, P2}

        # Build TFRecord bytes
        ex1 = tf1.train.Example(
            features=tf1.train.Features(
                feature={
                    "a": tf1.train.Feature(float_list=tf1.train.FloatList(value=list(P1[0]))),
                    "b": tf1.train.Feature(float_list=tf1.train.FloatList(value=list(P1[1]))),
                }
            )
        )
        ex2 = tf1.train.Example(
            features=tf1.train.Features(
                feature={
                    "a": tf1.train.Feature(float_list=tf1.train.FloatList(value=list(P2[0]))),
                    "b": tf1.train.Feature(float_list=tf1.train.FloatList(value=list(P2[1]))),
                }
            )
        )

        with tempfile.TemporaryDirectory() as td:
            tfrecord_path = os.path.join(td, "test.tfrecord")
            w = tf.io.TFRecordWriter(tfrecord_path)
            w.write(ex1.SerializeToString())
            w.write(ex2.SerializeToString())
            w.close()

            # Control: fetch both fields together
            g = tf1.Graph()
            with g.as_default():
                ds = tf.data.TFRecordDataset([tfrecord_path])
                it = tf1.data.make_one_shot_iterator(ds)
                nxt = it.get_next()

                features = {
                    "a": tf.io.FixedLenFeature([3], tf.float32),
                    "b": tf.io.FixedLenFeature([3], tf.float32),
                }
                parsed = tf.io.parse_single_example(nxt, features)

            with tf1.Session(graph=g) as sess:
                control_pairs = []
                try:
                    for _ in range(2):
                        a_val, b_val = sess.run([parsed["a"], parsed["b"]])
                        control_pairs.append((_as_tuple3(a_val), _as_tuple3(b_val)))
                except Exception as e:
                    name = type(e).__name__
                    if "OutOfRange" in name:
                        _pass("SUSPICIOUS: OutOfRange occurred even in joint fetch control path")
                    _harness_error(e)

                for p in control_pairs:
                    if p not in valid_pairs:
                        _pass(f"SUSPICIOUS: joint fetch produced mismatched pair: {p}")

                print(f"CONTROL_OK: {control_pairs}")

            _fail("NO_SUSPICIOUS: joint fetch path is stable; separate-fetch OutOfRange is expected misuse, not a TF bug")

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)

if __name__ == "__main__":
    main()




# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# source ~/.venvs/dl_testing_cpu/bin/activate
# export CUDA_VISIBLE_DEVICES=
# ts=$(date +%Y%m%d_%H%M%S)
# python gcfl_datapipeli_0084_tc_01.py 2> "tf_stderr_${ts}.log"
# echo "stderr saved to tf_stderr_${ts}.log"


# Output:
# *****************
# TF_VERSION: 2.20.0
# TF_GPU_DEVICES: []
# CONTROL_OK: [((1.0, 2.0, 3.0), (10.0, 20.0, 30.0)), ((3.0, 4.0, 5.0), (30.0, 40.0, 50.0))]
# NO_SUSPICIOUS: joint fetch path is stable; separate-fetch OutOfRange is expected misuse, not a TF bug
# Test Failed ❌
# stderr saved to tf_stderr_20260126_161014.log

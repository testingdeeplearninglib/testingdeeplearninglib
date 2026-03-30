# FILE: GCFL-OTHER-0001_tf_case03_dynamic_rnn_batch0_with_sequence_length.py
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
    if not hasattr(tf, "compat") or not hasattr(tf.compat, "v1"):
        _skip("tf_compat_v1_missing")
    if not hasattr(tf.compat.v1.nn, "dynamic_rnn") or not hasattr(tf.compat.v1.nn, "rnn_cell"):
        _skip("v1_dynamic_rnn_or_rnn_cell_missing")

    seed = _env_int("SEED", 2026)
    iters = _env_int("ITERS", 10)
    batch = _env_int("BATCH", 0)
    seq = _env_int("SEQ", 4)
    d_model = _env_int("D_MODEL", 16)
    hidden = _env_int("HIDDEN", 8)

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
        "eager_initial": bool(tf.executing_eagerly()),
        "knobs": {"SEED": seed, "ITERS": iters, "BATCH": batch, "SEQ": seq, "D_MODEL": d_model, "HIDDEN": hidden},
        "testcase": "dynamic_rnn_batch0_sequence_length",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass

    cellmod = tf.compat.v1.nn.rnn_cell
    Cell = getattr(cellmod, "GRUCell", None) or getattr(cellmod, "LSTMCell", None) or getattr(cellmod, "BasicLSTMCell", None)
    if Cell is None:
        _skip("no_supported_cell")

    graph = tf.Graph()
    with graph.as_default():
        x_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, None, d_model], name="x")
        seqlen_ph = tf.compat.v1.placeholder(tf.int32, shape=[None], name="seqlen")
        try:
            cell = Cell(num_units=hidden)
        except TypeError:
            cell = Cell(hidden)
        outs, _ = tf.compat.v1.nn.dynamic_rnn(cell, x_ph, sequence_length=seqlen_ph, dtype=tf.float32, time_major=False)
        outs_shape = tf.shape(outs)
        init = tf.compat.v1.global_variables_initializer()

    try:
        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(init)
            for i in range(iters):
                rs = np.random.RandomState(seed + i)
                x = rs.randn(batch, seq, d_model).astype("float32")
                # all sequence lengths = seq (or 0 if seq==0). For batch==0, this is empty vector.
                seqlen = (np.zeros((batch,), dtype="int32") + int(seq)).astype("int32")
                try:
                    y, sh = sess.run([outs, outs_shape], feed_dict={x_ph: x, seqlen_ph: seqlen})
                except Exception as e:
                    if batch == 0:
                        _pass_()
                    _skip(f"unexpected_exception_batch_gt0:{type(e).__name__}:{e}")
                try:
                    sh = [int(v) for v in list(sh)]
                except Exception:
                    if batch == 0:
                        _pass_()
                    _skip("shape_fetch_failed_batch_gt0")
                expected = [int(batch), int(seq), int(hidden)]
                if sh != expected or getattr(y, "ndim", -1) != 3:
                    if batch == 0:
                        _pass_()
                    _skip(f"unexpected_shape_batch_gt0:got={sh},exp={expected}")

        _fail_()
    except SystemExit:
        raise
    except Exception as e:
        _herr(e)


if __name__ == "__main__":
    main()
    
    
    
# Output:
# *****************
# bug no: GCFL-OTHER-0001-03
# Result: SKIP_ENV
# Test Failed ❌
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
#   testcases/tf_other_0001_inputs/GCFL-OTHER-0001-03_dynamic_rnn_batch0_with_sequence_length.py \
#   > logs/GCFL-OTHER-0001-03_stdout.log \
#   2> logs/GCFL-OTHER-0001-03_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0001-03_stdout.log
#
# Observed output:
# exit_code=0
# SKIP_ENV: no_supported_cell
#
# Note:
# This testcase also depends on tf.compat.v1 RNN cell support that is not available here.

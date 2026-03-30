# FILE: GCFL-OTHER-0001_tf_case01_dynamic_rnn_batch0.py
import os
import sys
import json
import random
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")


def _skip(reason: str):
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass():
    print("Test Passed ✅")
    sys.exit(0)


def _fail():
    print("Test Failed ❌")
    sys.exit(0)


def _harness_error(e: BaseException):
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
    sys.exit(1)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _check_env(tf, np):
    if not (sys.version_info.major == 3 and sys.version_info.minor in (10, 11)):
        _skip(f"python_not_supported:{sys.version_info.major}.{sys.version_info.minor}")
    if getattr(tf, "__version__", "") != "2.20.0":
        _skip(f"tf_version_mismatch:{getattr(tf,'__version__','unknown')}")
    if not hasattr(tf, "compat") or not hasattr(tf.compat, "v1"):
        _skip("tf_compat_v1_missing")
    if not hasattr(tf.compat.v1, "nn") or not hasattr(tf.compat.v1.nn, "dynamic_rnn"):
        _skip("tf_compat_v1_nn_dynamic_rnn_missing")
    if not hasattr(tf.compat.v1.nn, "rnn_cell"):
        _skip("tf_compat_v1_nn_rnn_cell_missing")
    cellmod = tf.compat.v1.nn.rnn_cell
    if not any(hasattr(cellmod, n) for n in ("BasicLSTMCell", "LSTMCell", "GRUCell")):
        _skip("no_v1_rnn_cell_available")


def main():
    try:
        import numpy as np
    except Exception as e:
        _skip(f"numpy_import_failed:{type(e).__name__}:{e}")
    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"tf_import_failed:{type(e).__name__}:{e}")

    _check_env(tf, np)

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
        "eager_initial": bool(tf.executing_eagerly()),
        "gpu_count": len(gpus),
        "knobs": {"SEED": seed, "ITERS": iters, "BATCH": batch, "SEQ": seq, "D_MODEL": d_model, "HIDDEN": hidden},
        "testcase": "dynamic_rnn_batch0_graph",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    # graph-mode
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass

    cellmod = tf.compat.v1.nn.rnn_cell
    Cell = None
    for n in ("BasicLSTMCell", "LSTMCell", "GRUCell"):
        if hasattr(cellmod, n):
            Cell = getattr(cellmod, n)
            break
    if Cell is None:
        _skip("no_supported_cell")

    graph = tf.Graph()
    with graph.as_default():
        x_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, None, d_model], name="x")
        try:
            cell = Cell(num_units=hidden)
        except TypeError:
            cell = Cell(hidden)
        outs, _ = tf.compat.v1.nn.dynamic_rnn(cell, x_ph, dtype=tf.float32, time_major=False)
        outs_shape = tf.shape(outs)
        init = tf.compat.v1.global_variables_initializer()

    try:
        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(init)
            for i in range(iters):
                x = np.random.RandomState(seed + i).randn(batch, seq, d_model).astype("float32")
                try:
                    y, sh = sess.run([outs, outs_shape], feed_dict={x_ph: x})
                except Exception as e:
                    if batch == 0:
                        _pass()
                    _skip(f"unexpected_exception_batch_gt0:{type(e).__name__}:{e}")

                # validate shape contract
                try:
                    sh = [int(v) for v in list(sh)]
                except Exception:
                    if batch == 0:
                        _pass()
                    _skip("shape_fetch_failed_batch_gt0")

                expected = [int(batch), int(seq), int(hidden)]
                if sh != expected or getattr(y, "ndim", -1) != 3:
                    if batch == 0:
                        _pass()
                    _skip(f"unexpected_shape_batch_gt0:got={sh},exp={expected}")

        _fail()
    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)


if __name__ == "__main__":
    main()



# Output:
# *****************
# bug no: GCFL-OTHER-0001-01
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
#   testcases/tf_other_0001_inputs/GCFL-OTHER-0001-01_dynamic_rnn_batch0.py \
#   > logs/GCFL-OTHER-0001-01_stdout.log \
#   2> logs/GCFL-OTHER-0001-01_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0001-01_stdout.log
#
# Observed output:
# exit_code=0
# SKIP_ENV: no_v1_rnn_cell_available
#
# Note:
# This environment does not expose a usable tf.compat.v1 RNN cell API for this testcase,
# so the probe did not execute its actual bug oracle.
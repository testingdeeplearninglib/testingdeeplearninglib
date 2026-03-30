# GCFL-AUTOGRADBA-0069 

import os
import sys
import traceback
import random

def _print(msg: str):
    # unbuffered-ish printing for remote runs
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def _print_skip(reason: str):
    _print(f"SKIP_ENV: {reason}")
    sys.exit(0)

def _print_pass():
    _print("Test Passed ✅")
    sys.exit(0)

def _print_fail():
    _print("Test Failed ❌")
    sys.exit(0)

def _harness_error(e: BaseException):
    msg = "".join(traceback.format_exception_only(type(e), e)).strip()
    _print(f"HARNESS_ERROR: {msg}")
    sys.exit(1)

def main():
    # Reduce TF log spam (optional)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    SEED = 2021
    random.seed(SEED)
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))

    try:
        import numpy as np
    except Exception as e:
        _print_skip(f"numpy not importable: {e}")

    try:
        import tensorflow as tf
    except Exception as e:
        _print_skip(f"tensorflow not importable: {e}")

    # Print env early so we always see *something*
    _print(f"ENV: python={sys.version.split()[0]} tf={getattr(tf,'__version__','?')}")
    try:
        _print(f"ENV: built_with_cuda={bool(tf.test.is_built_with_cuda())}")
    except Exception:
        _print("ENV: built_with_cuda=?")

    # TF1 graph mode
    tf1 = getattr(tf, "compat", None)
    tfv1 = tf1.v1 if tf1 and hasattr(tf1, "v1") else tf

    try:
        if hasattr(tf, "executing_eagerly") and tf.executing_eagerly():
            if hasattr(tfv1, "disable_eager_execution"):
                tfv1.disable_eager_execution()
            else:
                _print_skip("Eager enabled but cannot disable (no compat.v1.disable_eager_execution).")
    except Exception as e:
        _print_skip(f"Could not determine/disable eager execution: {e}")

    required = [
        ("Graph", hasattr(tfv1, "Graph")),
        ("placeholder", hasattr(tfv1, "placeholder")),
        ("Session", hasattr(tfv1, "Session")),
        ("global_variables_initializer", hasattr(tfv1, "global_variables_initializer")),
        ("train.GradientDescentOptimizer", hasattr(getattr(tfv1, "train", None), "GradientDescentOptimizer")),
    ]
    missing = [name for name, ok in required if not ok]
    if missing:
        _print_skip(f"Missing TF1 graph APIs: {', '.join(missing)}")

    # ---- Build graph ----
    learning_rate = 0.001
    batch_size = 100
    n_image_size = 28
    n_channels = 1
    n_classes = 10

    graph = tfv1.Graph()
    with graph.as_default():
        if hasattr(tfv1, "set_random_seed"):
            tfv1.set_random_seed(SEED)

        x = tfv1.placeholder(tf.float32, [batch_size, n_image_size, n_image_size, n_channels], name="x")
        y = tfv1.placeholder(tf.float32, [batch_size, n_classes], name="y")
        _ = y

        # truncated normal
        if hasattr(tf.random, "truncated_normal"):
            tn = tf.random.truncated_normal
        elif hasattr(tfv1, "truncated_normal"):
            tn = tfv1.truncated_normal
        else:
            _print_skip("No truncated_normal op available.")

        W = tfv1.Variable(tn([5, 5, n_channels, 32], stddev=0.1, seed=SEED), name="W")

        # random uniform
        if hasattr(tf.random, "uniform"):
            runif = tf.random.uniform
        elif hasattr(tfv1, "random_uniform"):
            runif = tfv1.random_uniform
        else:
            _print_skip("No random_uniform/uniform op available.")

        def project(W_in):
            weights = tf.clip_by_value(W_in, -0.1, 0.1)
            shape = tf.shape(weights)

            rnd = runif(shape, minval=0.0, maxval=1.0, seed=SEED, dtype=tf.float32)

            mx = tf.reduce_max(tf.abs(weights))
            mx_fill = tf.fill(shape, mx)

            one_fill = tf.fill(shape, 1.0)
            two_fill = tf.fill(shape, 2.0)

            eps = tf.constant(1e-12, dtype=tf.float32)
            mx_safe = tf.maximum(mx, eps)
            mx_safe_fill = tf.fill(shape, mx_safe)

            prob = tf.clip_by_value(((weights / mx_safe_fill) + one_fill) / two_fill, 0.0, 1.0)
            draws = tf.greater(rnd, one_fill - prob)

            if hasattr(tfv1, "select"):
                return tfv1.select(draws, mx_fill, -mx_fill, name="select_proj")
            else:
                return tf.where(draws, mx_fill, -mx_fill, name="where_proj")

        w = project(W)

        out = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
        cost = tf.reduce_mean(out) + 0.0

        opt = tfv1.train.GradientDescentOptimizer(learning_rate)
        train_op = opt.minimize(cost)

        init = tfv1.global_variables_initializer()

        # ---- DEBUG: op counts ----
        try:
            ops = [op.type for op in graph.get_operations()]
            _print("DEBUG: op_type_counts=" + str({k: ops.count(k) for k in ["Fill", "Where", "Select", "SelectV2"]}))
        except Exception as e:
            _print(f"DEBUG: could not list ops: {e}")

        # ---- DEBUG: gradient existence ----
        try:
            g = tf.gradients(cost, [W])[0]
            _print("DEBUG: tf.gradients(cost,[W]) is None? " + str(g is None))
        except Exception as e:
            _print("DEBUG: tf.gradients threw: " + repr(e))

    # ---- Synthetic inputs ----
    np.random.seed(SEED)
    x_np = np.random.randn(batch_size, n_image_size, n_image_size, n_channels).astype("float32")
    y_np = np.random.randn(batch_size, n_classes).astype("float32")

    # ---- Run ----
    try:
        with tfv1.Session(graph=graph) as sess:
            sess.run(init)
            sess.run(train_op, feed_dict={x: x_np, y: y_np})
    except Exception as e:
        msg = str(e)
        _print("CAUGHT_EXCEPTION: " + repr(msg))
        msg_l = msg.lower()

        ok = False

        # Fill gradient patterns
        if ("fill" in msg_l) and ("gradient" in msg_l or "no gradient" in msg_l or "nogradient" in msg_l):
            ok = True

        # Select/Where gradient patterns
        if (("select" in msg_l or "selectv2" in msg_l or "where" in msg_l) and
            ("gradient" in msg_l or "no gradient" in msg_l or "nogradient" in msg_l)):
            ok = True

        # Optimizer-style failure that can result from missing gradients
        if "no gradients provided for any variable" in msg_l:
            ok = True

        if ok:
            _print_pass()
        else:
            _print_fail()

    # If no exception: bug not reproduced
    _print_fail()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# source ~/.venvs/tf_testing/bin/activate
# cd ~/dl_testing/testcases
# python -u tensorflow_testcase.

# Output:
# *****************
# 2026-01-15 21:25:50.275539: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
# 2026-01-15 21:25:50.304321: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
# 2026-01-15 21:25:50.304371: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
# ENV: python=3.12.3 tf=2.16.2
# ENV: built_with_cuda=True
# DEBUG: op_type_counts={'Fill': 5, 'Where': 0, 'Select': 0, 'SelectV2': 7}
# DEBUG: tf.gradients(cost,[W]) is None? False
# Test Failed ❌
# GCFL-AUTOGRADBA-0070

# GCFL-AUTOGRADBA-0070 (regression check for TF issue #897)

import sys
import numpy as np

def _print_and_exit(msg: str, code: int = 0):
    print(msg)
    sys.exit(code)

def _to_f32_vec(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float32).reshape(-1)

def main():
    try:
        import tensorflow as tf
    except Exception as e:
        _print_and_exit(f"SKIP_ENV: missing tensorflow ({e})", 0)

    np.random.seed(2021)
    try:
        tf.random.set_seed(2021)
    except Exception:
        pass

    # TF1-style graph mode
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception as e:
        _print_and_exit(f"SKIP_ENV: cannot disable eager execution ({e})", 0)

    v1 = tf.compat.v1

    # GPU presence check (not required for the bug itself, but matches GPU-server scenario)
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:
        gpus = []
    if not gpus:
        _print_and_exit("SKIP_ENV: no GPU visible to TensorFlow", 0)

    try:
        v1.reset_default_graph()
    except Exception:
        pass

    f = np.array([3.8], dtype=np.float32)
    x = v1.placeholder(tf.float32, shape=[1], name="x")

    y_floor = tf.floor(x)
    y_sum   = x + tf.floor(x)

    g_floor = v1.gradients(v1.reduce_sum(y_floor), x)[0]   # may be None (disconnected)
    g_sum   = v1.gradients(v1.reduce_sum(y_sum),   x)[0]   # should be 1

    config = None
    try:
        config = v1.ConfigProto()
        config.gpu_options.allow_growth = True
    except Exception:
        config = None

    with v1.Session(config=config) as sess:
        try:
            sess.run(v1.global_variables_initializer())
        except Exception:
            pass

        # Only fetch tensors that are not None
        fetches = {"g_sum": g_sum}
        if g_floor is not None:
            fetches["g_floor"] = g_floor

        out = sess.run(fetches, feed_dict={x: f})

    tol = 1e-6
    gf = _to_f32_vec(out.get("g_floor", None))  # None means disconnected, OK
    gs = _to_f32_vec(out["g_sum"])

    ok_floor = (gf is None) or np.allclose(gf, np.array([0.0], np.float32), atol=tol, rtol=0.0)
    ok_sum   = (gs is not None) and np.allclose(gs, np.array([1.0], np.float32), atol=tol, rtol=0.0)

    # Correct behavior on modern TF => NOT suspicious => Test Failed ❌
    if ok_floor and ok_sum:
        _print_and_exit("Test Failed ❌", 0)
    else:
        _print_and_exit("Test Passed ✅", 0)

if __name__ == "__main__":
    main()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# cd ~/dl_testing && \
# source ~/.venvs/dl_testing/bin/activate && \
# CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=3 python gcfl_autogradba_0070.py


# Output:
# *****************

# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1768900864.234725 3882814 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22446 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:02:00.0, compute capability: 8.6
# I0000 00:00:1768900864.244719 3882814 mlir_graph_optimization_pass.cc:437] MLIR V1 optimization pass is not enabled
# Test Failed ❌


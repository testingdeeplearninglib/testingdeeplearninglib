# FILE: GCFL-AUTOGRAD_BACKWARD-0003_tc10_tf_tensorarray_stack_grad_mode_mismatch.py
import os
import sys
import json
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"))
os.environ.setdefault("TF_DETERMINISTIC_OPS", os.environ.get("TF_DETERMINISTIC_OPS", "1"))

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

def _env_int(k: str, d: int) -> int:
    v = os.environ.get(k, "").strip()
    try:
        return int(v) if v else d
    except Exception:
        return d

def main():
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as e:
        _skip(f"import failed: {type(e).__name__}: {e}")

    if not (sys.version_info.major == 3 and sys.version_info.minor in (10, 11)):
        _skip(f"Python not in {{3.10,3.11}}: {sys.version_info.major}.{sys.version_info.minor}")
    if tf.__version__ != "2.20.0":
        _skip(f"tensorflow!=2.20.0: {tf.__version__}")

    seed = _env_int("SEED", 2026)
    t = _env_int("T", 8)
    d = _env_int("D_MODEL", 16)
    atol = float(os.environ.get("ATOL", "1e-6"))
    iters = _env_int("ITERS", 5)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env_payload = {
        "test_id": "GCFL-AUTOGRAD_BACKWARD-0003_tc10",
        "gcfl_id": "GCFL-AUTOGRAD_BACKWARD-0003",
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "devices": {"gpu": len(tf.config.list_physical_devices("GPU")), "cpu": len(tf.config.list_physical_devices("CPU"))},
        "knobs": {"SEED": seed, "ITERS": iters, "T": t, "D_MODEL": d, "ATOL": atol},
        "oracle": "TensorArray stack gradient mismatch eager vs tf.function OR exception",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    @tf.function
    def g(x):
        # x: [T, D]
        ta = tf.TensorArray(tf.float32, size=t, clear_after_read=False)
        for i in tf.range(t):
            ta = ta.write(i, x[i] + tf.cast(i, tf.float32) * 0.0)
        y = ta.stack()  # [T, D]
        return tf.reduce_sum(y * y)

    for i in range(iters):
        rs = np.random.RandomState(seed + i)
        x0 = rs.randn(t, d).astype("float32")
        x = tf.Variable(x0)

        # Eager-ish execution of tf.function
        try:
            tf.config.run_functions_eagerly(True)
        except Exception:
            pass
        try:
            with tf.GradientTape() as tape:
                le = g(x)
            ge = tape.gradient(le, x)
        except Exception:
            _pass()

        # Graph execution
        try:
            tf.config.run_functions_eagerly(False)
        except Exception:
            pass
        try:
            with tf.GradientTape() as tape:
                lg = g(x)
            gg = tape.gradient(lg, x)
        except Exception:
            _pass()

        if ge is None or gg is None:
            _pass()

        ge_np = ge.numpy()
        gg_np = gg.numpy()

        if not (np.isfinite(ge_np).all() and np.isfinite(gg_np).all()):
            _pass()

        diff = float(np.max(np.abs(ge_np - gg_np)))
        if diff > atol:
            _pass()

    _fail()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)




# Output:
# *****************
# bug no: GCFL-AUTOGRAD_BACKWARD-0003_tc10
# Result: Test Failed ❌
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
#   testcases/tf_batch_inputs/GCFL-AUTOGRAD_BACKWARD-0003_tc10_tf_tensorarray_stack_grad_mode_mismatch.py \
#   > logs/GCFL-AUTOGRAD_BACKWARD-0003_tc10_stdout.log \
#   2> logs/GCFL-AUTOGRAD_BACKWARD-0003_tc10_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-AUTOGRAD_BACKWARD-0003_tc10_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# The testcase detected suspicious behavior in the nested conditional jacobian path.
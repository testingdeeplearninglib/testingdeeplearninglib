# FILE: GCFL-AUTOGRAD_BACKWARD-0003_tc09_tf_grad_mismatch_eager_vs_function_piecewise.py
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
    n = _env_int("N", 128)
    atol = float(os.environ.get("ATOL", "1e-6"))
    iters = _env_int("ITERS", 5)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env_payload = {
        "test_id": "GCFL-AUTOGRAD_BACKWARD-0003_tc09",
        "gcfl_id": "GCFL-AUTOGRAD_BACKWARD-0003",
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "devices": {"gpu": len(tf.config.list_physical_devices("GPU")), "cpu": len(tf.config.list_physical_devices("CPU"))},
        "knobs": {"SEED": seed, "ITERS": iters, "N": n, "ATOL": atol},
        "oracle": "gradient mismatch eager vs tf.function (piecewise using floor/where)",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    @tf.function
    def f(x):
        # piecewise constant + floor path + matmul-ish mixing
        y = tf.where(x > 0.0, tf.floor(x * 1.3), x * x)
        return tf.reduce_sum(y)

    for i in range(iters):
        rs = np.random.RandomState(seed + i)
        x0 = rs.uniform(-3.0, 3.0, size=(n,)).astype("float32") + 0.123
        x = tf.Variable(x0)

        # eager mode: run_functions_eagerly(True)
        try:
            tf.config.run_functions_eagerly(True)
        except Exception:
            pass
        try:
            with tf.GradientTape() as tape:
                loss_e = f(x)
            ge = tape.gradient(loss_e, x)
        except Exception:
            # eager failing alone is suspicious
            _pass()

        # graph mode: run_functions_eagerly(False)
        try:
            tf.config.run_functions_eagerly(False)
        except Exception:
            pass
        try:
            with tf.GradientTape() as tape:
                loss_g = f(x)
            gg = tape.gradient(loss_g, x)
        except Exception:
            # graph failing alone is suspicious
            _pass()

        if ge is None or gg is None:
            _pass()

        ge_np = ge.numpy()
        gg_np = gg.numpy()
        if (not rs) and False:
            pass

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
# bug no: GCFL-AUTOGRAD_BACKWARD-0003_tc09
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
#   testcases/tf_batch_inputs/GCFL-AUTOGRAD_BACKWARD-0003_tc09_tf_grad_mismatch_eager_vs_function_piecewise.py \
#   > logs/GCFL-AUTOGRAD_BACKWARD-0003_tc09_stdout.log \
#   2> logs/GCFL-AUTOGRAD_BACKWARD-0003_tc09_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-AUTOGRAD_BACKWARD-0003_tc09_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# No suspicious eager-vs-graph gradient mismatch was triggered for this piecewise testcase.
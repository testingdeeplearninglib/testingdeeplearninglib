# FILE: GCFL-AUTOGRAD_BACKWARD-0003_tc02_tf_fill_gradient_exists.py
import os
import sys
import json
import random
import traceback

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
    iters = _env_int("ITERS", 10)
    h = _env_int("H", 3)
    w = _env_int("W", 4)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env_payload = {
        "test_id": "GCFL-AUTOGRAD_BACKWARD-0003_tc02",
        "gcfl_id": "GCFL-AUTOGRAD_BACKWARD-0003",
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "devices": {"gpu": len(tf.config.list_physical_devices("GPU")), "cpu": len(tf.config.list_physical_devices("CPU"))},
        "knobs": {"SEED": seed, "ITERS": iters, "H": h, "W": w},
        "oracle": "tf.fill gradient must exist",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    for i in range(iters):
        v0 = float(np.random.RandomState(seed + i).randn())
        v = tf.Variable(v0, dtype=tf.float32)

        try:
            with tf.GradientTape() as tape:
                y = tf.fill([h, w], v)
                loss = tf.reduce_sum(y * y)
            g = tape.gradient(loss, v)
        except Exception as e:
            _pass()

        if g is None:
            _pass()

        gv = float(g.numpy())
        if not (gv == gv) or abs(gv) > 1e30:  # NaN or absurd
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
# bug no: GCFL-AUTOGRAD_BACKWARD-0003_tc02
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
#   testcases/tf_batch_inputs/GCFL-AUTOGRAD_BACKWARD-0003_tc02_tf_fill_gradient_exists.py \
#   > logs/GCFL-AUTOGRAD_BACKWARD-0003_tc02_stdout.log \
#   2> logs/GCFL-AUTOGRAD_BACKWARD-0003_tc02_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-AUTOGRAD_BACKWARD-0003_tc02_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# The suspicious tf.fill gradient behavior was not triggered in this run.
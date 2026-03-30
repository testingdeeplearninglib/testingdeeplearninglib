# FILE: GCFL-AUTOGRAD_BACKWARD-0003_tc04_tf_while_loop_concat_grad_exception.py
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
    iters = _env_int("ITERS", 5)
    steps = _env_int("STEPS", 8)
    d = _env_int("D_MODEL", 16)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env_payload = {
        "test_id": "GCFL-AUTOGRAD_BACKWARD-0003_tc04",
        "gcfl_id": "GCFL-AUTOGRAD_BACKWARD-0003",
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "devices": {"gpu": len(tf.config.list_physical_devices("GPU")), "cpu": len(tf.config.list_physical_devices("CPU"))},
        "knobs": {"SEED": seed, "ITERS": iters, "STEPS": steps, "D_MODEL": d},
        "oracle": "exception in gradient through while_loop concat body (graph)",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    @tf.function
    def f(x):
        # x: [T, D]
        ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        i0 = tf.constant(0)

        def cond(i, ta_):
            return i < tf.shape(x)[0]

        def body(i, ta_):
            v = x[i]  # [D]
            # concat pattern in the loop (historical failure patterns)
            v2 = tf.concat([v[: d // 2], v[d // 2 :]], axis=0)
            ta_ = ta_.write(i, v2)
            return i + 1, ta_

        _, ta2 = tf.while_loop(cond, body, [i0, ta])
        y = ta2.stack()  # [T, D]
        return tf.reduce_sum(y * y)

    for i in range(iters):
        rs = np.random.RandomState(seed + i)
        x0 = rs.randn(steps, d).astype("float32")
        x = tf.Variable(x0)

        try:
            with tf.GradientTape() as tape:
                loss = f(x)
            g = tape.gradient(loss, x)
        except Exception:
            _pass()

        if g is None:
            _pass()
        gv = g.numpy()
        if not np.isfinite(gv).all():
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
# bug no: GCFL-AUTOGRAD_BACKWARD-0003_tc04
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
#   testcases/tf_batch_inputs/GCFL-AUTOGRAD_BACKWARD-0003_tc04_tf_while_loop_concat_grad_exception.py \
#   > logs/GCFL-AUTOGRAD_BACKWARD-0003_tc04_stdout.log \
#   2> logs/GCFL-AUTOGRAD_BACKWARD-0003_tc04_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-AUTOGRAD_BACKWARD-0003_tc04_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# No exception or suspicious gradient failure was triggered in the while_loop/concat path.
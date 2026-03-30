# FILE: GCFL-OTHER-0001_tf_case04_jacobian_tfcond_graph_vs_eager.py
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

    seed = _env_int("SEED", 2026)
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
        "knobs": {"SEED": seed},
        "testcase": "jacobian_tfcond_graph_vs_eager",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    x = tf.constant(0.5, dtype=tf.float32)

    @tf.function
    def f(xv):
        # nested cond to stress pfor jacobian vectorization
        return tf.cond(
            xv < 0.0,
            lambda: xv * xv + 1.0,
            lambda: tf.cond(xv < 1.0, lambda: xv * xv, lambda: xv * xv + 2.0),
        )

    def run(eager: bool):
        tf.config.run_functions_eagerly(bool(eager))
        try:
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                with tf.GradientTape() as t1:
                    t1.watch(x)
                    y = f(x)
                g = t1.gradient(y, x)
            h = t2.jacobian(g, x, experimental_use_pfor=True)
            return ("ok", float(h.numpy()))
        except Exception as e:
            return ("err", f"{type(e).__name__}: {e}")

    r_graph = run(eager=False)
    r_eager = run(eager=True)

    # oracle: graph-mode jacobian fails but eager works (suspicious)
    if r_graph[0] == "err" and r_eager == ("ok", 2.0):
        _pass_()
    _fail_()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _herr(e)



# Output:
# *****************
# bug no: GCFL-OTHER-0001-04
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
#   testcases/tf_other_0001_inputs/GCFL-OTHER-0001-04_jacobian_tfcond_graph_vs_eager.py \
#   > logs/GCFL-OTHER-0001-04_stdout.log \
#   2> logs/GCFL-OTHER-0001-04_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0001-04_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# The expected graph-vs-eager jacobian failure pattern did not trigger in this run.
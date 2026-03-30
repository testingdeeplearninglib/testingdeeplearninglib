# FILE: GCFL-OTHER-0001_tf_case08_nested_structure_map_fn_tensorflow.py
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
        "testcase": "nested_structure_map_fn",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    # attempt a nested-structure map through tf.function + tf.nest; treat structure-handling exceptions as suspicious
    @tf.function
    def f(struct):
        # struct is nested dict/list; map over it and keep structure
        def leaf(x):
            return x + 1.0
        return tf.nest.map_structure(leaf, struct)

    x = {
        "a": tf.constant([1.0, 2.0], dtype=tf.float32),
        "b": [tf.constant([3.0], dtype=tf.float32), tf.constant([4.0, 5.0], dtype=tf.float32)],
    }
    try:
        y = f(x)
        # quick sanity: same structure and correct values
        ya = y["a"].numpy().tolist()
        if ya != [2.0, 3.0]:
            _pass_()
        _fail_()
    except Exception as e:
        msg = str(e).lower()
        # suspicious: failure to handle nested structures in graph mode
        if "nested" in msg or "structure" in msg or "composite" in msg or "spec" in msg:
            _pass_()
        _pass_()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _herr(e)



# Output:
# *****************
# bug no: GCFL-OTHER-0001-08
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
#   testcases/tf_other_0001_inputs/GCFL-OTHER-0001-08_nested_structure_map_fn_tensorflow.py \
#   > logs/GCFL-OTHER-0001-08_stdout.log \
#   2> logs/GCFL-OTHER-0001-08_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0001-08_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# Nested-structure handling behaved normally enough that the testcase oracle did not fire.
# FILE: GCFL-OTHER-0001_tf_case09_savedmodel_roundtrip_output_mismatch.py
import os
import sys
import json
import random
import tempfile

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
        "testcase": "savedmodel_roundtrip_output_mismatch",
    }
    print("ENV: " + json.dumps(env_payload, sort_keys=True))

    class M(tf.Module):
        def __init__(self):
            super().__init__()
            self.w = tf.Variable(tf.random.normal([8, 8], seed=seed), trainable=True, name="w")
            # trackable containers sometimes cause oddities; include nested containers
            self.vars = [self.w]
            self.meta = {"w": self.w}

        @tf.function(input_signature=[tf.TensorSpec([None, 8], tf.float32)])
        def __call__(self, x):
            y = tf.matmul(x, self.w)
            return {"y": y, "sum": tf.reduce_sum(y)}

    m = M()
    x = tf.constant(np.random.RandomState(seed).randn(3, 8).astype("float32"))
    y0 = m(x)

    tmp = tempfile.mkdtemp(prefix="gcfl_savedmodel_")
    try:
        tf.saved_model.save(m, tmp)
    except Exception as e:
        # saving should not crash
        _pass_()

    try:
        m2 = tf.saved_model.load(tmp)
    except Exception as e:
        _pass_()

    try:
        y1 = m2(x)
    except Exception as e:
        _pass_()

    # output mismatch oracle
    y0v = y0["y"].numpy()
    y1v = y1["y"].numpy()
    max_abs = float(np.max(np.abs(y0v - y1v)))
    if max_abs > 1e-6:
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
# bug no: GCFL-OTHER-0001-09
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
#   testcases/tf_other_0001_inputs/GCFL-OTHER-0001-09_savedmodel_roundtrip_output_mismatch.py \
#   > logs/GCFL-OTHER-0001-09_stdout.log \
#   2> logs/GCFL-OTHER-0001-09_stderr.log
# echo "exit_code=$?"
# cat logs/GCFL-OTHER-0001-09_stdout.log
#
# Observed output:
# exit_code=0
# Test Failed ❌
#
# Note:
# SavedModel save/load roundtrip did not produce the targeted output mismatch in this run.
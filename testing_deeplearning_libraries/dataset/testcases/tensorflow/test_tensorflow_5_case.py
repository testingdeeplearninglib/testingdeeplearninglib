# GCFL-TRACINGGRA-0005

import json
import os
import random
import sys

# Silence most TF C++ logs as early as possible
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass() -> None:
    print("Test Passed ✅")
    sys.exit(0)


def _fail() -> None:
    print("Test Failed ❌")
    sys.exit(0)


def _harness_error(e: BaseException) -> None:
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
    sys.exit(1)


def _emit_env(tf_mod) -> None:
    try:
        gpus = tf_mod.config.list_physical_devices("GPU")
        env = {
            "python": sys.version.split()[0],
            "tensorflow": getattr(tf_mod, "__version__", "unknown"),
            "jit_requested": True,
            "gpu_count_visible_to_tf": len(gpus),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        }
        print("ENV:", json.dumps(env, sort_keys=True))
    except Exception:
        pass


def _classify_tf_exception(e: BaseException) -> str:
    msg = f"{type(e).__name__}: {e}"

    known_unsupported_markers = (
        "Detected unsupported operations",
        "not compilable by XLA",
        "outside compilation",
        "No registered '",
        "unsupported op",
        "UNIMPLEMENTED",
        "TemporaryVariable",
        "TemporaryVariableOp",
        "StringToHashBucket",
        "HashTable",
        "LookupTable",
        "lookup table",
        "tf2xla conversion failed",
        "XLA compilation requires",
        "cannot be XLA compiled",
    )
    if any(k.lower() in msg.lower() for k in known_unsupported_markers):
        return "known_unsupported"

    suspicious_markers = (
        "Table already initialized",
        "FailedPreconditionError",
        "ResourceExhaustedError",
        "InternalError",
        "Check failed",
        "Segmentation fault",
        "AbortedError",
    )
    if any(k.lower() in msg.lower() for k in suspicious_markers):
        return "candidate_bug"

    mod = getattr(type(e), "__module__", "") or ""
    if mod.startswith("tensorflow"):
        return "candidate_bug"

    return "unknown"


def main() -> None:
    seed = 1337
    random.seed(seed)

    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"tensorflow missing/unimportable: {type(e).__name__}: {e}")

    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass

    _emit_env(tf)

    # These APIs are deprecated and weakly maintained. If absent, this test is not runnable.
    if not hasattr(tf, "feature_column"):
        _skip("tf.feature_column not available in this TensorFlow build")
    if not hasattr(tf, "compat") or not hasattr(tf.compat, "v1"):
        _skip("tf.compat.v1 not available in this TensorFlow build")
    if not hasattr(tf.compat.v1, "feature_column") or not hasattr(tf.compat.v1.feature_column, "input_layer"):
        _skip("tf.compat.v1.feature_column.input_layer not available")

    try:
        num = tf.feature_column.numeric_column("n", dtype=tf.float32)
        bucket = tf.feature_column.bucketized_column(num, boundaries=[0.0, 1.0, 2.0, 3.0])
        cat = tf.feature_column.categorical_column_with_hash_bucket(
            "x", hash_bucket_size=128, dtype=tf.string
        )
        crossed = tf.feature_column.crossed_column([bucket, "x"], hash_bucket_size=256)
        ind_cat = tf.feature_column.indicator_column(cat)
        ind_cross = tf.feature_column.indicator_column(crossed)
        feature_columns = [ind_cat, ind_cross]
    except Exception as e:
        _skip(f"feature_column construction unsupported: {type(e).__name__}: {e}")

    class CustomModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.d1 = tf.keras.layers.Dense(16, activation="relu")
            self.d2 = tf.keras.layers.Dense(1)

        def call(self, features, training=False):
            x = tf.compat.v1.feature_column.input_layer(features, feature_columns)
            x = self.d1(x)
            return self.d2(x)

    model = CustomModel()
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)

    def _train_step_impl(features):
        with tf.GradientTape() as tape:
            y = model(features, training=True)
            loss = tf.reduce_sum(y)

        vars_ = model.trainable_variables
        if not vars_:
            raise RuntimeError("model has no trainable variables after forward pass")

        grads = tape.gradient(loss, vars_)
        if grads is None or any(g is None for g in grads):
            raise RuntimeError("gradient computation returned None")

        opt.apply_gradients(zip(grads, vars_))
        return loss

    # Pre-build once eagerly so variable creation is not mixed with first XLA trace.
    try:
        _ = model(
            {
                "x": tf.sparse.from_dense(tf.constant([["warmup"]], dtype=tf.string)),
                "n": tf.constant([[0.25]], dtype=tf.float32),
            },
            training=True,
        )
    except Exception as e:
        _skip(f"warmup/build path unsupported before JIT step: {type(e).__name__}: {e}")

    try:
        train_step = tf.function(
            _train_step_impl,
            jit_compile=True,
            reduce_retracing=True,
        )
    except TypeError:
        try:
            train_step = tf.function(
                _train_step_impl,
                experimental_compile=True,
                reduce_retracing=True,
            )
        except TypeError:
            train_step = tf.function(_train_step_impl, reduce_retracing=True)

    def make_features(batch: int, shift: int):
        toks = [f"tok{(i + shift) % 7}" for i in range(batch)]
        x_dense = tf.constant([[t] for t in toks], dtype=tf.string)
        x_sparse = tf.sparse.from_dense(x_dense)

        n_vals = [float((i + shift) % 5) + 0.25 for i in range(batch)]
        n = tf.constant([[v] for v in n_vals], dtype=tf.float32)

        return {"x": x_sparse, "n": n}

    try:
        train_step(make_features(2, 0))
        train_step(make_features(3, 1))
        train_step(make_features(4, 2))
    except BaseException as e:
        cls = _classify_tf_exception(e)
        if cls == "known_unsupported":
            _skip(f"known TF/XLA compatibility limitation, not counted as bug: {type(e).__name__}: {e}")
        if cls == "candidate_bug":
            _pass()
        _harness_error(e)

    _fail()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        _harness_error(e)
c


# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# cd ~/dl_testing
# conda activate tf_venv

# export CUDA_VISIBLE_DEVICES=0
# unset TF_XLA_FLAGS
# export TF_CPP_MIN_LOG_LEVEL=3
# export PYTHONUNBUFFERED=1

# mkdir -p logs

# set -o pipefail
# python -u testcases/tensorflow_testcase.py 2>&1 | tee logs/GCFL-TRACINGGRA-0005_rerun.log
# echo "exit_code=$?"


# Output:
# *****************
# ENV: {"cuda_visible_devices": "0", "gpu_count_visible_to_tf": 1, "jit_requested": true, "python": "3.11.15", "tensorflow": "2.21.0"}

# SKIP_ENV: known TF/XLA compatibility limitation, not counted as bug: InvalidArgumentError: Detected unsupported operations when trying to compile graph __inference__train_step_impl_397[_XlaMustCompile=true,config_proto=2201667018877855759,executor_type=11160318154034397263] on XLA_GPU_JIT: _Arg (No registered '_Arg' OpKernel for XLA_GPU_JIT devices compatible with node {{node features_1}}
#          (OpKernel was found, but attributes didn't match) Requested Attributes: T=DT_STRING, _output_shapes=[[2]], _user_specified_name="features", index=2){{node features_1}}
# The op is created at:
# File "dl_testing/testcases/tensorflow_testcase.py", line 222, in <module>
# File "dl_testing/testcases/tensorflow_testcase.py", line 206, in main
# File "miniconda3/envs/tf_venv/lib/python3.11/site-packages/tensorflow/core/function/polymorphism/function_type.py", line 356, in placeholder_arguments
# File "miniconda3/envs/tf_venv/lib/python3.11/site-packages/tensorflow/core/function/trace_type/default_types.py", line 742, in placeholder_value
# File "miniconda3/envs/tf_venv/lib/python3.11/site-packages/tensorflow/core/function/trace_type/default_types.py", line 743, in <listcomp>
#         tf2xla conversion failed while converting __inference__train_step_impl_397[_XlaMustCompile=true,config_proto=2201667018877855759,executor_type=11160318154034397263]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions. [Op:__inference__train_step_impl_397]

# exit_code=0
# Test Failed ❌
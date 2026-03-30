# GCFL-OTHER-0088

import os
import sys
import traceback


def _skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass() -> None:
    print("Test Passed ✅")
    sys.exit(0)


def _fail() -> None:
    print("Test Failed ❌")
    sys.exit(0)


def _harness_error(e: Exception) -> None:
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
    sys.exit(1)


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    try:
        import numpy as np
    except Exception as e:
        _skip(f"missing numpy ({e})")

    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"missing tensorflow ({e})")

    np.random.seed(2021)
    try:
        tf.keras.utils.set_random_seed(2021)
    except Exception:
        try:
            tf.random.set_seed(2021)
        except Exception:
            pass

    print(f"TF_VERSION: {getattr(tf, '__version__', 'unknown')}")
    try:
        print(f"GPU_DEVICES: {tf.config.list_physical_devices('GPU')}")
    except Exception:
        pass

    inputs = tf.keras.Input(shape=(4, 4, 1))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    def bad_metric(y_true, y_pred):
        del y_true

        def _py_bad(t):
            return t.swapaxes(0, 1)  # intentionally wrong for EagerTensor

        v = tf.py_function(func=_py_bad, inp=[y_pred], Tout=tf.float32)
        return tf.reduce_mean(v)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="mse",
        metrics=[bad_metric],
        run_eagerly=False,
    )

    x_data = np.random.rand(2, 4, 4, 1).astype(np.float32)
    y_data = np.random.rand(2, 1).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(1)

    try:
        model.fit(ds, epochs=1, steps_per_epoch=1, verbose=0)
    except Exception as e:
        tb = traceback.format_exc().lower()
        msg = (str(e) or "").lower()
        combined = tb + "\n" + msg

        if "swapaxes" in combined and (
            "eagertensor" in combined
            or "tensorflow.python.framework.ops.eagertensor" in combined
            or "has no attribute" in combined
        ):
            _pass()

        _fail()

    _fail()


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
# cd ~/dl_testing
# conda activate tf_venv

# export CUDA_VISIBLE_DEVICES=0
# export PYTHONUNBUFFERED=1
# export TF_CPP_MIN_LOG_LEVEL=1

# set -o pipefail
# python testcase/tensorflow_testcase.py 2>&1 | tee logs/GCFL-OTHER-0088.log
# echo "exit_code=$?"


# Output:
# *****************
# TF_VERSION: 2.21.0
# GPU_DEVICES: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# W0000 00:00:1773941672.452063 1046065 op_kernel.cc:1845] UNKNOWN: AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'swapaxes'
# Traceback (most recent call last):

#   File "/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
#     return func(device, token, args)

#   File "/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
#     outputs = self._call(device, args)

#   File "/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
#     ret = self._func(*args)

#   File "/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
#     return func(*args, **kwargs)

#   File "/home/talha/dl_testing/testcase/tensorflow_testcase.py", line 65, in _py_bad
#     return t.swapaxes(0, 1)

#   File "/home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/tensorflow/python/framework/tensor.py", line 260, in __getattr__
#     self.__getattribute__(name)

# AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'swapaxes'

# [[{{function_node __inference_one_step_on_data_589}}{{node EagerPyFunc}}]]
# # Test Failed ❌
# exit_code=0
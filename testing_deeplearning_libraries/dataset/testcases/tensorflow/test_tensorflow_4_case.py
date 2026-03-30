# bug no: keras-team/keras#19855
import os
import sys
import json
import random
import inspect

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

def _seed(seed=1337):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

def _final(msg, code):
    print(msg)
    sys.exit(code)

try:
    _seed(1337)

    import numpy as np
    import tensorflow as tf
    import keras

    try:
        tf.random.set_seed(1337)
    except Exception:
        pass

    print("ENV: " + json.dumps({
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "visible_gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }, sort_keys=True))

    def my_loss(y_true, y_pred):
        y1, y2 = y_true
        pred_sum = keras.ops.sum(y_pred)
        return keras.ops.abs(keras.ops.sum(y1) - pred_sum) + keras.ops.abs(
            keras.ops.sum(y2) - pred_sum
        )

    input1 = keras.Input((2,), name="input_1")
    input2 = keras.Input((3, 6), name="input_2")

    x1 = keras.ops.expand_dims(keras.layers.Dense(10)(input1), 1)
    x2 = keras.layers.Dense(10)(input2)
    x = keras.ops.sum(x1 + x2, axis=1)
    out = keras.layers.Dense(8)(x)

    model = keras.Model([input1, input2], out)

    try:
        model.compile(loss=my_loss, optimizer="sgd", jit_compile=False, run_eagerly=True)
    except TypeError:
        try:
            model.compile(loss=my_loss, optimizer="sgd", run_eagerly=True)
        except TypeError:
            model.compile(loss=my_loss, optimizer="sgd")

    a = np.random.rand(10, 2).astype("float32")
    b = np.random.rand(10, 3, 6).astype("float32")
    y_pred = model((a, b), training=False)

    sig = inspect.signature(model.compute_loss)
    kwargs = {}
    if "x" in sig.parameters:
        kwargs["x"] = None
    if "y" in sig.parameters:
        kwargs["y"] = (a, b)
    if "y_pred" in sig.parameters:
        kwargs["y_pred"] = y_pred
    if "sample_weight" in sig.parameters:
        kwargs["sample_weight"] = None
    if "training" in sig.parameters:
        kwargs["training"] = True

    model.compute_loss(**kwargs)

    _final("Test Failed ❌", 0)

except Exception as e:
    print(f"DETAIL: {type(e).__name__}: {e}")
    if "different structures" in str(e):
        _final("Test Passed ✅", 0)
    _final(f"HARNESS_ERROR: {type(e).__name__}: {e}", 1)




# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# GPU run
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=0
# export KERAS_BACKEND=tensorflow
# unset TF_XLA_FLAGS

# python testcases/tensorflow_testcase_min.py 2>&1 | tee logs/structured_y_true_case/min_gpu.log

# CPU-only run
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=""
# export KERAS_BACKEND=tensorflow
# unset TF_XLA_FLAGS

# python testcases/tensorflow_testcase_min.py 2>&1 | tee logs/structured_y_true_case/min_cpu.log


# Output:
# *****************

# GPU output
# ENV: {"cuda_visible_devices": "0", "keras": "3.13.2", "python": "3.11.15", "tensorflow": "2.21.0", "visible_gpus": ["/physical_device:GPU:0"]}
# DETAIL: ValueError: y_true and y_pred have different structures.
# y_true: ('*', '*')
# y_pred: *

# Test Passed ✅

# CPU-only output
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# E0000 00:00:1774365180.502511 3553095 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# ENV: {"cuda_visible_devices": "", "keras": "3.13.2", "python": "3.11.15", "tensorflow": "2.21.0", "visible_gpus": []}
# DETAIL: ValueError: y_true and y_pred have different structures.
# y_true: ('*', '*')
# y_pred: *

# Test Passed ✅

# ////////////////////////////////////////////////////////////////////////
#                           Already Reported
# Yes, this is a real bug, but it is already reported: Keras issue #19855.
# ////////////////////////////////////////////////////////////////////////
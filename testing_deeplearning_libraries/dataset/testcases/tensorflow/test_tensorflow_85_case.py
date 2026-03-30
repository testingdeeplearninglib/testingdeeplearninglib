# GCFL-DATAPIPELI-0085_tc_01

import os
import sys

SEED = 2021


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


def main():
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as e:
        _skip(str(e))

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

    # GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    keras = tf.keras

    def np_make_example(_):
        img = np.random.rand(32 * 32 * 3).astype(np.float32)
        y = np.int32(np.random.randint(0, 2))
        return img, y

    base = tf.data.Dataset.range(4)

    def map_fn(i):
        img, y = tf.numpy_function(np_make_example, [i], [tf.float32, tf.int32])
        # *** INTENTIONALLY leave TensorShape unknown here ***
        return img, y

    ds = base.map(map_fn, num_parallel_calls=1).batch(1).prefetch(1)

    inputs = keras.Input(shape=(32, 32, 3), name="img")
    x = keras.layers.Flatten()(inputs)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")

    try:
        model.fit(ds, epochs=1, steps_per_epoch=1, verbose=0)
        _fail()
    except Exception as e:
        m = str(e).lower()
        if isinstance(e, (ValueError, TypeError)) and (
            "as_list() is not defined on an unknown tensorshape" in m
            or "unknown rank" in m
            or "unknown shape" in m
        ):
            _pass()
        else:
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
# source ~/.venvs/tf311/bin/activate

# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=2
# export TF_DETERMINISTIC_OPS=1

# python testcases/tensorflow_testcase.py



# Output:
# *****************
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 ... Created device /job:localhost/replica:0/task:0/device:GPU:0 ...
# Test Passed ✅


# ******************************************************************************

# Reported ✅
# Link: 
# https://github.com/tensorflow/tensorflow/issues/109333
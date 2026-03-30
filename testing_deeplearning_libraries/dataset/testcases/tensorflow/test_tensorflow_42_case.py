# GCFL-OTHER-0042

# A) testcases/gcfl_other_0042_tf_function.py

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import sys
import random

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
        import keras

        # Ensure graph execution for tf.function
        try:
            tf.config.run_functions_eagerly(False)
        except Exception:
            pass

        seed = 2021
        random.seed(seed)
        np.random.seed(seed)
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass

        x_np = np.random.RandomState(seed).randn(1, 12, 16).astype("float32")
        x = tf.constant(x_np)

        layer = keras.layers.SimpleRNN(
            64, dropout=0.0, recurrent_dropout=0.7, return_sequences=True
        )

        _ = layer(x, training=False)  # build

        @tf.function
        def f(inp):
            return layer(inp, training=True)

        def max_abs_diff(a, b):
            return float(tf.reduce_max(tf.abs(a - b)).numpy())

        outs = [f(x) for _ in range(6)]
        diffs = [max_abs_diff(outs[0], outs[i]) for i in range(1, len(outs))]

        thr = 1e-5
        any_diff = any(d > thr for d in diffs)
        print(f"DEBUG: mode=tf_function thr={thr} diffs={diffs}")

        if not any_diff:
            _pass()
        else:
            _fail()

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)

if __name__ == "__main__":
    main()
    
    


# B) testcases/gcfl_other_0042_xla.py

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import sys
import random

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
        import keras

        seed = 2021
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        x = tf.constant(np.random.RandomState(seed).randn(1, 12, 16).astype("float32"))

        layer = keras.layers.SimpleRNN(
            64, dropout=0.0, recurrent_dropout=0.7, return_sequences=True
        )
        _ = layer(x, training=False)

        try:
            @tf.function(jit_compile=True)
            def f(inp):
                return layer(inp, training=True)
        except Exception as e:
            _skip(f"jit_compile not available: {e}")

        def max_abs_diff(a, b):
            return float(tf.reduce_max(tf.abs(a - b)).numpy())

        try:
            outs = [f(x) for _ in range(6)]
        except Exception as e:
            _skip(f"XLA execution failed: {type(e).__name__}: {e}")

        diffs = [max_abs_diff(outs[0], outs[i]) for i in range(1, len(outs))]

        thr = 1e-5
        any_diff = any(d > thr for d in diffs)
        print(f"DEBUG: mode=xla_jit thr={thr} diffs={diffs}")

        if not any_diff:
            _pass()
        else:
            _fail()

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)

if __name__ == "__main__":
    main()




# C) testcases/gcfl_other_0042_serialize.py:
    
# GCFL-OTHER-0042

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import sys
import random

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
        import keras

        seed = 2021
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        x = tf.constant(np.random.RandomState(seed).randn(1, 12, 16).astype("float32"))

        inp = keras.Input(shape=(12, 16))
        out = keras.layers.SimpleRNN(
            64, dropout=0.0, recurrent_dropout=0.7, return_sequences=True
        )(inp, training=True)
        model = keras.Model(inp, out)

        _ = model(x, training=False)

        cfg = model.get_config()
        model2 = keras.Model.from_config(cfg)
        model2.set_weights(model.get_weights())

        def max_abs_diff(a, b):
            return float(tf.reduce_max(tf.abs(a - b)).numpy())

        outs = [model2(x, training=True) for _ in range(6)]
        diffs = [max_abs_diff(outs[0], outs[i]) for i in range(1, len(outs))]

        thr = 1e-5
        any_diff = any(d > thr for d in diffs)
        print(f"DEBUG: mode=serialize_roundtrip thr={thr} diffs={diffs}")

        if not any_diff:
            _pass()
        else:
            _fail()

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)

if __name__ == "__main__":
    main()    




# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Keras


# Commands
# *****************
# conda activate keras_venv
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=0
# export KERAS_BACKEND=tensorflow

# # A) tf.function variant
# python testcases/gcfl_other_0042_tf_function.py
# echo "exit_code=$?"

# # B) XLA jit_compile variant
# python testcases/gcfl_other_0042_xla.py
# echo "exit_code=$?"

# # C) serialization round-trip variant
# python testcases/gcfl_other_0042_serialize.py
# echo "exit_code=$?"



# Output:
# *****************
# Output 1:
# DEBUG: mode=tf_function thr=1e-05 diffs=[1.9756486415863037, 1.9329109191894531, 1.9950804710388184, 1.9632480144500732, 1.9883911609649658]
# Test Failed ❌
# exit_code=0


# Output 2:
# I0000 ... device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
# DEBUG: mode=xla_jit thr=1e-05 diffs=[1.9928654432296753, 1.9779529571533203, 1.9284303188323975, 1.99204421043396, 1.9845867156982422]
# Test Failed ❌
# exit_code=0

# Output 3:
# DEBUG: mode=serialize_roundtrip thr=1e-05 diffs=[1.9650301933288574, 1.9832055568695068, 1.9916187524795532, 1.9822971820831299, 1.996384859085083]
# Test Failed ❌
# exit_code=0
# GCFL-OTHER-0053

import os
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


def _set_determinism():
    seed = 2021
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np  # noqa: F401
        np.random.seed(seed)
    except Exception:
        pass


def main():
    try:
        # Force TF backend for Keras 3 BEFORE importing keras.
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")

        _set_determinism()

        try:
            import numpy as np
        except Exception as e:
            _skip(f"numpy not available: {e}")

        try:
            import tensorflow as tf
        except Exception as e:
            _skip(f"tensorflow not available: {e}")

        try:
            import keras
        except Exception as e:
            _skip(f"keras not available: {e}")

        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        keras_version = getattr(keras, "__version__", "unknown")
        tf_version = getattr(tf, "__version__", "unknown")
        print(f"KERAS_VERSION: {keras_version}")
        print(f"TF_VERSION: {tf_version}")

        def _parse_ver(v: str):
            try:
                parts = (v.split("+")[0].split("-")[0]).split(".")
                nums = [int(p) for p in parts[:3]] + [0] * (3 - len(parts[:3]))
                return tuple(nums[:3])
            except Exception:
                return None

        tfv = _parse_ver(str(tf_version))
        if tfv is not None and tfv < (2, 16, 1):
            _skip(f"tensorflow version < 2.16.1 ({tf_version}); spec targets TF 2.16.1+")

        # Optional visibility info
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            print("TF_VISIBLE_GPUS: [] (CUDA_VISIBLE_DEVICES is empty; skipping GPU probe)")
        else:
            try:
                gpus = tf.config.list_physical_devices("GPU")
                print(f"TF_VISIBLE_GPUS: {gpus}")
            except Exception as e:
                print(f"TF_GPU_QUERY_ERROR: {e}")
                print("TF_VISIBLE_GPUS: []")

        class MatrixLayer(keras.layers.Layer):
            def __init__(self, out_dim=4, **kwargs):
                super().__init__(**kwargs)
                self.out_dim = int(out_dim)

            def build(self, input_shape):
                try:
                    num_fs = int(input_shape[-1])
                except Exception:
                    num_fs = 3

                # BUGGY CALL: positional arg intended as name, but treated as shape in keras3.
                self.matrix = self.add_weight(
                    name="matrix",  # intentionally positional
                    shape=(num_fs, self.out_dim),
                    initializer="glorot_uniform",
                    trainable=True,
                )
                super().build(input_shape)

            def call(self, inputs):
                return keras.ops.matmul(inputs, self.matrix)

        try:
            inp = keras.Input(shape=(5, 9), name="inputs")
            x = inp[:, :, :-1]
            _mask = inp[:, :, -1:]  # noqa: F841

            y = MatrixLayer(out_dim=4, name="matrix_layer")(x)
            model = keras.Model(inp, y)

            data = np.random.RandomState(2021).randn(2, 5, 9).astype("float32")
            _ = model(data, training=False)

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"OBSERVED_EXCEPTION: {msg}")

            # STRICT oracle: only accept the known bug symptom.
            if isinstance(e, TypeError) and "multiple values for argument 'shape'" in str(e):
                _pass()
            else:
                _harness_error(e)

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
# GPU (single GPU):
# conda activate keras_venv
# CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND=tensorflow python testcases/keras_testcase.py 2>logs/tf_stderr.log

# CPU-only:
# conda activate keras_venv
# CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=tensorflow python testcases/keras_testcase.py 2>logs/tf_stderr.log


# Output:
# *****************
# CUDA_VISIBLE_DEVICES: 0
# KERAS_VERSION: 3.4.1
# TF_VERSION: 2.17.0
# TF_VISIBLE_GPUS: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# OBSERVED_EXCEPTION: TypeError: Layer.add_weight() got multiple values for argument 'shape'
# Test Passed ✅





# **************************** Reported ✅ ****************************
# Link: 
# https://github.com/keras-team/keras/issues/22253
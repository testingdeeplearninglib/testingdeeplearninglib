# GCFL-OTHER-0054

import os
import sys
import random
import io
import contextlib
import warnings
import traceback


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


def _parse_major_version(v: str) -> int:
    try:
        s = str(v).strip()
        major = s.split(".", 1)[0]
        return int(major)
    except Exception:
        return -1


def _configure_tf(seed: int):
    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"tensorflow backend not available for keras: {e}")

    # Seed (best effort)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    # Memory growth so TF doesn't reserve all VRAM on a 4x3090 server
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    except Exception:
        pass

    return tf


def _looks_like_signal(text: str) -> bool:
    t = (text or "").lower()
    if "add_metric" not in t:
        return False
    # cover common wording variants
    return any(k in t for k in [
        "deprecat", "deprecated", "deprecation",
        "removed", "no longer supported", "not supported",
        "permanently disabled", "not implemented",
    ])


def main():
    seed = 2021
    random.seed(seed)

    try:
        import numpy as np
    except Exception as e:
        _skip(f"numpy not available: {e}")
    np.random.seed(seed)

    # Must be set BEFORE importing keras.
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    try:
        import keras
    except Exception as e:
        _skip(f"keras not available: {e}")

    keras_major = _parse_major_version(getattr(keras, "__version__", ""))
    if keras_major != -1 and keras_major < 3:
        _skip(f"keras version {getattr(keras, '__version__', 'unknown')} is not Keras 3")

    _configure_tf(seed)

    try:
        class AutoEncoderLike(keras.Model):
            def __init__(self, in_dim: int):
                super().__init__()
                self.d1 = keras.layers.Dense(8, activation="relu")
                self.d2 = keras.layers.Dense(in_dim)

            def call(self, inputs, training=None):
                h = self.d1(inputs)
                reconstructed = self.d2(h)

                reconstruction_loss = keras.losses.MeanSquaredError()(inputs, reconstructed)
                self.add_loss(reconstruction_loss)

                # Under test:
                self.add_metric(reconstruction_loss, name="reconstruction_loss")
                return reconstructed

        x = np.random.randn(8, 4).astype("float32")
        y = x.copy()

        model = AutoEncoderLike(in_dim=4)

        # Key fix for GPU servers: avoid default JIT behavior + keep execution eager
        compile_kwargs = dict(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.MeanSquaredError(),
            run_eagerly=True,
            jit_compile=False,
        )
        try:
            model.compile(**compile_kwargs)
        except TypeError:
            # If this Keras build doesn't accept one of these args, fall back safely.
            compile_kwargs.pop("jit_compile", None)
            try:
                model.compile(**compile_kwargs)
            except TypeError:
                compile_kwargs.pop("run_eagerly", None)
                model.compile(**compile_kwargs)

        observed_signal = False

        buf_out = io.StringIO()
        buf_err = io.StringIO()

        try:
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    model.fit(x, y, batch_size=8, epochs=1, verbose=0)

                    for ww in w:
                        msg = str(getattr(ww, "message", ""))
                        if _looks_like_signal(msg):
                            observed_signal = True
                            break

        except Exception as e:
            # Some implementations raise NotImplementedError with empty message.
            tb = traceback.format_exc().lower()
            if "add_metric" in tb:
                observed_signal = True
            else:
                emsg = str(e)
                if _looks_like_signal(emsg):
                    observed_signal = True

        # Log-based signals
        combined_logs = (buf_out.getvalue() + "\n" + buf_err.getvalue())
        if _looks_like_signal(combined_logs):
            observed_signal = True

        # Behavior-based signal: if add_metric is disabled/no-op, the metric often won't register.
        try:
            has_metric = any(getattr(m, "name", "") == "reconstruction_loss" for m in getattr(model, "metrics", []))
            if not has_metric:
                # treat missing metric as a "disabled/removed" signal
                observed_signal = True
        except Exception:
            pass

        if observed_signal:
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

# Keras + Tensorflow


# Commands
# *****************
# conda activate keras_venv
# export CUDA_VISIBLE_DEVICES=0
# python testcases/keras_testcase.py


# conda activate keras_venv
# export CUDA_VISIBLE_DEVICES=0
# python testcases/keras_testcase.py 2> tf_init_noise.log



# Output:
# *****************
# ... cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
# ... cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
# ... cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered


# Expected output:
# *****************
# 2026-.. ..:..:.. .......: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
# 2026-.. ..:..:.. .......: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
# 2026-.. ..:..:.. .......: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
# Test Passed ✅




# **************************** Reported ✅ ****************************
# Link: 
# https://github.com/tensorflow/tensorflow/issues/110884
# Bug categorized in Keras but reported in tensorflow because backend the issue is related to backend Tensorflow
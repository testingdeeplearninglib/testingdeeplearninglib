# GCFL-TRACINGGRA-0034

import contextlib
import io
import logging
import os
import random
import sys
from typing import List, Tuple


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
    print(f"HARNESS_ERROR: {repr(e)}")
    sys.exit(1)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        self.lines.append(f"{record.levelname}:{record.name}:{msg}")


def _setup_logging_capture() -> Tuple[logging.Logger, _ListHandler]:
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = _ListHandler()
    handler.setLevel(logging.INFO)
    root.addHandler(handler)

    for name in ("absl", "tensorflow"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = True

    return root, handler


def _capture_output_and_logs(fn):
    root, handler = _setup_logging_capture()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            fn()
    finally:
        try:
            root.removeHandler(handler)
        except Exception:
            pass

    text = stdout_buf.getvalue() + "\n" + stderr_buf.getvalue()
    return handler.lines[:], text


def _combined_text(lines: List[str], text: str) -> str:
    return "\n".join(lines) + "\n" + text


def _set_determinism():
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(0)
    try:
        import numpy as np
        np.random.seed(0)
    except Exception:
        pass


def _make_tf_function(tf, fn):
    try:
        return tf.function(fn, reduce_retracing=False)
    except TypeError:
        try:
            return tf.function(fn, experimental_relax_shapes=False)
        except TypeError:
            return tf.function(fn)


def main():
    try:
        _set_determinism()

        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")

        try:
            import absl.logging as absl_logging
        except Exception as e:
            _skip(f"missing absl-py (absl.logging): {type(e).__name__}")

        try:
            import tensorflow as tf
        except Exception as e:
            _skip(f"missing tensorflow: {type(e).__name__}")

        keras = None
        try:
            import keras as _keras  # type: ignore
            keras = _keras
        except Exception:
            try:
                keras = tf.keras  # type: ignore
            except Exception as e:
                _skip(f"missing keras (standalone) and tf.keras unusable: {type(e).__name__}")

        try:
            import numpy as np
        except Exception as e:
            _skip(f"missing numpy: {type(e).__name__}")

        try:
            tf.random.set_seed(0)
        except Exception:
            pass

        try:
            if hasattr(keras, "utils") and hasattr(keras.utils, "set_random_seed"):
                keras.utils.set_random_seed(0)
        except Exception:
            pass

        try:
            tf.config.run_functions_eagerly(False)
        except Exception:
            pass

        try:
            absl_logging.set_verbosity(absl_logging.INFO)
            try:
                absl_logging.set_stderrthreshold("info")
            except Exception:
                pass
        except Exception as e:
            _skip(f"cannot configure absl logging: {type(e).__name__}")

        target = "Creating new FuncGraph for Python function"

        warmup_trace_count = {"count": 0}

        def _warmup_probe():
            def _core_fn(x):
                warmup_trace_count["count"] += 1
                print(f"PY_TRACE_SENTINEL_{warmup_trace_count['count']}")
                return x + tf.constant(1.0, dtype=x.dtype)

            traced = _make_tf_function(tf, _core_fn)

            traced.get_concrete_function(tf.TensorSpec([1, 1], tf.float32))
            traced.get_concrete_function(tf.TensorSpec([2, 1], tf.float32))
            traced.get_concrete_function(tf.TensorSpec([2, 2], tf.float32))

        warmup_lines, warmup_text = _capture_output_and_logs(_warmup_probe)
        warmup_blob = _combined_text(warmup_lines, warmup_text)

        if warmup_trace_count["count"] < 2:
            _skip("unable to force tf.function retracing")

        if target not in warmup_blob:
            _skip("current TF build does not expose the target FuncGraph INFO log; oracle is invalid here")

        def _keras_phase():
            try:
                if hasattr(keras, "utils") and hasattr(keras.utils, "clear_session"):
                    keras.utils.clear_session()
            except Exception:
                pass

            x1 = np.random.RandomState(0).randn(20, 4).astype("float32")
            x2 = np.random.RandomState(1).randn(20, 3).astype("float32")
            y = np.random.RandomState(2).randn(20, 1).astype("float32")

            inp1 = keras.Input(shape=(4,), name="a")
            inp2 = keras.Input(shape=(3,), name="b")

            try:
                concat = keras.layers.Concatenate()([inp1, inp2])
            except Exception:
                concat = keras.layers.Lambda(lambda t: tf.concat(t, axis=-1))([inp1, inp2])

            out = keras.layers.Dense(1)(concat)
            model = keras.Model([inp1, inp2], out)

            model.compile(
                optimizer="adam",
                loss="mse",
                run_eagerly=False,
            )

            model.fit(
                [x1, x2],
                y,
                epochs=3,
                batch_size=5,
                validation_data=([x1, x2], y),
                verbose=0,
            )

            for _ in range(3):
                model.evaluate([x1, x2], y, batch_size=5, verbose=0)

        keras_lines, keras_text = _capture_output_and_logs(_keras_phase)
        keras_blob = _combined_text(keras_lines, keras_text)

        if target not in keras_blob:
            _pass()
        else:
            _fail()

    except SystemExit:
        raise
    except BaseException as e:
        _harness_error(e)


if __name__ == "__main__":
    main()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# conda activate tf_venv
# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=0
# export KERAS_BACKEND=tensorflow
# export TF_CPP_MIN_LOG_LEVEL=0
# export PYTHONUNBUFFERED=1
# python testcases/tensorflow_testcase.py 2>&1 | tee logs/gcfl_tracinggra_0034/run_<timestamp>.log



# Output:
# *****************
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I.... cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions...
# I.... gpu_device.cc:2043] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22446 MB memory ...
# SKIP_ENV: current TF build does not expose the target FuncGraph INFO log; oracle is invalid here
# Test Failed ❌
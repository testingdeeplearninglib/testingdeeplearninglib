# GCFL-SERIALIZATI-0012

import os
import sys
import gc
import time
import json
import tempfile
import random
import shutil

# This testcase is intended to run CPU-only for stability.
# Force this BEFORE importing TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


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


def _read_rss_kb_linux() -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None


def _set_determinism(tf, np):
    random.seed(0)
    np.random.seed(0)
    try:
        tf.keras.utils.set_random_seed(0)
    except Exception:
        try:
            tf.random.set_seed(0)
        except Exception:
            pass


def _print_env(tf, np):
    info = {
        "python": sys.version.split()[0],
        "tensorflow": getattr(tf, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpus_in_process": [d.name for d in tf.config.list_physical_devices("GPU")],
        "rss_kb_start": _read_rss_kb_linux(),
    }
    print("ENV:", json.dumps(info, sort_keys=True))


def _build_compiled_model(tf):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def _cleanup_dir(path: str):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _is_directory_error_message(msg: str) -> bool:
    low = msg.lower()
    return (
        "is a directory" in low
        or "errno = 21" in low
        or "errno 21" in low
        or "unable to open file" in low
        or "failed to open file" in low
        or "could not open file" in low
    )


def _has_high_level_non_directory_guidance(msg: str) -> bool:
    low = msg.lower()
    return (
        "please specify a non-directory filepath" in low
        or ("non-directory" in low and "filepath" in low)
        or ("file path" in low and "directory" in low)
        or ("filepath" in low and "directory" in low and "not" in low)
    )


def _test_dir_as_h5_filepath(tf, np) -> bool:
    tmp = tempfile.mkdtemp(prefix="gcfl_serialization_0012_")
    dir_as_h5 = os.path.join(tmp, "temp.h5")
    os.makedirs(dir_as_h5, exist_ok=True)

    x = np.random.RandomState(0).rand(8, 4).astype("float32")
    y = np.random.RandomState(1).rand(8, 1).astype("float32")

    model = None
    try:
        model = _build_compiled_model(tf)
        cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=dir_as_h5,
            save_best_only=False,
            save_weights_only=False,
        )
        model.fit(x, y, epochs=1, batch_size=4, verbose=0, callbacks=[cb])
        return True
    except BaseException as e:
        msg = str(e)

        if _has_high_level_non_directory_guidance(msg):
            return False

        if _is_directory_error_message(msg):
            return True

        return False
    finally:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        gc.collect()
        _cleanup_dir(tmp)


def _save_h5_model_once(tf, np, h5_path: str) -> bool:
    model = None
    try:
        model = _build_compiled_model(tf)
        _ = model(np.zeros((1, 4), dtype="float32"), training=False)
        model.save(h5_path)
        return os.path.exists(h5_path)
    except BaseException:
        return False
    finally:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        gc.collect()


def _test_load_model_rss_leak(tf, np) -> bool:
    if _read_rss_kb_linux() is None:
        return False

    tmp = tempfile.mkdtemp(prefix="gcfl_serialization_0012_model_")
    h5_path = os.path.join(tmp, "model.h5")

    try:
        if not _save_h5_model_once(tf, np, h5_path):
            return False

        for _ in range(2):
            mm = None
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
            try:
                mm = tf.keras.models.load_model(h5_path, compile=False)
                _ = mm(np.zeros((1, 4), dtype="float32"), training=False)
            finally:
                try:
                    del mm
                except Exception:
                    pass
                gc.collect()

        rss = []
        for _ in range(20):
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass

            mm = None
            try:
                mm = tf.keras.models.load_model(h5_path, compile=False)
                _ = mm(np.zeros((1, 4), dtype="float32"), training=False)
            finally:
                try:
                    del mm
                except Exception:
                    pass
                gc.collect()

            cur = _read_rss_kb_linux()
            if cur is None:
                return False
            rss.append(cur)
            time.sleep(0.05)

        if len(rss) < 6:
            return False

        diffs = [rss[i] - rss[i - 1] for i in range(1, len(rss))]
        total_inc = rss[-1] - rss[0]
        tail_inc = rss[-1] - rss[-6]

        mb = 1024
        big_pos = sum(1 for d in diffs if d > 512)
        big_neg = sum(1 for d in diffs if d < -512)
        required_big_pos = max(1, int(0.8 * len(diffs)))

        return (
            total_inc >= 40 * mb
            and tail_inc >= 5 * 512
            and big_neg == 0
            and big_pos >= required_big_pos
        )
    finally:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()
        _cleanup_dir(tmp)


def main():
    try:
        try:
            import numpy as np
        except Exception as e:
            _skip(f"missing numpy: {type(e).__name__}: {e}")

        try:
            import tensorflow as tf
        except Exception as e:
            _skip(f"missing tensorflow: {type(e).__name__}: {e}")

        try:
            import h5py  # noqa: F401
        except Exception as e:
            _skip(f"missing h5py (required for .h5 serialization/load_model): {type(e).__name__}: {e}")

        _set_determinism(tf, np)
        _print_env(tf, np)

        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    tf.config.set_visible_devices([], "GPU")
                except Exception:
                    pass
        except Exception:
            pass

        bug_a = _test_dir_as_h5_filepath(tf, np)
        bug_b = _test_load_model_rss_leak(tf, np)

        print(f"SUBTEST_DIRPATH_BUG={bug_a}")
        print(f"SUBTEST_RSS_LEAK_BUG={bug_b}")

        if bug_a or bug_b:
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
# cd ~/dl_testing
# conda activate tf_venv

# unset TF_XLA_FLAGS
# unset CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=""
# export PYTHONUNBUFFERED=1

# set -o pipefail
# python testcases/tensorflow_testcase.py 2>&1 | tee logs_gcfl_serialization_0012_rerun.txt
# echo "exit_code=$?"


# Output:
# # *****************
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# E0000 00:00:1774270097.591132 1482447 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# ENV: {"cuda_visible_devices": "", "numpy": "1.26.4", "pid": 1482447, "python": "3.11.15", "rss_kb_start": 564376, "tensorflow": "2.21.0", "visible_gpus_in_process": []}
# WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# SUBTEST_DIRPATH_BUG=False
# SUBTEST_RSS_LEAK_BUG=False
# Test Failed ❌
# exit_code=0
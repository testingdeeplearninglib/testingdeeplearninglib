# GCFL-OTHER-0071

import importlib.util
import json
import locale
import os
import platform
import random
import sys
import tempfile
from pathlib import Path


def _safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        fallback = msg.encode("ascii", errors="backslashreplace").decode("ascii")
        sys.stdout.write(fallback + "\n")
        sys.stdout.flush()


def _skip(reason: str):
    _safe_print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass():
    _safe_print("Test Passed ✅")
    sys.exit(0)


def _fail():
    _safe_print("Test Failed ❌")
    sys.exit(0)


def _harness_error(e: BaseException):
    _safe_print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
    sys.exit(1)


def _set_determinism():
    random.seed(20260214)
    try:
        import numpy as np
        np.random.seed(20260214)
    except Exception:
        pass


def _force_ascii_locale_best_effort():
    # Best-effort emulation of old ASCII-locale behavior.
    try:
        os.environ.setdefault("LC_ALL", "C")
        os.environ.setdefault("LANG", "C")
        os.environ.setdefault("PYTHONCOERCECLOCALE", "0")
        locale.setlocale(locale.LC_ALL, "C")
    except Exception:
        pass


def _env_snapshot():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "preferred_encoding": locale.getpreferredencoding(False),
        "LC_ALL": os.environ.get("LC_ALL"),
        "LANG": os.environ.get("LANG"),
        "PYTHONUTF8": os.environ.get("PYTHONUTF8"),
        "PYTHONCOERCECLOCALE": os.environ.get("PYTHONCOERCECLOCALE"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _import_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception as e:
        _skip(f"tensorflow not available: {type(e).__name__}: {e}")


def _load_local_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_data_utils():
    # 1) Original old TensorFlow tutorial path
    try:
        from tensorflow.models.rnn.translate import data_utils  # type: ignore
        return data_utils
    except Exception:
        pass

    try:
        data_utils = __import__(
            "tensorflow.models.rnn.translate.data_utils",
            fromlist=["*"],
        )
        return data_utils
    except Exception:
        pass

    # 2) Local vendored fallback next to this testcase
    here = Path(__file__).resolve().parent
    candidates = [
        here / "legacy_tf_translate_data_utils.py",
        here / "data_utils.py",
    ]

    for candidate in candidates:
        if candidate.is_file():
            try:
                return _load_local_module(candidate)
            except Exception as e:
                _skip(f"found local legacy data_utils but failed to import {candidate.name}: {type(e).__name__}: {e}")

    _skip(
        "seq2seq translate tutorial modules not available. "
        "Modern TensorFlow wheels do not ship tensorflow.models.rnn.translate.data_utils; "
        "vendor legacy data_utils.py beside this testcase to attempt true reproduction."
    )


def main():
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        _set_determinism()
        _force_ascii_locale_best_effort()

        print("ENV:", json.dumps(_env_snapshot(), sort_keys=True, ensure_ascii=True))

        tf = _import_tf()
        print(f"TF_VERSION: {getattr(tf, '__version__', 'unknown')}")

        data_utils = _import_data_utils()
        if not hasattr(data_utils, "create_vocabulary"):
            _skip("data_utils.create_vocabulary not found; cannot run reproduction")

        with tempfile.TemporaryDirectory() as td:
            train_path = os.path.join(td, "train.fr")
            vocab_path = os.path.join(td, "vocab40000.fr")

            lines = [
                "hello world\n",
                "café crème\n",
                "naïve façade\n",
                "こんにちは 世界\n",
            ]

            with open(train_path, "wb") as f:
                f.write("".join(lines).encode("utf-8"))

            try:
                data_utils.create_vocabulary(vocab_path, train_path, 40000)
            except (UnicodeError, TypeError):
                _pass()
            except Exception as e:
                print(f"UNEXPECTED_EXCEPTION: {type(e).__name__}: {e}")
                _fail()
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

# Tensorflow


# Commands
# *****************
# conda activate tf_venv
# export PYTHONUTF8=0
# export PYTHONCOERCECLOCALE=0
# export LC_ALL=C
# export LANG=C
# export CUDA_VISIBLE_DEVICES=""

# python ~/dl_testing/testcases/tensorflow_testcase.py 2>&1 | tee ~/dl_testing/testcases/tensorflow_testcase_with_legacy_module.log
# echo "exit_code=$?"


# Output:
# *****************
# ENV: {"CUDA_VISIBLE_DEVICES": "", "LANG": "C", "LC_ALL": "C", "PYTHONCOERCECLOCALE": "0", "PYTHONUTF8": "0", "platform": "Linux-6.8.0-31-generic-x86_64-with-glibc2.39", "preferred_encoding": "ANSI_X3.4-1968", "python": "3.10.19"}
# TF_VERSION: 2.21.0
# Creating vocabulary /tmp/tmp62pdjvbj/vocab40000.fr from data /tmp/tmp62pdjvbj/train.fr
# Test Failed \u274c
# exit_code=0


# Test Failed ❌
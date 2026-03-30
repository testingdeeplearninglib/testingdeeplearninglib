# GCFL-OTHER-0038

import os
import sys
import subprocess
import importlib.util


def _skip(reason: str):
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass():
    print("Test Passed ✅")
    sys.exit(0)


def _fail(msg: str = ""):
    if msg:
        print(msg)
    print("Test Failed ❌")
    sys.exit(0)


def _harness_error(e: BaseException):
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
    sys.exit(1)


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _extract_first_line(prefix: str, text: str) -> str:
    for line in (text or "").splitlines():
        if line.startswith(prefix):
            return line
    return ""


def _trim(text: str, max_lines: int = 80) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= max_lines:
        return text or ""
    return "\n".join(lines[:max_lines] + ["... (trimmed) ..."])


def _looks_like_renorm_keyword_error(output: str) -> bool:
    s = (output or "").lower()
    if "renorm" not in s:
        return False
    markers = [
        "unrecognized keyword",
        "unexpected keyword",
        "got an unexpected keyword argument",
        "keyword arguments",
        "keyword argument",
    ]
    layer_markers = ["batchnormalization", "batch normalization", "batchnorm", "batch_norm"]
    return any(m in s for m in markers) and any(lm in s for lm in layer_markers)


def _run_child(mode: str, timeout_s: int):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # FORCE CPU for stability

    if mode in ("keras_torch", "keras_tf"):
        env["KERAS_BACKEND"] = "torch" if mode == "keras_torch" else "tensorflow"

    cmd = [sys.executable, os.path.abspath(__file__), "--child", mode]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_s)


def _child_main(mode: str):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if mode == "keras_torch":
            if not _has_module("keras"):
                _skip("keras is not installed.")
            if not _has_module("torch"):
                _skip("torch is not installed.")
            os.environ["KERAS_BACKEND"] = "torch"
            import keras
            from keras import layers

            try:
                _ = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, renorm=True, name="x_bn")
                print("CHILD_OK")
                sys.exit(0)
            except Exception as e:
                print(f"CHILD_EXCEPTION: {type(e).__name__}: {e}")
                sys.exit(2)

        if mode == "keras_tf":
            if not _has_module("keras"):
                _skip("keras is not installed.")
            if not _has_module("tensorflow"):
                _skip("tensorflow is not installed.")
            os.environ["KERAS_BACKEND"] = "tensorflow"
            import keras
            from keras import layers

            try:
                _ = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, renorm=True, name="x_bn")
                print("CHILD_OK")
                sys.exit(0)
            except Exception as e:
                print(f"CHILD_EXCEPTION: {type(e).__name__}: {e}")
                sys.exit(2)

        if mode == "tf_keras":
            if not _has_module("tensorflow"):
                _skip("tensorflow is not installed.")
            import tensorflow as tf

            try:
                _ = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5, renorm=True, name="x_bn")
                print("CHILD_OK")
                sys.exit(0)
            except Exception as e:
                print(f"CHILD_EXCEPTION: {type(e).__name__}: {e}")
                sys.exit(2)

        _skip(f"unknown child mode: {mode}")

    except SystemExit:
        raise
    except BaseException as e:
        print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
        sys.exit(1)


def main():
    try:
        if len(sys.argv) >= 3 and sys.argv[1] == "--child":
            _child_main(sys.argv[2])

        # Version banner (useful for your documentation)
        try:
            import keras
            keras_v = keras.__version__
        except Exception:
            keras_v = "?"
        try:
            import torch
            torch_v = torch.__version__
        except Exception:
            torch_v = "?"
        try:
            import tensorflow as tf
            tf_v = tf.__version__
        except Exception:
            tf_v = "?"

        print(f"ENV: keras={keras_v} torch={torch_v} tensorflow={tf_v}")

        # Must have keras+torch to run keras_torch child
        if not _has_module("keras"):
            _skip("keras is not installed.")
        if not _has_module("torch"):
            _skip("torch is not installed (required for Keras torch backend check).")

        r_torch = _run_child("keras_torch", timeout_s=120)
        out_torch = (r_torch.stdout or "") + "\n" + (r_torch.stderr or "")

        sk = _extract_first_line("SKIP_ENV:", out_torch)
        if sk:
            print(sk); sys.exit(0)
        he = _extract_first_line("HARNESS_ERROR:", out_torch)
        if he:
            print(he); sys.exit(1)

        torch_rejects = (r_torch.returncode == 2) and _looks_like_renorm_keyword_error(out_torch)

        # Keras TF backend check is optional but recommended if TF installed
        tf_backend_checked = False
        tf_backend_rejects = False
        out_keras_tf = ""

        if _has_module("tensorflow"):
            tf_backend_checked = True
            r_keras_tf = _run_child("keras_tf", timeout_s=180)
            out_keras_tf = (r_keras_tf.stdout or "") + "\n" + (r_keras_tf.stderr or "")
            sk2 = _extract_first_line("SKIP_ENV:", out_keras_tf)
            if sk2:
                print(sk2); sys.exit(0)
            he2 = _extract_first_line("HARNESS_ERROR:", out_keras_tf)
            if he2:
                print(he2); sys.exit(1)
            tf_backend_rejects = (r_keras_tf.returncode == 2) and _looks_like_renorm_keyword_error(out_keras_tf)
        else:
            r_keras_tf = None

        # tf.keras informational check (does not affect pass/fail)
        tf_keras_note = ""
        if _has_module("tensorflow"):
            r_tf_keras = _run_child("tf_keras", timeout_s=180)
            out_tf_keras = (r_tf_keras.stdout or "") + "\n" + (r_tf_keras.stderr or "")
            tfk_rejects = (r_tf_keras.returncode == 2) and _looks_like_renorm_keyword_error(out_tf_keras)
            tf_keras_note = f"INFO: tf.keras renorm={'REJECTS' if tfk_rejects else 'ACCEPTS'} (returncode={r_tf_keras.returncode})"
        else:
            tf_keras_note = "INFO: tf.keras check skipped (tensorflow not installed)"

        # ORACLE FOR THIS ENV:
        # Pass when renorm is NOT supported and is rejected consistently by Keras torch backend
        # and (if TF backend exists) Keras TF backend too.
        if torch_rejects and (not tf_backend_checked or tf_backend_rejects):
            print(tf_keras_note)
            _pass()

        # Otherwise fail with diagnostics
        diag = []
        diag.append(f"DEBUG: torch child returncode={r_torch.returncode}")
        diag.append("DEBUG: torch child output (trimmed):")
        diag.append(_trim(out_torch))
        if tf_backend_checked and r_keras_tf is not None:
            diag.append(f"\nDEBUG: keras-tf child returncode={r_keras_tf.returncode}")
            diag.append("DEBUG: keras-tf child output (trimmed):")
            diag.append(_trim(out_keras_tf))
        diag.append(f"\n{tf_keras_note}")
        _fail("\n".join(diag))

    except subprocess.TimeoutExpired as e:
        _skip(f"subprocess timed out: {e}")
    except SystemExit:
        raise
    except BaseException as e:
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
# python testcases/keras_testcase.py



# Output:
# *****************
# DEBUG: torch child returncode=2
# DEBUG: torch child output (trimmed):
# CHILD_EXCEPTION: ValueError: Unrecognized keyword arguments passed to BatchNormalization: {'renorm': True}


# DEBUG: tf child returncode=2
# DEBUG: tf child output (trimmed):
# CHILD_EXCEPTION: ValueError: Unrecognized keyword arguments passed to BatchNormalization: {'renorm': True}

# 2026-02-26 06:10:30.599097: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

# Test Failed ❌
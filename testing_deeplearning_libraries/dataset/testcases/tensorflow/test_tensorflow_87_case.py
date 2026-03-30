# GCFL-OTHER-0087

import os
import sys
import platform


def _print_and_exit(msg: str, code: int = 0):
    print(msg)
    sys.exit(code)


def _skip(reason: str):
    _print_and_exit(f"SKIP_ENV: {reason}", 0)


def _pass():
    _print_and_exit("Test Passed ✅", 0)


def _fail():
    _print_and_exit("Test Failed ❌", 0)


def main():
    try:
        import numpy as np
    except Exception as e:
        _skip(f"missing numpy ({type(e).__name__}: {e})")

    np.random.seed(2021)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"missing tensorflow ({type(e).__name__}: {e})")

    try:
        tf.compat.v1.disable_eager_execution()
    except Exception as e:
        _skip(f"cannot disable eager execution / enter TF1 graph mode ({type(e).__name__}: {e})")

    v1 = tf.compat.v1

    try:
        import tensorflow_hub as hub
    except Exception as e:
        _skip(f"missing tensorflow_hub ({type(e).__name__}: {e})")

    if not hasattr(hub, "Module"):
        _skip("tensorflow_hub.Module unavailable in this install")

    gpus = []
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:
        pass

    print(
        "ENV: "
        f"python={platform.python_version()} "
        f"tf={getattr(tf, '__version__', 'unknown')} "
        f"numpy={getattr(np, '__version__', 'unknown')} "
        f"tf_hub={getattr(hub, '__version__', 'unknown')} "
        f"visible_gpus={len(gpus)}"
    )

    url = "https://tfhub.dev/google/elmo/2"

    try:
        graph = tf.Graph()
        with graph.as_default():
            x_ph = v1.placeholder(tf.string, shape=[None, None], name="text_input")

            # Legacy TF1 Hub module. This testcase intentionally targets that path.
            embed = hub.Module(url, trainable=True)

            # Intentional trigger: [1,1] -> scalar [] after squeeze.
            x_squeezed = tf.squeeze(x_ph, name="squeezed_text")

            out_dict = embed(x_squeezed, signature="default", as_dict=True)
            if "elmo" not in out_dict:
                _skip("ELMo module did not expose 'elmo' output")

            out = out_dict["elmo"]

            init_op = v1.group(
                v1.global_variables_initializer(),
                v1.tables_initializer(),
            )

    except SystemExit:
        raise
    except Exception as e:
        _skip(f"cannot build TF1 graph / hub.Module ELMo pipeline ({type(e).__name__}: {e})")

    try:
        config = v1.ConfigProto()
        config.gpu_options.allow_growth = True

        with v1.Session(graph=graph, config=config) as sess:
            try:
                sess.run(init_op)
            except Exception as e:
                _skip(f"graph initialization failed (likely hub/network/cache/env issue) ({type(e).__name__}: {e})")

            def run_on(text_list):
                feed = {x_ph: np.asarray([text_list], dtype=str)}
                return sess.run(out, feed_dict=feed)

            # Baseline: should be a vector after squeeze, not scalar.
            try:
                _ = run_on(["hello", "my name is Simone"])
            except Exception as e:
                _skip(f"baseline multi-element input failed; reproduction not trustworthy ({type(e).__name__}: {e})")

            # Trigger: becomes scalar [] after squeeze, which should provoke the reported error.
            try:
                _ = run_on(["hello"])
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                low = msg.lower()

                if (
                    "input must be a vector" in low
                    or "got shape: []" in low
                    or ("shape: []" in low and "vector" in low)
                ):
                    _pass()

                _fail()

            # No exception means bug did not reproduce.
            _fail()

    except SystemExit:
        raise
    except Exception as e:
        _print_and_exit(f"HARNESS_ERROR: {type(e).__name__}: {e}", 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        _print_and_exit(f"HARNESS_ERROR: {type(e).__name__}: {e}", 1)



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Tensorflow


# Commands
# *****************
# cd ~/dl_testing
# conda activate tf_venv

# python -m pip install --upgrade "setuptools<82"
# python -m pip show setuptools

# python - <<'PY'
# import tensorflow as tf
# import tensorflow_hub as hub

# print("TF:", tf.__version__)
# print("TF-Hub:", getattr(hub, "__version__", "unknown"))
# print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
# print("has hub.Module:", hasattr(hub, "Module"))
# PY

# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=1
# export TFHUB_CACHE_DIR="$HOME/.cache/tfhub"
# mkdir -p "$TFHUB_CACHE_DIR"

# set -o pipefail
# python testcase/tensorflow_testcase.py 2>&1 | tee logs/GCFL-OTHER-0087.log
# echo "exit_code=$?"


# Output:
# *****************
# Name: setuptools
# Version: 81.0.0

# TF: 2.21.0
# TF-Hub: 0.16.1
# Visible GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# has hub.Module: False

# /home/talha/miniconda3/envs/tf_venv/lib/python3.11/site-packages/tensorflow_hub/__init__.py:61: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
#   from pkg_resources import parse_version
# SKIP_ENV: tensorflow_hub.Module unavailable in this install
# exit_code=0
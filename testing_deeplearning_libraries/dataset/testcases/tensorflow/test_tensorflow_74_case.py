# GCFL-OTHER-0074

import sys
import random
from typing import Optional


def _print_and_exit(msg: str, code: int):
    print(msg)
    sys.exit(code)


def _skip(reason: str):
    _print_and_exit(f"SKIP_ENV: {reason}", 0)


def _pass():
    _print_and_exit("Test Passed ✅", 0)


def _fail():
    _print_and_exit("Test Failed ❌", 0)


def _harness_error(e: BaseException):
    _print_and_exit(f"HARNESS_ERROR: {type(e).__name__}: {e}", 1)


def _set_seeds(seed: int = 2021):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _import_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception as e:
        _skip(f"missing tensorflow ({e})")


def _get_sparse_column_fns(tf):
    try:
        contrib = getattr(tf, "contrib", None)
        if contrib is None:
            return None, None
        layers = getattr(contrib, "layers", None)
        if layers is None:
            return None, None
        f_hash = getattr(layers, "sparse_column_with_hash_bucket", None)
        f_keys = getattr(layers, "sparse_column_with_keys", None)
        if not callable(f_hash) or not callable(f_keys):
            return None, None
        return f_hash, f_keys
    except Exception:
        return None, None


def _try_reproduce_via_public_estimator(tf, col_a, col_b) -> Optional[bool]:
    """
    Try the most faithful path: instantiate DNNLinearCombinedClassifier and trigger
    the internal sorting behavior.
    Returns:
      True  -> bug reproduced (TypeError during sorting/order)
      False -> no bug (no TypeError)
      None  -> cannot attempt this path
    """
    try:
        contrib = getattr(tf, "contrib", None)
        if contrib is None:
            return None
        learn = getattr(contrib, "learn", None)
        if learn is None:
            return None
        DNN = getattr(learn, "DNNLinearCombinedClassifier", None)
        if DNN is None or not callable(DNN):
            return None

        try:
            est = DNN(
                model_dir=None,
                linear_feature_columns=None,
                dnn_feature_columns=[col_a, col_b],
                dnn_hidden_units=[2],
            )
        except TypeError:
            return True

        meth = getattr(est, "_get_dnn_feature_columns", None)
        if callable(meth):
            try:
                _ = meth()
                return False
            except TypeError:
                return True

        return None
    except Exception:
        return None


def _try_reproduce_via_module_method(tf, col_a, col_b) -> Optional[bool]:
    """
    Fallback: import the contrib.learn dnn_linear_combined module and invoke the method
    `_get_dnn_feature_columns` on an object with `_dnn_feature_columns`.
    Returns True/False/None same as above.
    """
    import importlib
    import inspect

    module_paths = [
        "tensorflow.contrib.learn.python.learn.estimators.dnn_linear_combined",
    ]

    dlc = None
    for p in module_paths:
        try:
            dlc = importlib.import_module(p)
            break
        except Exception:
            dlc = None

    if dlc is None:
        return None

    target_cls = None
    for _, cls in inspect.getmembers(dlc, inspect.isclass):
        if getattr(cls, "__module__", "") != getattr(dlc, "__name__", ""):
            continue
        if hasattr(cls, "_get_dnn_feature_columns") and callable(getattr(cls, "_get_dnn_feature_columns")):
            target_cls = cls
            break

    target_func = getattr(dlc, "_get_dnn_feature_columns", None)
    if target_cls is None and not callable(target_func):
        return None

    try:
        if target_cls is not None:
            try:
                obj = target_cls.__new__(target_cls)
            except Exception:
                class _Dummy(object):
                    pass
                obj = _Dummy()

            setattr(obj, "_dnn_feature_columns", [col_a, col_b])
            meth = getattr(target_cls, "_get_dnn_feature_columns")
            try:
                _ = meth(obj)
                return False
            except TypeError:
                return True
        else:
            class _Dummy(object):
                pass
            obj = _Dummy()
            setattr(obj, "_dnn_feature_columns", [col_a, col_b])
            try:
                _ = target_func(obj)
                return False
            except TypeError:
                return True
    except Exception:
        return None


def main():
    try:
        if not ((3, 5) <= sys.version_info[:2] <= (3, 7)):
            _skip("requires Python 3.5-3.7 because this testcase targets TensorFlow 1.x contrib")

        _set_seeds(2021)
        tf = _import_tf()

        contrib = getattr(tf, "contrib", None)
        if contrib is None:
            _skip("tensorflow.contrib not available (likely TF 2.x build)")

        learn = getattr(contrib, "learn", None)
        if learn is None:
            _skip("tensorflow.contrib.learn not available")

        f_hash, f_keys = _get_sparse_column_fns(tf)
        if f_hash is None or f_keys is None:
            _skip("tensorflow.contrib.layers sparse column APIs not available")

        try:
            col_hashed = f_hash("hashed_col", hash_bucket_size=64)
            col_keys = f_keys("keys_col", keys=["a", "b", "c"])
        except Exception as e:
            _skip("cannot construct contrib sparse columns ({})".format(e))

        res = _try_reproduce_via_public_estimator(tf, col_hashed, col_keys)
        if res is True:
            _pass()
        if res is False:
            _fail()

        res2 = _try_reproduce_via_module_method(tf, col_hashed, col_keys)
        if res2 is True:
            _pass()
        if res2 is False:
            _fail()

        _skip("unable to access contrib.learn dnn_linear_combined code path in this TensorFlow build")

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
# export CUDA_VISIBLE_DEVICES=""
# python testcase/tensorflow_testcase.py
# echo "exit_code=$?"


# Output:
# *****************
# WARNING:tensorflow:
# The TensorFlow contrib module will not be included in TensorFlow 2.0.
# For more information, please see:
#   * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
#   * https://github.com/tensorflow/addons
#   * https://github.com/tensorflow/io (for I/O related ops)
# If you depend on functionality not listed there, please file an issue.

# WARNING:tensorflow:From testcase/tensorflow_testcase.py:89: calling DNNLinearCombinedClassifier.__init__ (from tensorflow.contrib.learn.python.learn.estimators.dnn_linear_combined) with fix_global_step_increment_bug=False is deprecated and will be removed after 2017-04-15.
# Instructions for updating:
# Please set fix_global_step_increment_bug=True and update training steps in your pipeline. See pydoc for details.

# WARNING:tensorflow:From /home/talha/miniconda3/envs/tf_venv/lib/python3.7/site-packages/tensorflow_core/contrib/learn/python/learn/estimators/dnn_linear_combined.py:676: multi_class_head (from tensorflow.contrib.learn.python.learn.estimators.head) is deprecated and will be removed in a future version.
# Instructions for updating:
# Please switch to tf.contrib.estimator.*_head.

# WARNING:tensorflow:From /home/talha/miniconda3/envs/tf_venv/lib/python3.7/site-packages/tensorflow_core/contrib/learn/python/learn/estimators/estimator.py:1180: BaseEstimator.__init__ (from tensorflow.contrib.learn.python.learn.estimators.estimator) is deprecated and will be removed in a future version.
# Instructions for updating:
# Please replace uses of any Estimator from tf.contrib.learn with an Estimator from tf.estimator.*

# WARNING:tensorflow:From /home/talha/miniconda3/envs/tf_venv/lib/python3.7/site-packages/tensorflow_core/contrib/learn/python/learn/estimators/estimator.py:427: RunConfig.__init__ (from tensorflow.contrib.learn.python.learn.estimators.run_config) is deprecated and will be removed in a future version.
# Instructions for updating:
# When switching to tf.estimator.Estimator, use tf.estimator.RunConfig instead.

# WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmppa6k9w61
# Test Failed ❌
# exit_code=0
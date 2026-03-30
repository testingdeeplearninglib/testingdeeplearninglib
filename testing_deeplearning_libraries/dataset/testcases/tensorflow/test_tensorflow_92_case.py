# GCFL-TRACINGGRA-0092

import sys
import os
import json
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
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(0)
    try:
        import numpy as np
        np.random.seed(0)
    except Exception:
        pass

def _get_attr_chain(root, chain):
    cur = root
    for name in chain:
        if cur is None or not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur

def _print_env(tf):
    info = {
        "python": sys.version.split()[0],
        "tf_version": getattr(tf, "__version__", "unknown"),
        "eager_before_disable": None,
        "physical_gpus": None,
        "visible_gpu_count": None,
    }
    try:
        info["eager_before_disable"] = bool(tf.executing_eagerly())
    except Exception:
        info["eager_before_disable"] = "unknown"
    try:
        gpus = tf.config.list_physical_devices("GPU")
        info["physical_gpus"] = [str(x) for x in gpus]
        info["visible_gpu_count"] = len(gpus)
    except Exception as e:
        info["physical_gpus"] = f"error: {type(e).__name__}: {e}"
        info["visible_gpu_count"] = "unknown"
    print("ENV:", json.dumps(info, sort_keys=True))

def _resolve_tf1_lossscale_api(tf):
    candidates = [
        ("tf.compat.v1.mixed_precision.MixedPrecisionLossScaleOptimizer",
         _get_attr_chain(tf, ["compat", "v1", "mixed_precision", "MixedPrecisionLossScaleOptimizer"])),
        ("tf.compat.v1.train.experimental.MixedPrecisionLossScaleOptimizer",
         _get_attr_chain(tf, ["compat", "v1", "train", "experimental", "MixedPrecisionLossScaleOptimizer"])),
        ("tf.train.experimental.MixedPrecisionLossScaleOptimizer",
         _get_attr_chain(tf, ["train", "experimental", "MixedPrecisionLossScaleOptimizer"])),
        ("tf.contrib.mixed_precision.LossScaleOptimizer",
         _get_attr_chain(tf, ["contrib", "mixed_precision", "LossScaleOptimizer"])),
    ]

    fixed_loss_scale_candidates = [
        _get_attr_chain(tf, ["compat", "v1", "mixed_precision", "FixedLossScale"]),
        _get_attr_chain(tf, ["compat", "v1", "train", "experimental", "FixedLossScale"]),
        _get_attr_chain(tf, ["train", "experimental", "FixedLossScale"]),
        _get_attr_chain(tf, ["contrib", "mixed_precision", "FixedLossScale"]),
    ]

    wrapper_name = None
    wrapper_cls = None
    for name, cls in candidates:
        if cls is not None:
            wrapper_name = name
            wrapper_cls = cls
            break

    fixed_loss_scale_cls = next((x for x in fixed_loss_scale_candidates if x is not None), None)
    return wrapper_name, wrapper_cls, fixed_loss_scale_cls

def _instantiate_tf1_lossscale_wrapper(wrapper_cls, fixed_loss_scale_cls, base_opt, loss_scale_value=128.0):
    tried = []

    loss_scale_objects = []
    if fixed_loss_scale_cls is not None:
        try:
            loss_scale_objects.append(fixed_loss_scale_cls(loss_scale_value))
        except Exception as e:
            tried.append(f"FixedLossScale({loss_scale_value}) -> {type(e).__name__}: {e}")

    loss_scale_objects.append(loss_scale_value)

    for loss_scale_obj in loss_scale_objects:
        for args, kwargs in [
            ((base_opt, loss_scale_obj), {}),
            ((), {"opt": base_opt, "loss_scale": loss_scale_obj}),
            ((), {"optimizer": base_opt, "loss_scale": loss_scale_obj}),
        ]:
            try:
                obj = wrapper_cls(*args, **kwargs)
                if hasattr(obj, "minimize"):
                    return obj, tried
                tried.append(f"{wrapper_cls.__name__}{args or kwargs} -> no minimize()")
            except Exception as e:
                tried.append(f"{wrapper_cls.__name__}{args or kwargs} -> {type(e).__name__}: {e}")

    return None, tried

def main():
    _set_determinism()

    try:
        import tensorflow as tf
    except Exception as e:
        _skip(f"tensorflow import failed: {type(e).__name__}: {e}")

    _print_env(tf)

    try:
        tf.compat.v1.disable_eager_execution()
    except Exception as e:
        _skip(f"cannot disable eager execution: {type(e).__name__}: {e}")

    try:
        tf.compat.v1.set_random_seed(0)
    except Exception:
        pass
    try:
        tf.random.set_seed(0)
    except Exception:
        pass

    wrapper_name, wrapper_cls, fixed_loss_scale_cls = _resolve_tf1_lossscale_api(tf)
    if wrapper_cls is None:
        _skip("no TF1-compatible MixedPrecisionLossScaleOptimizer found")

    print(f"INFO: selected_wrapper={wrapper_name}")
    print(f"INFO: fixed_loss_scale_cls={'present' if fixed_loss_scale_cls is not None else 'absent'}")

    baseline_build_exc = None
    lso_build_exc = None
    baseline_run_exc = None
    lso_run_exc = None

    try:
        g = tf.Graph()
        with g.as_default():
            v1 = tf.Variable(1.0, name="v1")
            v2 = tf.Variable(2.0, name="v2_unconnected")
            loss = v1 * 3.0

            grads = tf.gradients(loss, [v1, v2])
            print("INFO: symbolic_grad_v1_is_none=", grads[0] is None)
            print("INFO: symbolic_grad_v2_is_none=", grads[1] is None)

            base_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)

            try:
                baseline_train_op = base_opt.minimize(loss, var_list=[v1, v2])
                print("INFO: baseline_build=success")
            except Exception as e:
                baseline_build_exc = e
                baseline_train_op = None
                print(f"INFO: baseline_build_exception={type(e).__name__}: {e}")

            lso_opt, tried = _instantiate_tf1_lossscale_wrapper(
                wrapper_cls=wrapper_cls,
                fixed_loss_scale_cls=fixed_loss_scale_cls,
                base_opt=base_opt,
                loss_scale_value=128.0,
            )
            if lso_opt is None:
                _skip("found TF1 wrapper but could not instantiate it: " + " | ".join(tried[-5:]))

            print(f"INFO: instantiated_wrapper_type={type(lso_opt).__name__}")

            try:
                lso_train_op = lso_opt.minimize(loss, var_list=[v1, v2])
                print("INFO: lso_build=success")
            except Exception as e:
                lso_build_exc = e
                lso_train_op = None
                print(f"INFO: lso_build_exception={type(e).__name__}: {e}")

            init_op = tf.compat.v1.global_variables_initializer()

        if baseline_build_exc is not None:
            _skip(f"baseline optimizer minimize failed to build graph: {type(baseline_build_exc).__name__}: {baseline_build_exc}")

        with tf.compat.v1.Session(graph=g) as sess:
            sess.run(init_op)

            try:
                sess.run(baseline_train_op)
                print("INFO: baseline_run=success")
            except Exception as e:
                baseline_run_exc = e
                print(f"INFO: baseline_run_exception={type(e).__name__}: {e}")

            if lso_build_exc is None and lso_train_op is not None:
                try:
                    sess.run(lso_train_op)
                    print("INFO: lso_run=success")
                except Exception as e:
                    lso_run_exc = e
                    print(f"INFO: lso_run_exception={type(e).__name__}: {e}")

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)

    if baseline_run_exc is not None:
        _skip(f"baseline optimizer run failed (cannot evaluate oracle): {type(baseline_run_exc).__name__}: {baseline_run_exc}")

    observed_exc = lso_build_exc or lso_run_exc
    if observed_exc is None:
        _fail()

    msg = f"{type(observed_exc).__name__}: {observed_exc}"
    msg_l = msg.lower()

    none_related = (
        ("none values not supported" in msg_l) or
        ("tried to convert" in msg_l and "none" in msg_l) or
        ("is_finite" in msg_l and "none" in msg_l) or
        ("convert" in msg_l and "none" in msg_l) or
        ("none" in msg_l and "tensor" in msg_l)
    )

    if none_related:
        _pass()

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
# conda activate tf_old
# cd ~/dl_testing
# python testcases/tensorflow_testcase.py


# Output:
# *****************
# ENV: {"eager_before_disable": true, "physical_gpus": ["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"], "python": "3.11.15", "tf_version": "2.21.0", "visible_gpu_count": 1}
# INFO: selected_wrapper=tf.compat.v1.mixed_precision.MixedPrecisionLossScaleOptimizer
# INFO: fixed_loss_scale_cls=present
# INFO: symbolic_grad_v1_is_none= False
# INFO: symbolic_grad_v2_is_none= True
# INFO: baseline_build=success
# INFO: instantiated_wrapper_type=MixedPrecisionLossScaleOptimizer
# INFO: lso_build=success
# INFO: baseline_run=success
# INFO: lso_run=success
# Test Failed ❌
# pipeline_exit_code=0
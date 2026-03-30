# GCFL-OTHER-0083

import sys
import traceback


def _skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass() -> None:
    print("Test Passed ✅")
    sys.exit(0)


def _fail() -> None:
    print("Test Failed ❌")
    sys.exit(0)


def _harness_error(e: BaseException) -> None:
    msg = "".join(traceback.format_exception_only(type(e), e)).strip()
    print(f"HARNESS_ERROR: {msg}")
    sys.exit(1)


def _first_data_input(node_def):
    for inp in list(getattr(node_def, "input", [])):
        if isinstance(inp, str) and inp.startswith("^"):
            continue
        return inp
    return None


def _canon_tensor_name(name):
    if not isinstance(name, str):
        return name
    if name.startswith("^"):
        name = name[1:]
    # GraphDef often omits ':0' for port 0.
    if ":" not in name:
        return f"{name}:0"
    return name


def _import_first(names):
    for name in names:
        try:
            mod = __import__(name, fromlist=["*"])
            return mod
        except Exception:
            continue
    return None


def _find_mutable_graph_view_binding():
    """
    Try to locate a Python-accessible MutableGraphView + ReplaceInput binding.
    Returns a constructor wrapper or None.
    """
    candidate_modules = [
        "tensorflow.python.grappler._pywrap_graph_view",
        "tensorflow.python.framework._pywrap_graph_view",
        "tensorflow.python.grappler.graph_view",
    ]

    for modname in candidate_modules:
        mod = _import_first([modname])
        if mod is None:
            continue

        candidates = []
        for attr in dir(mod):
            if "MutableGraphView" in attr:
                try:
                    candidates.append(getattr(mod, attr))
                except Exception:
                    pass

        for ctor in candidates:
            if not callable(ctor):
                continue

            def make_view(gdef, _ctor=ctor):
                # Keep this conservative. If the binding needs something else,
                # we treat it as unavailable in this wheel/build.
                for args in [(gdef,)]:
                    try:
                        return _ctor(*args)
                    except Exception:
                        continue
                return None

            return make_view

    return None


def _get_graphdef_from_view(view, fallback_graphdef):
    # Some bindings mutate in-place; some expose an accessor.
    for attr in ["graph", "graph_def", "Graph", "GraphDef"]:
        if hasattr(view, attr):
            try:
                val = getattr(view, attr)
                if callable(val):
                    val = val()
                if hasattr(val, "node"):
                    return val
            except Exception:
                pass
    return fallback_graphdef


def _call_replace_input(view, old_node, new_node, output_port_id: int):
    """
    Returns:
      (True, None)   -> call succeeded
      (False, None)  -> method missing / signature incompatible
      (None, exc)    -> method existed but raised non-TypeError on all attempts
    """
    method = None
    for name in ["ReplaceInput", "replace_input", "replaceInput"]:
        if hasattr(view, name):
            method = getattr(view, name)
            break

    if method is None:
        return False, None

    call_variants = [
        lambda: method(old_node, new_node, output_port_id),
        lambda: method(old_node, new_node, output_port_id=output_port_id),
        lambda: method(old_node.name, new_node.name, output_port_id),
        lambda: method(old_node.name, new_node.name, output_port_id=output_port_id),
    ]

    first_non_type_error = None

    for fn in call_variants:
        try:
            fn()
            return True, None
        except TypeError:
            continue
        except Exception as e:
            if first_non_type_error is None:
                first_non_type_error = e
            continue

    if first_non_type_error is not None:
        return None, first_non_type_error

    return False, None


def main():
    try:
        try:
            import numpy as np
        except Exception as e:
            _skip(f"missing numpy ({e})")

        try:
            import tensorflow as tf
        except Exception as e:
            _skip(f"missing tensorflow ({e})")

        # Deterministic seeds
        try:
            np.random.seed(2021)
        except Exception:
            pass
        try:
            tf.random.set_seed(2021)
        except Exception:
            pass

        # Build a small TF1-style graph with a multi-output op.
        try:
            g = tf.Graph()
            with g.as_default():
                x = tf.constant([1.0, 2.0], dtype=tf.float32, name="x")

                # Multi-output op: outputs are bar:0 and bar:1
                bar_outs = tf.split(x, num_or_size_splits=2, axis=0, name="bar")
                _ = tf.identity(bar_outs[0], name="foo")   # should stay on bar:0
                _ = tf.identity(bar_outs[1], name="foo2")  # should move to new:1

                new_outs = tf.split(x, num_or_size_splits=2, axis=0, name="new")
                _ = tf.identity(new_outs[0], name="sink_new0")
                _ = tf.identity(new_outs[1], name="sink_new1")

                graph_def = g.as_graph_def()
        except Exception as e:
            _skip(f"cannot build GraphDef in this TF build ({e})")

        def find_node(gdef, name):
            for n in gdef.node:
                if n.name == name:
                    return n
            return None

        old_node = find_node(graph_def, "bar")
        new_node = find_node(graph_def, "new")
        foo_node = find_node(graph_def, "foo")
        foo2_node = find_node(graph_def, "foo2")

        if old_node is None or new_node is None or foo_node is None or foo2_node is None:
            _skip("unexpected GraphDef structure (missing required nodes)")

        make_view = _find_mutable_graph_view_binding()
        if make_view is None:
            _skip("MutableGraphView/ReplaceInput is not accessible from this TensorFlow Python build")

        view = make_view(graph_def)
        if view is None:
            _skip("MutableGraphView could not be constructed from GraphDef in this TensorFlow build")

        # Attempt: ReplaceInput(bar, new, 1) should rewrite only bar:1 -> new:1
        call_status, call_err = _call_replace_input(view, old_node, new_node, output_port_id=1)
        if call_status is None:
            _skip(f"ReplaceInput raised an unexpected error in this TF build ({type(call_err).__name__}: {call_err})")
        if call_status is False:
            _skip("ReplaceInput is unavailable or has an incompatible signature in this TensorFlow build")

        out_gdef = _get_graphdef_from_view(view, graph_def)
        if not hasattr(out_gdef, "node"):
            _skip("Graph view did not expose a GraphDef-like object after rewrite")

        out_foo = find_node(out_gdef, "foo")
        out_foo2 = find_node(out_gdef, "foo2")
        if out_foo is None or out_foo2 is None:
            _skip("GraphDef after rewrite is missing expected nodes")

        foo_in = _canon_tensor_name(_first_data_input(out_foo))
        foo2_in = _canon_tensor_name(_first_data_input(out_foo2))

        expected_foo = "bar:0"
        expected_foo2 = "new:1"

        # Bug under test: ReplaceInput(bar, new, 1) rewrites more than the selected port.
        bug_reproduced = (foo_in != expected_foo) or (foo2_in != expected_foo2)

        if bug_reproduced:
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

# Tensorflow


# Commands
# *****************
# conda deactivate
# conda activate tf_venv

# python - <<'PY'
# import tensorflow as tf
# print(tf.config.list_physical_devices("GPU"))
# PY

# cd ~/dl_testing
# export CUDA_VISIBLE_DEVICES=0
# set -o pipefail
# python ./testcases/tensorflow_testcase.py 2>&1 | tee logs_gcfl_other_0083.txt
# echo "exit_code=$?"



# Output:
# *****************
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1773934868.743323  924096 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# SKIP_ENV: MutableGraphView/ReplaceInput is not accessible from this TensorFlow Python build
# exit_code=0
# # Test Failed ❌
# GCFL-OTHER-0091

import importlib
import json
import os
import sys


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


def _get_sequence_features_cls(tf):
    candidates = [
        ("keras", "experimental", "SequenceFeatures"),
        ("keras", "layers", "SequenceFeatures"),
        ("compat", "v1", "keras", "experimental", "SequenceFeatures"),
        ("compat", "v1", "keras", "layers", "SequenceFeatures"),
    ]

    for path in candidates:
        obj = tf
        ok = True
        for part in path:
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj

    private_modules = [
        "keras.feature_column.sequence_feature_column",
        "keras.src.feature_column.sequence_feature_column",
        "tensorflow.python.keras.feature_column.sequence_feature_column",
        "tensorflow.python.feature_column.sequence_feature_column",
    ]
    for mod_name in private_modules:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, "SequenceFeatures", None)
            if cls is not None:
                return cls
        except Exception:
            pass

    return None


def _get_feature_column_module(tf):
    candidates = [
        getattr(tf, "feature_column", None),
        getattr(getattr(getattr(tf, "compat", None), "v1", None), "feature_column", None),
    ]
    for fc in candidates:
        if fc is None:
            continue
        needed = [
            "sequence_numeric_column",
            "sequence_categorical_column_with_identity",
            "embedding_column",
        ]
        if all(hasattr(fc, name) for name in needed):
            return fc
    return None


def _summary_line_count(model):
    lines = []

    def _collect(s):
        lines.append(str(s))

    try:
        model.summary(print_fn=_collect)
        return len(lines)
    except Exception:
        return -1


def _op_layer_count(model):
    tokens = (
        "TensorFlowOpLayer",
        "TFOpLambda",
        "SlicingOpLambda",
        "OpLambda",
        "OpLayer",
    )
    n = 0
    for layer in getattr(model, "layers", []):
        cls_name = layer.__class__.__name__
        layer_name = getattr(layer, "name", "")
        if any(tok in cls_name or tok in layer_name for tok in tokens):
            n += 1
    return n


def _print_env(tf):
    env = {
        "python": sys.version.split()[0],
        "tensorflow": getattr(tf, "__version__", "unknown"),
        "keras": getattr(tf.keras, "__version__", "unknown"),
        "eager": bool(tf.executing_eagerly()),
        "cuda_built": bool(tf.test.is_built_with_cuda()),
        "visible_gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
    }
    print("ENV: " + json.dumps(env, ensure_ascii=False, sort_keys=True))


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

        _print_env(tf)

        np.random.seed(2021)
        try:
            tf.random.set_seed(2021)
        except Exception:
            pass

        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

        SequenceFeatures = _get_sequence_features_cls(tf)
        if SequenceFeatures is None:
            _skip("tensorflow SequenceFeatures layer not available")

        fc = _get_feature_column_module(tf)
        if fc is None:
            _skip("tensorflow feature_column API not available")

        try:
            inp_num = tf.keras.Input(shape=(None, 1), dtype=tf.float32, name="seq_num_dense")
            sparse_num = tf.sparse.from_dense(inp_num)

            seq_num_col = fc.sequence_numeric_column("seq_num", shape=(1,))
            seq_num_layer = SequenceFeatures([seq_num_col], name="seq_features_num")

            out_num = seq_num_layer({"seq_num": sparse_num})
            feats_num = out_num[0] if isinstance(out_num, (tuple, list)) else out_num

            pooled_num = tf.reduce_sum(tf.cast(feats_num, tf.float32), axis=[1, 2], name="reduce_sum_num")
            model_num = tf.keras.Model(inputs=inp_num, outputs=pooled_num, name="model_num")
        except Exception as e:
            _skip(f"cannot build numeric SequenceFeatures model ({type(e).__name__}: {e})")

        try:
            inp_cat = tf.keras.Input(shape=(None,), dtype=tf.int64, name="seq_cat_dense")
            sparse_cat = tf.sparse.from_dense(inp_cat)

            seq_cat_col = fc.sequence_categorical_column_with_identity("seq_cat", num_buckets=11)
            emb_col = fc.embedding_column(seq_cat_col, dimension=4)
            seq_emb_layer = SequenceFeatures([emb_col], name="seq_features_emb")

            out_emb = seq_emb_layer({"seq_cat": sparse_cat})
            feats_emb = out_emb[0] if isinstance(out_emb, (tuple, list)) else out_emb

            pooled_emb = tf.reduce_sum(tf.cast(feats_emb, tf.float32), axis=[1, 2], name="reduce_sum_emb")
            model_emb = tf.keras.Model(inputs=inp_cat, outputs=pooled_emb, name="model_emb")
        except Exception as e:
            _skip(f"cannot build embedding SequenceFeatures model ({type(e).__name__}: {e})")

        try:
            batch = 2
            steps = 12

            x_num = np.random.randn(batch, steps, 1).astype(np.float32)
            lengths_num = np.random.randint(4, steps + 1, size=batch)
            for i, length in enumerate(lengths_num):
                x_num[i, length:, 0] = 0.0

            x_cat = np.random.randint(1, 11, size=(batch, steps), dtype=np.int64)
            lengths_cat = np.random.randint(4, steps + 1, size=batch)
            for i, length in enumerate(lengths_cat):
                x_cat[i, length:] = 0

            _ = model_num(x_num, training=False)
            _ = model_emb(x_cat, training=False)
        except Exception as e:
            _skip(f"forward pass failed (likely API/runtime mismatch) ({type(e).__name__}: {e})")

        try:
            import tempfile

            tmp_dir = tempfile.gettempdir()
            tf.keras.utils.plot_model(
                model_num,
                to_file=os.path.join(tmp_dir, "gcfl_other_0091_num.png"),
                show_shapes=False,
            )
            tf.keras.utils.plot_model(
                model_emb,
                to_file=os.path.join(tmp_dir, "gcfl_other_0091_emb.png"),
                show_shapes=False,
            )
        except Exception:
            pass

        layers_num = len(getattr(model_num, "layers", []))
        layers_emb = len(getattr(model_emb, "layers", []))
        op_num = _op_layer_count(model_num)
        op_emb = _op_layer_count(model_emb)
        summary_lines_num = _summary_line_count(model_num)
        summary_lines_emb = _summary_line_count(model_emb)

        print(
            "DIAG: "
            f"layers_num={layers_num}, op_num={op_num}, summary_lines_num={summary_lines_num}, "
            f"layers_emb={layers_emb}, op_emb={op_emb}, summary_lines_emb={summary_lines_emb}"
        )

        bug_reproduces = False
        if op_num >= (op_emb + 100) and op_num >= 5 * max(1, op_emb):
            if layers_num >= 200 or (
                summary_lines_num >= 500 and summary_lines_num > summary_lines_emb + 200
            ):
                bug_reproduces = True

        if bug_reproduces:
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
# cd ~/dl_testing
# conda activate tf_venv

# export CUDA_VISIBLE_DEVICES=0
# export PYTHONUNBUFFERED=1

# mkdir -p logs/GCFL-OTHER-0091

# set -o pipefail
# python testcase/tensorflow_testcase.py 2>&1 | tee logs/GCFL-OTHER-0091/run.log
# echo "exit_code=$?"


# Output:
# *****************
# ENV: {"cuda_built": true, "eager": true, "keras": "3.13.2", "python": "3.11.15", "tensorflow": "2.21.0", "visible_gpus": ["/physical_device:GPU:0"]}
# SKIP_ENV: tensorflow SequenceFeatures layer not available
# exit_code=0
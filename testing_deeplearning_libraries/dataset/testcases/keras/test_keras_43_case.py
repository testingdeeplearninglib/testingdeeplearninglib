# GCFL-OTHER-0043

import os
import sys
import re
import io
import json
import time
import random
import warnings
import argparse
import subprocess
from typing import Dict, Any, List, Tuple

def _skip(reason: str) -> None:
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)

def _warn(msg: str) -> None:
    print(f"WARN: {msg}")

def _numel(shape) -> int:
    prod = 1
    for d in list(shape):
        if d is None:
            raise ValueError(f"Encountered dynamic/None dimension in weight shape: {shape}")
        prod *= int(d)
    return int(prod)

def _capture_summary(model) -> str:
    buf = io.StringIO()
    model.summary(print_fn=lambda s: buf.write(str(s) + "\n"))
    return buf.getvalue()

def _parse_param_line(text: str, key: str):
    # matches both "Trainable params: 68" and " Trainable params: 68"
    pattern = rf"^{re.escape(key)}:\s*([0-9][0-9,]*)\b"
    for line in text.splitlines():
        m = re.search(pattern, line.strip())
        if m:
            return int(m.group(1).replace(",", ""))
    return None

def _compute_counts(model) -> Dict[str, int]:
    # Deduplicate weights by object id to avoid rare double-counting corner cases.
    def uniq(ws):
        seen = set()
        out = []
        for w in ws:
            i = id(w)
            if i not in seen:
                seen.add(i)
                out.append(w)
        return out

    tw = uniq(list(getattr(model, "trainable_weights", [])))
    ntw = uniq(list(getattr(model, "non_trainable_weights", [])))

    trainable = sum(_numel(w.shape) for w in tw)
    non_trainable = sum(_numel(w.shape) for w in ntw)
    total = trainable + non_trainable
    return {"trainable": trainable, "non_trainable": non_trainable, "total": total}

def _summary_counts(model) -> Dict[str, Any]:
    s = _capture_summary(model)
    st = _parse_param_line(s, "Trainable params")
    snt = _parse_param_line(s, "Non-trainable params")
    stot = _parse_param_line(s, "Total params")
    return {"trainable": st, "non_trainable": snt, "total": stot, "text": s}

def _assert_summary_matches(model, tag: str) -> Tuple[bool, Dict[str, Any]]:
    cc = _compute_counts(model)
    sc = _summary_counts(model)

    suspicious = False
    reasons = []

    # Hard requirements: must parse trainable & total
    if sc["trainable"] is None or sc["total"] is None:
        suspicious = True
        reasons.append("summary missing Trainable/Total params lines (parse failed)")

    # If parsed, compare.
    if sc["trainable"] is not None and sc["trainable"] != cc["trainable"]:
        suspicious = True
        reasons.append(f"summary trainable != computed trainable ({sc['trainable']} != {cc['trainable']})")
    if sc["non_trainable"] is not None and sc["non_trainable"] != cc["non_trainable"]:
        suspicious = True
        reasons.append(f"summary non-trainable != computed non-trainable ({sc['non_trainable']} != {cc['non_trainable']})")
    if sc["total"] is not None and sc["total"] != cc["total"]:
        suspicious = True
        reasons.append(f"summary total != computed total ({sc['total']} != {cc['total']})")

    details = {"tag": tag, "computed": cc, "summary": {k: sc[k] for k in ["trainable","non_trainable","total"]}}
    if suspicious:
        details["reasons"] = reasons
        details["summary_text"] = sc["text"]
    return suspicious, details

def _build_and_call(model, keras, np, input_shape=(1, 10), dtype="int32"):
    try:
        x = keras.ops.zeros(input_shape, dtype=dtype)
    except Exception:
        x = np.zeros(input_shape, dtype=dtype)
    _ = model(x)
    return x

def _make_weights_for_simple_model(np, seed: int) -> Dict[str, List]:
    rs = np.random.RandomState(seed)
    # embedding: (100,16)
    emb = rs.randn(100, 16).astype("float32") * 0.02
    # dense: kernel (16,4), bias (4,)
    k = rs.randn(16, 4).astype("float32") * 0.02
    b = rs.randn(4).astype("float32") * 0.01
    return {"fixed_embedding": [emb], "head": [k, b]}

def _make_weights_for_bn_model(np, seed: int) -> Dict[str, List]:
    rs = np.random.RandomState(seed)
    # dense1: (16,16) + (16,)
    k1 = rs.randn(16, 16).astype("float32") * 0.02
    b1 = rs.randn(16).astype("float32") * 0.01
    # BN: gamma/beta (16,), moving_mean/var (16,)
    gamma = rs.randn(16).astype("float32") * 0.02
    beta = rs.randn(16).astype("float32") * 0.02
    mm = rs.randn(16).astype("float32") * 0.02
    mv = (rs.rand(16).astype("float32") * 0.5 + 0.5)
    # head: (16,4) + (4,)
    k2 = rs.randn(16, 4).astype("float32") * 0.02
    b2 = rs.randn(4).astype("float32") * 0.01
    return {"dense1": [k1, b1], "bn": [gamma, beta, mm, mv], "head": [k2, b2]}

def _set_layer_weights(model, weights_by_layer: Dict[str, List]):
    for name, ws in weights_by_layer.items():
        layer = model.get_layer(name)
        layer.set_weights(ws)

def _to_numpy(keras, x):
    try:
        return keras.ops.convert_to_numpy(x)
    except Exception:
        # fallback for older builds
        return x.numpy()

def _scenario_0043_baseline(keras, np) -> Tuple[bool, Dict[str, Any]]:
    # Original scenario: embedding trainable=False should be counted as non-trainable.
    inputs = keras.Input(shape=(10,), dtype="int32", name="tokens")
    emb = keras.layers.Embedding(100, 16, trainable=False, name="fixed_embedding")(inputs)
    x = keras.layers.GlobalAveragePooling1D(name="avg_pool")(emb)
    out = keras.layers.Dense(4, name="head")(x)
    model = keras.Model(inputs, out, name="s0043_baseline")

    _build_and_call(model, keras, np)

    # sanity: embedding frozen
    if model.get_layer("fixed_embedding").trainable is not False:
        return True, {"tag": "S0_baseline", "reasons": ["embedding layer not trainable=False (unexpected)"]}

    suspicious, details = _assert_summary_matches(model, "S0_baseline")
    return suspicious, details

def _scenario_bn_moving_stats(keras, np) -> Tuple[bool, Dict[str, Any]]:
    # BN has non-trainable moving mean/var. Summary must count them as non-trainable.
    inputs = keras.Input(shape=(16,), dtype="float32", name="x")
    x = keras.layers.Dense(16, name="dense1")(inputs)
    x = keras.layers.BatchNormalization(name="bn")(x)
    out = keras.layers.Dense(4, name="head")(x)
    model = keras.Model(inputs, out, name="s_bn_stats")

    _build_and_call(model, keras, np, input_shape=(2, 16), dtype="float32")

    suspicious, details = _assert_summary_matches(model, "S1_bn_moving_stats")
    return suspicious, details

def _scenario_nested_freeze(keras, np) -> Tuple[bool, Dict[str, Any]]:
    # Nested model where inner submodel has frozen embedding.
    inp = keras.Input(shape=(10,), dtype="int32", name="tokens_in")
    x = keras.layers.Embedding(100, 16, trainable=False, name="fixed_embedding")(inp)
    x = keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)
    sub = keras.Model(inp, x, name="submodel_frozen")

    top_in = keras.Input(shape=(10,), dtype="int32", name="tokens_top")
    z = sub(top_in)
    out = keras.layers.Dense(4, name="head")(z)
    model = keras.Model(top_in, out, name="s_nested")

    _build_and_call(model, keras, np)

    suspicious, details = _assert_summary_matches(model, "S2_nested_freeze")
    return suspicious, details

def _scenario_toggle_trainable_after_build(keras, np) -> Tuple[bool, Dict[str, Any]]:
    # Toggle trainable flag AFTER variables exist; summary must update accordingly.
    inputs = keras.Input(shape=(10,), dtype="int32", name="tokens")
    emb_layer = keras.layers.Embedding(100, 16, trainable=True, name="fixed_embedding")
    emb = emb_layer(inputs)
    x = keras.layers.GlobalAveragePooling1D(name="avg_pool")(emb)
    out = keras.layers.Dense(4, name="head")(x)
    model = keras.Model(inputs, out, name="s_toggle")

    _build_and_call(model, keras, np)

    # Now freeze embedding after build
    emb_layer.trainable = False

    suspicious, details = _assert_summary_matches(model, "S3_toggle_trainable_after_build")
    return suspicious, details

def _scenario_shared_layer_counting(keras, np) -> Tuple[bool, Dict[str, Any]]:
    # Shared layer used twice should not double-count weights.
    x0 = keras.Input(shape=(16,), dtype="float32", name="x0")
    x1 = keras.Input(shape=(16,), dtype="float32", name="x1")
    shared = keras.layers.Dense(8, name="shared_dense")

    y0 = shared(x0)
    y1 = shared(x1)
    y = keras.layers.Add(name="add")([y0, y1])
    out = keras.layers.Dense(4, name="head")(y)
    model = keras.Model([x0, x1], out, name="s_shared")

    # build
    _ = model([keras.ops.zeros((2, 16), dtype="float32"), keras.ops.zeros((2, 16), dtype="float32")])

    suspicious, details = _assert_summary_matches(model, "S4_shared_layer_counting")
    return suspicious, details

def _scenario_output_signature(keras, np, seed: int) -> Dict[str, Any]:
    # Deterministic weights -> deterministic output vector (small) for diffing across backends.
    inputs = keras.Input(shape=(10,), dtype="int32", name="tokens")
    emb = keras.layers.Embedding(100, 16, trainable=False, name="fixed_embedding")(inputs)
    x = keras.layers.GlobalAveragePooling1D(name="avg_pool")(emb)
    out = keras.layers.Dense(4, name="head")(x)
    model = keras.Model(inputs, out, name="s_out_sig")

    _build_and_call(model, keras, np)

    w = _make_weights_for_simple_model(np, seed)
    _set_layer_weights(model, w)

    # fixed input
    rs = np.random.RandomState(seed)
    tokens = rs.randint(0, 100, size=(2, 10), dtype="int32")
    y = model(tokens, training=False)
    y_np = _to_numpy(keras, y).astype("float32")

    return {
        "tag": "S5_output_signature",
        "seed": seed,
        "shape": list(y_np.shape),
        "vals": [float(v) for v in y_np.reshape(-1)[:8]],  # keep it tiny
        "max_abs": float(np.max(np.abs(y_np))),
        "has_nan": bool(np.isnan(y_np).any()),
        "has_inf": bool(np.isinf(y_np).any()),
    }

def _run_single_backend(iterations: int, do_diff_sig: bool) -> Dict[str, Any]:
    # must not import keras before KERAS_BACKEND is set
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore")

    backend = os.environ.get("KERAS_BACKEND", "").strip().lower()
    if backend not in ("tensorflow", "jax", "torch"):
        _skip(f"Set KERAS_BACKEND to one of: tensorflow, jax, torch (got {backend!r})")

    try:
        import numpy as np
    except Exception as e:
        _skip(f"Missing numpy: {e}")

    try:
        import keras
    except Exception as e:
        _skip(f"Unable to import keras: {e}")

    # quick backend sanity: avoid silently running legacy keras
    kv = getattr(keras, "__version__", "")
    if not str(kv).startswith("3."):
        _skip(f"Expected standalone keras 3.x, got {kv!r}")

    # set seeds
    SEED = 2021
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        if backend == "tensorflow":
            import tensorflow as tf
            tf.random.set_seed(SEED)
        elif backend == "jax":
            import jax  # noqa
    except Exception:
        pass

    suspicious_hits: List[Dict[str, Any]] = []
    scenarios = [
        _scenario_0043_baseline,
        _scenario_bn_moving_stats,
        _scenario_nested_freeze,
        _scenario_toggle_trainable_after_build,
        _scenario_shared_layer_counting,
    ]

    # Run deterministic base scenarios once
    for fn in scenarios:
        try:
            sus, det = fn(keras, np)
            if sus:
                suspicious_hits.append(det)
        except Exception as e:
            suspicious_hits.append({"tag": getattr(fn, "__name__", "scenario"), "reasons": [f"EXCEPTION: {e}"]})

    # Fuzz loop: randomize some toggles to try and shake out corner cases
    for i in range(max(0, iterations)):
        try:
            # random toggle: freeze/unfreeze BN and/or freeze after build
            use_bn = (i % 2 == 0)
            freeze_after = (i % 3 == 0)

            inp = keras.Input(shape=(16,), dtype="float32", name=f"fx{i}")
            x = keras.layers.Dense(16, name=f"dense1_{i}")(inp)
            if use_bn:
                bn = keras.layers.BatchNormalization(name=f"bn_{i}")
                x = bn(x)
            head = keras.layers.Dense(4, name=f"head_{i}")
            out = head(x)
            model = keras.Model(inp, out, name=f"fuzz_{i}")

            _build_and_call(model, keras, np, input_shape=(2, 16), dtype="float32")

            if use_bn and freeze_after:
                model.get_layer(f"bn_{i}").trainable = False

            sus, det = _assert_summary_matches(model, f"FZ_{i}_bn{int(use_bn)}_freezeAfter{int(freeze_after)}")
            if sus:
                suspicious_hits.append(det)
                break  # stop early on first strong hit
        except Exception as e:
            suspicious_hits.append({"tag": f"FZ_{i}", "reasons": [f"EXCEPTION: {e}"]})
            break

    out_sig = None
    if do_diff_sig:
        try:
            out_sig = _scenario_output_signature(keras, np, seed=1337)
        except Exception as e:
            out_sig = {"tag": "S5_output_signature", "error": str(e)}

    result = {
        "backend": backend,
        "keras_version": getattr(keras, "__version__", "unknown"),
        "iterations": iterations,
        "suspicious_count": len(suspicious_hits),
        "suspicious": suspicious_hits[:3],  # keep small
        "output_signature": out_sig,
    }
    return result

def _orchestrate(backends: List[str], iterations: int, do_diff_sig: bool) -> None:
    # IMPORTANT: do not import keras here
    me = os.path.abspath(__file__)
    py = sys.executable

    results = {}
    for be in backends:
        env = os.environ.copy()
        env["KERAS_BACKEND"] = be
        # reduce multi-gpu variability
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")

        cmd = [py, me, "--single-backend", f"--iterations={iterations}"]
        if do_diff_sig:
            cmd.append("--diff-sig")

        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"\n===== {be} OUTPUT =====")
        print(p.stdout)

        # Extract RESULT_JSON line
        rj = None
        for line in p.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                rj = line[len("RESULT_JSON:"):].strip()
                break
        if not rj:
            results[be] = {"backend": be, "error": "missing RESULT_JSON", "raw_tail": p.stdout.splitlines()[-50:]}
            continue
        try:
            results[be] = json.loads(rj)
        except Exception as e:
            results[be] = {"backend": be, "error": f"json parse failed: {e}", "raw": rj}

    # Cross-backend diff on output signature (only if both present)
    suspicious_cross = []
    if do_diff_sig and len(results) >= 2:
        # compare pairwise (tensorflow vs jax)
        if "tensorflow" in results and "jax" in results:
            a = results["tensorflow"].get("output_signature")
            b = results["jax"].get("output_signature")
            if isinstance(a, dict) and isinstance(b, dict) and ("vals" in a) and ("vals" in b):
                import math
                av = a["vals"]
                bv = b["vals"]
                if len(av) == len(bv):
                    diffs = [abs(float(x) - float(y)) for x, y in zip(av, bv)]
                    maxdiff = max(diffs) if diffs else 0.0
                    # only flag if it’s clearly large (avoid false positives)
                    if (a.get("has_nan") or a.get("has_inf") or b.get("has_nan") or b.get("has_inf")):
                        suspicious_cross.append({"tag": "X_backend_nan_inf", "tf": a, "jax": b})
                    elif maxdiff > 1e-3:
                        suspicious_cross.append({"tag": "X_backend_large_mismatch", "max_abs_diff_first8": maxdiff, "tf": av, "jax": bv})

    any_suspicious = False
    for be, r in results.items():
        if isinstance(r, dict) and r.get("suspicious_count", 0) > 0:
            any_suspicious = True
    if suspicious_cross:
        any_suspicious = True

    report = {"results": results, "cross_backend": suspicious_cross}
    print("\n===== FINAL REPORT =====")
    print(json.dumps(report, indent=2, sort_keys=True))

    if any_suspicious:
        print("SUSPICIOUS ✅")
    else:
        print("NO SUSPICIOUS ✅")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="", help="Comma list, e.g. tensorflow,jax")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--diff-sig", action="store_true", help="Cross-backend output signature check")
    ap.add_argument("--single-backend", action="store_true", help="Run inside one backend process")
    args = ap.parse_args()

    if args.single_backend:
        r = _run_single_backend(iterations=args.iterations, do_diff_sig=args.diff_sig)
        print("RESULT_JSON:", json.dumps(r, sort_keys=True))
        # user-friendly top-level line
        if r.get("suspicious_count", 0) > 0:
            print("SUSPICIOUS ✅")
        else:
            print("NO SUSPICIOUS ✅")
        return

    if args.backends.strip():
        bes = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
        _orchestrate(bes, iterations=args.iterations, do_diff_sig=args.diff_sig)
        return

    # Default: just run current backend
    r = _run_single_backend(iterations=args.iterations, do_diff_sig=args.diff_sig)
    print(json.dumps(r, indent=2, sort_keys=True))
    if r.get("suspicious_count", 0) > 0:
        print("SUSPICIOUS ✅")
    else:
        print("NO SUSPICIOUS ✅")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"HARNESS_ERROR: {e}")
        sys.exit(1)




# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Keras


# Commands
# *****************
# conda activate keras_venv
# export CUDA_VISIBLE_DEVICES=0

# # Cross-backend suspicious-case hunt (TF vs JAX) with output signature diff + fuzzing
# python testcases/keras_testcase.py --backends tensorflow,jax --diff-sig --iterations 50 | tee logs/GCFL-OTHER-0043_hunt.log
# echo "exit_code=$?"



# Output:
# *****************
# ===== tensorflow OUTPUT =====
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1772555476.905090  795405 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22446 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:02:00.0, compute capability: 8.6
# RESULT_JSON: {"backend": "tensorflow", "iterations": 50, "keras_version": "3.12.1", "output_signature": {"has_inf": false, "has_nan": false, "max_abs": 0.01979503594338894, "seed": 1337, "shape": [2, 4], "tag": "S5_output_signature", "vals": [-0.015062431804835796, -0.005191624630242586, -0.009971714578568935, -0.01979503594338894, -0.015119682997465134, -0.005599703174084425, -0.01187027059495449, -0.01932559348642826]}, "suspicious": [], "suspicious_count": 0}
# NO SUSPICIOUS 


# ===== jax OUTPUT =====
# WARNING:2026-03-04 00:31:23,319:jax._src.xla_bridge:794: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
# RESULT_JSON: {"backend": "jax", "iterations": 50, "keras_version": "3.12.1", "output_signature": {"has_inf": false, "has_nan": false, "max_abs": 0.01979503594338894, "seed": 1337, "shape": [2, 4], "tag": "S5_output_signature", "vals": [-0.015062431804835796, -0.005191624630242586, -0.009971714578568935, -0.01979503594338894, -0.015119682997465134, -0.005599703174084425, -0.01187027059495449, -0.01932559348642826]}, "suspicious": [], "suspicious_count": 0}
# NO SUSPICIOUS



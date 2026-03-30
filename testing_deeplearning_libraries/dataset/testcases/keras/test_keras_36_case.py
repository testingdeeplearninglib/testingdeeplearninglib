# GCFL-DTYPEPRECI-0036

# GCFL-DTYPEPRECI-0036

import os
import sys
import time

SEED = 1337
MAX_RUNTIME_SEC = 30


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


def _normalize_dtype(dt) -> str:
    if dt is None:
        return "none"
    try:
        s = str(dt)
    except Exception:
        s = repr(dt)
    s = s.strip().lower()
    s = s.replace("dtype(", "").replace(")", "")
    s = s.replace("<dtype: '", "").replace("'>", "")
    for p in ["torch.", "numpy.", "jax.numpy.", "jax.", "tf."]:
        s = s.replace(p, "")
    s = s.replace("<class '", "").replace("'>", "")
    return s


def _sym_dtype(t) -> str:
    try:
        return _normalize_dtype(getattr(t, "dtype", None))
    except Exception:
        return "unknown"


def _exec_dtype(v) -> str:
    try:
        if hasattr(v, "dtype"):
            return _normalize_dtype(v.dtype)
    except Exception:
        pass
    try:
        return _normalize_dtype(type(v))
    except Exception:
        return "unknown"


def _to_numpy_dtype(v):
    try:
        if hasattr(v, "numpy") and callable(v.numpy):
            arr = v.numpy()
            return _normalize_dtype(getattr(arr, "dtype", None))
    except Exception:
        pass
    try:
        if hasattr(v, "__array__"):
            arr = v.__array__()
            return _normalize_dtype(getattr(arr, "dtype", None))
    except Exception:
        pass
    return None


def _import_keras():
    try:
        import keras
        return keras, "keras"
    except Exception:
        pass
    try:
        import keras_core as keras
        return keras, "keras_core"
    except Exception as e:
        raise e


def main():
    start = time.time()
    tried = 0
    skipped = 0
    executed = 0
    sym_ops_built = 0

    first_mismatch = None  # dict

    try:
        try:
            import numpy as np
        except Exception as e:
            _skip(f"numpy not available: {e}")

        try:
            np.random.seed(SEED)
        except Exception:
            pass

        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        backend = os.environ.get("KERAS_BACKEND", "").strip().lower()

        try:
            keras, keras_import = _import_keras()
        except Exception as e:
            _skip(f"keras/keras_core not available: {e}")

        print(f"KERAS_IMPORT: {keras_import}")
        print(f"KERAS_VERSION: {getattr(keras, '__version__', 'unknown')}")
        print(f"KERAS_BACKEND_ENV: {backend}")

        dtype_pairs = [
            ("float16", "float32"),
            ("float32", "float64"),
            ("bfloat16", "float32"),
            ("int32", "float32"),
            ("int64", "float32"),
            ("uint8", "float32"),
        ]

        shapes = [
            (),
            (1,),
            (4,),
            (2, 3),
            (1, 3),
        ]

        broadcast_pairs = [
            ((2, 3), (1, 3)),
            ((2, 3), (2, 1)),
            ((4,), (1,)),
        ]

        def build_np(shape, dtype):
            # Always include batch dim to keep execution consistent.
            if shape == ():
                return np.zeros((1,), dtype=dtype)
            return np.zeros((1,) + tuple(shape), dtype=dtype)

        def check_case(x_shape, y_shape, dx, dy):
            nonlocal tried, skipped, executed, sym_ops_built, first_mismatch
            tried += 1

            # Build symbolic inputs
            try:
                x = keras.Input(shape=x_shape, dtype=dx, name="x")
                y = keras.Input(shape=y_shape, dtype=dy, name="y")
            except Exception:
                skipped += 1
                return

            ops = []
            try:
                ops.append(("add", x + y, y + x))
            except Exception:
                pass
            try:
                ops.append(("mul", x * y, y * x))
            except Exception:
                pass
            try:
                ops.append(("sub", x - y, y - x))
            except Exception:
                pass

            if not ops:
                skipped += 1
                return

            # Concrete inputs
            try:
                in_x = build_np(x_shape, dx)
                in_y = build_np(y_shape, dy)
            except Exception:
                skipped += 1
                return

            for opname, z0, z1 in ops:
                sym_ops_built += 1
                z0_dt = _sym_dtype(z0)
                z1_dt = _sym_dtype(z1)

                # Symbolic operand-order mismatch
                if z0_dt not in ["unknown", "none"] and z1_dt not in ["unknown", "none"] and z0_dt != z1_dt:
                    first_mismatch = {
                        "stage": "symbolic",
                        "op": opname,
                        "x_shape": x_shape,
                        "y_shape": y_shape,
                        "x_dtype": dx,
                        "y_dtype": dy,
                        "z0_dtype": z0_dt,
                        "z1_dtype": z1_dt,
                    }
                    return

                # Build models
                try:
                    m0 = keras.Model((x, y), z0)
                    m1 = keras.Model((x, y), z1)
                except Exception:
                    skipped += 1
                    continue

                # Execute both with same feeds
                try:
                    v0 = m0((in_x, in_y))
                    v1 = m1((in_x, in_y))
                    executed += 1
                except Exception:
                    skipped += 1
                    continue

                v0_dt = _exec_dtype(v0)
                v1_dt = _exec_dtype(v1)

                # Execution operand-order mismatch
                if v0_dt not in ["unknown", "none"] and v1_dt not in ["unknown", "none"] and v0_dt != v1_dt:
                    first_mismatch = {
                        "stage": "exec",
                        "op": opname,
                        "x_shape": x_shape,
                        "y_shape": y_shape,
                        "x_dtype": dx,
                        "y_dtype": dy,
                        "v0_dtype": v0_dt,
                        "v1_dtype": v1_dt,
                        "v0_numpy": _to_numpy_dtype(v0),
                        "v1_numpy": _to_numpy_dtype(v1),
                    }
                    return

                # Numpy dtype mismatch (extra guard)
                v0_np = _to_numpy_dtype(v0)
                v1_np = _to_numpy_dtype(v1)
                if v0_np and v1_np and v0_np != v1_np:
                    first_mismatch = {
                        "stage": "numpy",
                        "op": opname,
                        "x_shape": x_shape,
                        "y_shape": y_shape,
                        "x_dtype": dx,
                        "y_dtype": dy,
                        "v0_numpy": v0_np,
                        "v1_numpy": v1_np,
                        "v0_dtype": v0_dt,
                        "v1_dtype": v1_dt,
                    }
                    return

        # Run tests
        for dx, dy in dtype_pairs:
            for s in shapes:
                check_case(s, s, dx, dy)
                if first_mismatch:
                    print(f"MISMATCH: {first_mismatch}")
                    _pass()

        for dx, dy in dtype_pairs:
            for xs, ys in broadcast_pairs:
                check_case(xs, ys, dx, dy)
                if first_mismatch:
                    print(f"MISMATCH: {first_mismatch}")
                    _pass()
                check_case(ys, xs, dx, dy)
                if first_mismatch:
                    print(f"MISMATCH: {first_mismatch}")
                    _pass()

        # Summary (useful, does not violate oracle requirement)
        print(f"SUMMARY: tried={tried} sym_ops={sym_ops_built} executed={executed} skipped={skipped}")
        _fail()

    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)
    finally:
        try:
            if (time.time() - start) > MAX_RUNTIME_SEC:
                pass
        except Exception:
            pass


if __name__ == "__main__":
    main()



# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Keras


# Commands
# *****************
# conda activate keras_venv
# export KERAS_BACKEND=tensorflow
# python testcases/keras_testcase.py

# conda activate keras_venv
# export KERAS_BACKEND=jax
# export JAX_ENABLE_X64=1
# python testcases/keras_testcase.py



# Output:
# *****************
# KERAS_IMPORT: keras
# KERAS_VERSION: 3.12.1
# KERAS_BACKEND_ENV: tensorflow
# SUMMARY: tried=66 sym_ops=198 executed=198 skipped=0
# Test Failed ❌


# KERAS_IMPORT: keras
# KERAS_VERSION: 3.12.1
# KERAS_BACKEND_ENV: jax
# SUMMARY: tried=66 sym_ops=198 executed=198 skipped=0
# Test Failed ❌

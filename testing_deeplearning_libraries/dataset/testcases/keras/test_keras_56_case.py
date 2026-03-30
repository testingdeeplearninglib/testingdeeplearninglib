# GCFL-OTHER-0056

import os
import sys
import subprocess
import random
import importlib.util


def _skip(reason: str):
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _pass():
    print("Test Passed ✅")
    sys.exit(0)


def _fail(msg: str = ""):
    print("Test Failed ❌")
    if msg:
        print(msg)
    sys.exit(0)


def _harness_error(e: BaseException):
    print(f"HARNESS_ERROR: {type(e).__name__}: {e}")
    sys.exit(1)


def _set_determinism():
    random.seed(2021)
    try:
        import numpy as np  # type: ignore
        np.random.seed(2021)
    except Exception:
        pass


def _run(code: str, env: dict) -> tuple[int, str]:
    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        return p.returncode, out
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def main():
    # Hard requirement: keras installed
    if importlib.util.find_spec("keras") is None:
        _skip("keras not installed in this env")

    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")

    # Force TF backend so this test is meaningful without torch installed.
    if importlib.util.find_spec("tensorflow") is None:
        _skip("tensorflow not installed; cannot force TF backend")
    env["KERAS_BACKEND"] = "tensorflow"

    # Baseline: import keras should succeed with TF backend.
    baseline = r"""
import sys, traceback
try:
    import keras
    print("KERAS_IMPORT_OK", getattr(keras, "__version__", "unknown"))
    sys.exit(0)
except Exception:
    traceback.print_exc()
    sys.exit(3)
"""
    rc0, out0 = _run(baseline, env)
    if rc0 == 124:
        _skip("baseline timed out importing keras")
    if rc0 != 0:
        _skip("baseline import failed (env broken):\n" + out0[-2000:])

    # Bug probe: block torch imports and import keras
    probe = r"""
import builtins, sys, traceback
_orig = builtins.__import__

def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("blocked torch import for test")
    return _orig(name, globals, locals, fromlist, level)

builtins.__import__ = _blocked

# Prove the block is active
try:
    import torch
    print("TORCH_IMPORTED_UNEXPECTED")
except ImportError as e:
    print("TORCH_BLOCK_OK:", str(e))

try:
    import keras
    sys.exit(0)
except Exception:
    traceback.print_exc()
    sys.exit(3)
"""
    rc1, out1 = _run(probe, env)
    if rc1 == 124:
        _skip("probe timed out")
    if "TORCH_IMPORTED_UNEXPECTED" in out1:
        _skip("torch block failed; probe meaningless")

    bug_signatures = [
        "UnboundLocalError",
        "local variable 'torch' referenced before assignment",
        'local variable "torch" referenced before assignment',
        "NameError",
        "name 'torch' is not defined",
        'name "torch" is not defined',
    ]

    # This testcase is a BUG-DETECTOR: it "passes" only if the buggy unbound-name pattern appears.
    if rc1 == 3 and any(sig in out1 for sig in bug_signatures):
        _pass()

    _fail(f"Backend forced to: tensorflow\nProbe returncode: {rc1}\n--- probe output tail ---\n{out1[-2000:]}")


if __name__ == "__main__":
    try:
        _set_determinism()
        main()
    except SystemExit:
        raise
    except Exception as e:
        _harness_error(e)





# ******************************************************************************
#                                    Result 
# ******************************************************************************

# Keras


# Commands
# *****************
# conda activate keras_nightly
# conda env config vars set KERAS_BACKEND=tensorflow
# conda deactivate
# conda activate keras_nightly

# cd ~/dl_testing
# python testcases/keras_testcase.py | tee keras_gcfl_other_0056.log

# conda create -y -n keras_0056_old python=3.10
# conda activate keras_0056_old

# python -m pip install -U pip setuptools wheel
# python -m pip install "keras==3.10.0" "tensorflow==2.20.*"
# python -m pip uninstall -y torch torchvision torchaudio || true

# cd ~/dl_testing
# python testcases/keras_testcase.py | tee keras_gcfl_other_0056_old.log

# conda create -y -n keras_0056_tf216 python=3.10
# conda activate keras_0056_tf216

# python -m pip install -U pip setuptools wheel
# python -m pip install "tensorflow==2.16.1" "keras==3.0.0"
# python -m pip uninstall -y torch torchvision torchaudio || true

# cd ~/dl_testing
# python testcases/keras_testcase.py | tee keras_gcfl_other_0056_tf216.log


# Output:
# *****************
# Test Failed ❌
# Backend forced to: tensorflow
# Probe returncode: 0
# --- probe output tail ---
# TORCH_BLOCK_OK: blocked torch import for test
# GCFL-DTYPEPRECI-0020

import sys
import os
import json
import tempfile
import random

EXPECTED_SUBSTR = "requires dynamic loss scaling"

_ENGINE = None
_CFG_PATH = None


def _skip(reason: str):
    print(f"SKIP_ENV: {reason}")
    sys.exit(0)


def _cleanup_best_effort():
    global _ENGINE, _CFG_PATH
    try:
        if _ENGINE is not None:
            try:
                _ENGINE.destroy()
            except Exception:
                pass
    except Exception:
        pass

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    try:
        if _CFG_PATH and os.path.exists(_CFG_PATH):
            os.remove(_CFG_PATH)
    except Exception:
        pass


def _pass():
    _cleanup_best_effort()
    print("Test Passed ✅")
    sys.exit(0)


def _fail():
    _cleanup_best_effort()
    print("Test Failed ❌")
    sys.exit(0)


def _seed_all(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _write_json_config(cfg: dict) -> str:
    fd, path = tempfile.mkstemp(prefix="ds_cfg_", suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    return path


def main():
    global _ENGINE, _CFG_PATH

    try:
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            _skip(f"torch missing: {e}")

        try:
            import deepspeed
        except Exception as e:
            _skip(f"deepspeed missing: {e}")

        if not torch.cuda.is_available():
            _skip("CUDA not available")

        _seed_all(1337)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 4)

            def forward(self, x):
                return self.lin(x)

        model = TinyModel().cuda()

        ds_config = {
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True},
            "fp16": {"enabled": False, "loss_scale": 128},
            "loss_scale": 128,
            "optimizer": {"type": "Lamb", "params": {"lr": 1e-3}},
            "zero_optimization": {"stage": 0},
        }

        _CFG_PATH = _write_json_config(ds_config)

        try:
            engine, _, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=list(model.parameters()),
                config=_CFG_PATH,
            )
            _ENGINE = engine
        except BaseException as e:
            if EXPECTED_SUBSTR in str(e).lower():
                _pass()
            print(f"DEBUG_EXCEPTION_INIT: {e}")
            _fail()

        _fail()

    except SystemExit:
        raise
    except BaseException as e:
        _cleanup_best_effort()
        print(f"HARNESS_ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()






# ******************************************************************************
#                                    Result 
# ******************************************************************************

# deepspeed


# Commands
# *****************
# source ~/.venvs/dl_testing/bin/activate
# export TF_CPP_MIN_LOG_LEVEL=2
# PYTHONNOUSERSITE=1 deepspeed --num_gpus=1 testcases/deepspeed_testcase.py



# Output:
# *****************
# Test Passed ✅
# Exception ignored in: <function DeepSpeedEngine.__del__>
# Traceback (most recent call last):
#   File ".../deepspeed/runtime/engine.py", line 565, in __del__
#     self.destroy()
#   File ".../deepspeed/runtime/engine.py", line 570, in destroy
#     if self.is_deepcompile_active():
#   File ".../deepspeed/runtime/engine.py", line 4358, in is_deepcompile_active
#     return self._deepcompile_active
# AttributeError: 'DeepSpeedEngine' object has no attribute '_deepcompile_active'



# ******************************************************************************

# Reported ✅
# Link: 
# https://github.com/deepspeedai/DeepSpeed/issues/7812


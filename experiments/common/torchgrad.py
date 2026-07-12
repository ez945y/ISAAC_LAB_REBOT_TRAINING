"""Loader shim: the autograd gradient path now lives with the production guard
(tools/controll_scripts/safety/torchgrad.py) so ``grad_mode="autograd"`` means
the SAME code in pydam and in the real callback. Kept importable here so the
E3.7 scripts and pydam need no changes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[2] / "tools/controll_scripts/safety/torchgrad.py"
_spec = importlib.util.spec_from_file_location("safety_torchgrad", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

seg_seg_dist_torch = _mod.seg_seg_dist_torch
pred_dist_and_grad = _mod.pred_dist_and_grad

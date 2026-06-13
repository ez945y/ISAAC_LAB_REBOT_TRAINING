# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""診斷：確認啟動後 omni.appwindow / app window 是否就緒。

用法（跟你跑 demo 一樣的方式，不要加 --headless）：
    python scripts/diag_appwindow.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="omni.appwindow diagnostic")
# 先加一個自有參數，與 demo 一致；空 parser 會讓 add_app_launcher_args
# 無法正確掛上 --headless，導致 headless 讀數不可信。
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Auto-configure WebRTC livestream (publicIp + dynamic resize) when --livestream is set.
from livestream_support import apply_livestream_defaults
apply_livestream_defaults(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("=" * 60)
print("[DIAG] headless 設定 :", getattr(args_cli, "headless", "<unset>"))
print("[DIAG] livestream    :", getattr(args_cli, "livestream", "<unset>"))

# 1) omni.appwindow 能不能 import
try:
    import omni.appwindow  # noqa: F401
    print("[DIAG] import omni.appwindow : OK")
    try:
        win = omni.appwindow.get_default_app_window()
        print("[DIAG] get_default_app_window():", win)
    except Exception as exc:  # noqa: BLE001
        print("[DIAG] get_default_app_window() 失敗:", repr(exc))
except Exception as exc:  # noqa: BLE001
    print("[DIAG] import omni.appwindow 失敗:", repr(exc))

# 2) extension manager 裡 omni.appwindow 是否啟用
try:
    import omni.kit.app

    mgr = omni.kit.app.get_app().get_extension_manager()
    exts = mgr.get_extensions()
    appwin = [e for e in exts if "appwindow" in e.get("name", "")]
    print("[DIAG] 名稱含 'appwindow' 的 extension:")
    for e in appwin:
        print("        ", e.get("name"), "enabled=", e.get("enabled"))
    if not appwin:
        print("         <無>")
except Exception as exc:  # noqa: BLE001
    print("[DIAG] 查 extension manager 失敗:", repr(exc))

print("=" * 60)

simulation_app.close()

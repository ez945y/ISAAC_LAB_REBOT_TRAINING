# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Squad Dispatch extension: in-viewport markers + an omni.ui control panel.

Both surfaces are fed from :mod:`runtime` (the sim registers a DispatchHandle).
The viewport overlay (DebugDraw) renders into the streamed viewport, and — because
the WebRTC livestream streams the full Kit UI — the omni.ui panel shows in the
stream too. So the operator drives + watches everything from the stream alone, no
rviz, no second window.
"""

from __future__ import annotations

import time

import carb
import omni.ext
import omni.kit.app
import omni.ui as ui

from .runtime import get_handle

# Distinct per-group colors, matching scripts/11's GROUP_PALETTE.
_PALETTE = [
    (0.20, 0.60, 1.00), (1.00, 0.55, 0.10), (0.30, 0.90, 0.40),
    (0.90, 0.30, 0.90), (0.95, 0.85, 0.15), (0.40, 0.90, 0.90),
]
_ZONE_COLOR = (0.20, 0.80, 0.90, 1.0)
_PANEL_HZ = 4.0  # rebuild the panel at most this often (the overlay redraws every frame)


def _group_rgb(gid: str | None) -> tuple[float, float, float]:
    if not gid:
        return (0.85, 0.85, 0.85)
    digits = "".join(c for c in gid if c.isdigit())
    return _PALETTE[(int(digits) if digits else 0) % len(_PALETTE)]


class SquadDispatchExtension(omni.ext.IExt):
    """Live dispatch console for the Go2 squad."""

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._draw = None
        try:
            from isaacsim.util.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[squad.dispatch] debug_draw unavailable: {exc}")

        self._last_panel = 0.0
        self._root = None
        self._window = None
        # The UI panel needs a UI app (present when streaming the full Kit GUI). If
        # it isn't there (pure headless), skip the panel — the viewport overlay still
        # works on its own.
        try:
            self._window = ui.Window("Squad Dispatch", width=360, height=460)
            with self._window.frame:
                self._root = ui.Frame()
            self._root.set_build_fn(self._build_ui)
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[squad.dispatch] UI panel unavailable (overlay only): {exc}")

        self._sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="squad.dispatch.update")
        )
        carb.log_info("[squad.dispatch] started")

    def on_shutdown(self) -> None:
        self._sub = None
        if self._draw is not None:
            self._draw.clear_lines()
            self._draw.clear_points()
        if self._window is not None:
            self._window.destroy()
            self._window = None
        self._root = None

    # -- per-frame ------------------------------------------------------------

    def _on_update(self, _e) -> None:
        handle = get_handle()
        snap = handle.snapshot() if handle is not None else None
        self._draw_overlay(snap)
        now = time.monotonic()
        if now - self._last_panel >= 1.0 / _PANEL_HZ and self._root is not None:
            self._last_panel = now
            self._root.rebuild()

    def _draw_overlay(self, snap: dict | None) -> None:
        if self._draw is None:
            return
        self._draw.clear_points()
        self._draw.clear_lines()
        if not snap:
            return
        pts: list[tuple[float, float, float]] = []
        cols: list[tuple[float, float, float, float]] = []
        sizes: list[float] = []
        ls, le, lc, lw = [], [], [], []

        for z in snap.get("zones", []):
            pts.append((z["x"], z["y"], 0.08))
            cols.append(_ZONE_COLOR)
            sizes.append(26)

        for d in snap.get("dogs", []):
            r, g, b = _group_rgb(d.get("group"))
            # A group-colored dot floating over each dog (the dog body is already visible).
            pts.append((d["x"], d["y"], 0.75))
            cols.append((r, g, b, 1.0))
            sizes.append(22)
            tx, ty = d.get("tx"), d.get("ty")
            if tx is not None and ty is not None:
                # Target marker + a path line from the dog to it.
                pts.append((tx, ty, 0.12))
                cols.append((r, g, b, 0.7))
                sizes.append(16)
                ls.append((d["x"], d["y"], 0.12))
                le.append((tx, ty, 0.12))
                lc.append((r, g, b, 0.9))
                lw.append(3.0)

        if pts:
            self._draw.draw_points(pts, cols, sizes)
        if ls:
            self._draw.draw_lines(ls, le, lc, lw)

    # -- panel ----------------------------------------------------------------

    def _build_ui(self) -> None:
        handle = get_handle()
        snap = handle.snapshot() if handle is not None else None
        with ui.VStack(spacing=6, height=0):
            if snap is None:
                ui.Label("Waiting for the sim to register its squad…",
                         alignment=ui.Alignment.CENTER)
                return

            sel = snap.get("selected") or "ALL"
            groups = snap.get("groups", {})
            ui.Label(f"Steering: {sel}    Groups: {len(groups)}",
                     style={"font_size": 18})

            # Dispatch targets (named zones) -> steered group.
            ui.Label("Dispatch to zone:", style={"color": 0xFFAAAAAA})
            with ui.HStack(spacing=4, height=0):
                for i, z in enumerate(snap.get("zones", [])):
                    ui.Button(z["name"], clicked_fn=lambda i=i: self._do(lambda h: h.dispatch_zone(i)))

            with ui.HStack(spacing=4, height=0):
                ui.Button("Select ▶", clicked_fn=lambda: self._do(lambda h: h.select_next()))
                ui.Button("Formation", clicked_fn=lambda: self._do(lambda h: h.cycle_formation()))
                ui.Button("Regroup", clicked_fn=lambda: self._do(lambda h: h.cycle_regroup()))
            with ui.HStack(spacing=4, height=0):
                ui.Button("Patrol", clicked_fn=lambda: self._do(lambda h: h.toggle_patrol()))
                ui.Button("Recall", clicked_fn=lambda: self._do(lambda h: h.recall()))
                ui.Button("Halt", clicked_fn=lambda: self._do(lambda h: h.halt()))

            ui.Separator()
            ui.Label("Dogs:", style={"color": 0xFFAAAAAA})
            for d in snap.get("dogs", []):
                tgt = "—" if d.get("tx") is None else f"({d['tx']:+.1f},{d['ty']:+.1f})"
                mark = "✓" if d.get("arrived") else " "
                ui.Label(
                    f"  {d['id']}  {d.get('group') or '-':>3}  "
                    f"({d['x']:+5.1f},{d['y']:+5.1f})  → {tgt}  {mark}",
                    style={"font_size": 14},
                )

    def _do(self, fn) -> None:
        handle = get_handle()
        if handle is None:
            return
        try:
            fn(handle)
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[squad.dispatch] action failed: {exc}")
        if self._root is not None:
            self._root.rebuild()

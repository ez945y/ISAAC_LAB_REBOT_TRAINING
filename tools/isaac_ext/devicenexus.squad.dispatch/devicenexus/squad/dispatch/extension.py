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

import math
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

        # ONLY the per-dog body capsule (real Go2 size ~0.65 x 0.30 m). No group ring,
        # no comfort ring -- just each robot's footprint. The min_dist=0.7 safety
        # distance is the GAP kept between these bodies (not drawn as a bubble).
        safety = snap.get("safety")
        dogs = snap.get("dogs", [])
        if safety and dogs:
            body_half, body_r = 0.22, 0.17
            for d in dogs:
                self._stadium(d["x"], d["y"], d.get("yaw", 0.0), body_half, body_r,
                              (1.0, 0.3, 0.3, 0.9), z=0.05)

    def _circle(self, cx: float, cy: float, radius: float, color, z: float = 0.05,
                n: int = 20, width: int = 2) -> None:
        """Draw a flat ring on the floor as a closed spline (DebugDraw has no circle)."""
        if self._draw is None or radius <= 0.0:
            return
        pts = [(cx + radius * math.cos(2 * math.pi * k / n),
                cy + radius * math.sin(2 * math.pi * k / n), z) for k in range(n)]
        self._draw.draw_lines_spline(pts, color, width, True)

    def _stadium(self, cx: float, cy: float, yaw: float, half: float, radius: float,
                 color, z: float = 0.05, width: int = 1, nc: int = 7) -> None:
        """Draw a capsule outline (a stadium: a segment of half-length ``half`` along
        ``yaw`` swept by ``radius``) as exact straight segments -- a spline bows the
        straight sides inward and reads as two rings."""
        if self._draw is None or radius <= 0.0:
            return
        local = []
        for k in range(nc + 1):                     # front cap, -90deg -> +90deg
            a = -math.pi / 2 + math.pi * k / nc
            local.append((half + radius * math.cos(a), radius * math.sin(a)))
        for k in range(nc + 1):                     # back cap, +90deg -> +270deg
            a = math.pi / 2 + math.pi * k / nc
            local.append((-half + radius * math.cos(a), radius * math.sin(a)))
        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        pts = [(cx + lx * cyaw - ly * syaw, cy + lx * syaw + ly * cyaw, z) for lx, ly in local]
        starts, ends = pts, pts[1:] + pts[:1]       # closed polyline
        self._draw.draw_lines(starts, ends, [color] * len(pts), [float(width)] * len(pts))

    # -- panel ----------------------------------------------------------------

    def _row(self, label: str, btn_text: str, hint: str, on_click, tooltip: str) -> None:
        """One control row: 'what it is' | [button showing current value] | 'what it does'."""
        with ui.HStack(spacing=6, height=0):
            ui.Label(label, width=86, style={"font_size": 14})
            b = ui.Button(btn_text, width=130, clicked_fn=lambda: self._do(on_click))
            try:
                b.set_tooltip(tooltip)
            except Exception:  # noqa: BLE001 -- tooltip is a nicety, never fatal
                pass
            ui.Label(hint, style={"color": 0xFF888888, "font_size": 12})

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
            size = max((len(v) for v in groups.values()), default=0)
            formations = snap.get("formation", {})
            if sel != "ALL":
                shape = formations.get(sel, "wedge")
            else:
                shape = next(iter(formations.values()), "wedge")
            patrol_on = bool(snap.get("patrol"))

            ui.Label("SQUAD DISPATCH", style={"font_size": 18})
            ui.Label("Click a control to change it. Most commands act on the STEERED group.",
                     style={"color": 0xFF9999AA, "font_size": 12}, word_wrap=True)
            ui.Separator()

            # The three "cycle" controls — current value is shown IN the button.
            self._row("Steering", f"{sel} >", "who commands affect (cycle ALL>G0>G1...)",
                      lambda h: h.select_next(),
                      "Pick which group your commands control. ALL = the whole squad.")
            self._row("Grouping", f"{len(groups)}x{size} >", "split the squad (1x6 -> 2x3 -> 3x2)",
                      lambda h: h.cycle_regroup(),
                      "Re-partition the 6 dogs into N groups of M.")
            self._row("Formation", f"{shape.upper()} >", "line shape (wedge -> row -> column)",
                      lambda h: h.cycle_formation(),
                      "Change how the steered group lines up when it moves.")

            ui.Separator()
            ui.Label("Send the steered group to a zone:", style={"color": 0xFFAAAAAA})
            with ui.HStack(spacing=4, height=0):
                for i, z in enumerate(snap.get("zones", [])):
                    zb = ui.Button(z["name"], clicked_fn=lambda i=i: self._do(lambda h: h.dispatch_zone(i)))
                    try:
                        zb.set_tooltip(f"March the steered group to {z['name']} "
                                       f"({z['x']:+.1f}, {z['y']:+.1f}).")
                    except Exception:  # noqa: BLE001
                        pass

            ui.Separator()
            with ui.HStack(spacing=4, height=0):
                pb = ui.Button(f"Patrol: {'ON' if patrol_on else 'OFF'}",
                               clicked_fn=lambda: self._do(lambda h: h.toggle_patrol()))
                rb = ui.Button("Recall -> HOME", clicked_fn=lambda: self._do(lambda h: h.recall()))
                hb = ui.Button("Halt all", clicked_fn=lambda: self._do(lambda h: h.halt()))
            for b, t in ((pb, "Toggle auto-patrol: the squad loops through every zone."),
                         (rb, "Send the whole squad back to HOME (0, 0)."),
                         (hb, "Stop every dog where it stands.")):
                try:
                    b.set_tooltip(t)
                except Exception:  # noqa: BLE001
                    pass

            ui.Separator()
            ui.Label("Dogs   (id | group | position | -> target | *=arrived):",
                     style={"color": 0xFFAAAAAA, "font_size": 12})
            for d in snap.get("dogs", []):
                tgt = "-" if d.get("tx") is None else f"({d['tx']:+.1f},{d['ty']:+.1f})"
                mark = "*" if d.get("arrived") else " "
                ui.Label(
                    f"  {d['id']}  {d.get('group') or '-':>3}  "
                    f"({d['x']:+5.1f},{d['y']:+5.1f})  -> {tgt}  {mark}",
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

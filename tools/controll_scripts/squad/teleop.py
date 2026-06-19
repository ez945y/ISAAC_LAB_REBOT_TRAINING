# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleop console — the operator interface to the squad ABI.

A thin, Isaac-free input layer. It reads single keypresses from the terminal and
translates them into squad-level :class:`Command` *intents* (go-to-zone, change
formation, regroup, patrol, halt, recall). It does NOT touch Isaac, the Dispatcher,
or the Go2 — exactly like the missions/scheduler, it sits above the ABI boundary.
The runtime drains :meth:`poll` each loop and applies each Command to the
``Dispatcher`` / ``Squad``.

Why stdin (not the Kit/WebRTC keyboard): the demo streams a *headless,
``--no-window``* native SimulationApp, so there is no app window to capture key
events. The operator drives from the terminal that launched the run while watching
the WebRTC stream — a clean decoupling that needs no Kit input plumbing.

Single source of truth for the key bindings is :data:`KEYMAP`; :func:`help_text`
renders it for the on-screen legend.
"""

from __future__ import annotations

import queue
import select
import sys
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """An operator intent expressed against the squad ABI (no Isaac, no Go2).

    ``verb`` is the action; ``arg`` is an optional payload (e.g. a zone index).
    """

    verb: str          # zone | shape | regroup | patrol | recall | halt | help | quit
    arg: int | None = None


# key -> Command. Zone keys 1..6 carry the zone index; the rest are verbs.
KEYMAP: dict[str, Command] = {
    "1": Command("zone", 0),
    "2": Command("zone", 1),
    "3": Command("zone", 2),
    "4": Command("zone", 3),
    "5": Command("zone", 4),
    "s": Command("select"),    # cycle steered group: ALL -> G0 -> G1 -> ...
    "f": Command("shape"),     # cycle steered group's formation WEDGE/ROW/COLUMN
    "g": Command("regroup"),   # cycle 1x6 -> 2x3 -> 3x2
    "p": Command("patrol"),    # toggle auto-patrol of the zones
    "0": Command("recall"),    # whole squad back to HOME
    "x": Command("halt"),      # stop everyone where they are
    "h": Command("help"),
    "?": Command("help"),
    "q": Command("quit"),
}


def help_text(zone_names: list[str]) -> str:
    """The operator legend, built from :data:`KEYMAP` and the live zone names."""
    zones = "   ".join(f"[{i + 1}] {n}" for i, n in enumerate(zone_names))
    return (
        "\n" + "=" * 70 + "\n"
        "  GO2 SQUAD — keyboard dispatch\n"
        "  [s] steer ALL/G0/G1/...  then act on that group only:\n"
        "  Dispatch:  " + zones + "\n"
        "  [f] formation  WEDGE/ROW/COLUMN     [g] regroup 1x6/2x3/3x2\n"
        "  [p] patrol on/off   [0] recall HOME   [x] halt   [h/?] help   [q] quit\n"
        + "=" * 70 + "\n"
    )


class KeyboardConsole:
    """Background single-keypress reader -> queue of :class:`Command`.

    Puts the terminal in cbreak mode so each key fires immediately (no Enter).
    Degrades to a no-op if stdin is not a TTY (e.g. piped/`--auto`), so the same
    script still runs unattended. Always call :meth:`stop` to restore the terminal.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[Command] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._old_attr = None

    @property
    def active(self) -> bool:
        return self._thread is not None

    def start(self) -> bool:
        if not sys.stdin.isatty():
            print("[teleop] stdin is not a TTY — keyboard disabled (auto only).", flush=True)
            return False
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._old_attr = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                continue
            ch = sys.stdin.read(1)
            cmd = KEYMAP.get(ch.lower())
            if cmd is not None:
                self._q.put(cmd)

    def poll(self) -> list[Command]:
        """Drain and return every Command queued since the last call."""
        out: list[Command] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    def stop(self) -> None:
        self._stop.set()
        if self._fd is not None and self._old_attr is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attr)
            self._old_attr = None

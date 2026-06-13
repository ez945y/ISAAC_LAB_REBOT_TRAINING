"""Project livestream helper for Isaac Lab scripts.

Isaac Lab's ``--livestream 2`` only does ``--enable omni.kit.livestream.app`` on
its headless experience. That alone does NOT give a remote WebRTC client a usable
stream -- several Kit settings the official isaacsim.exp.full.streaming app bakes
in are missing. This helper reproduces them by injecting into ``args_cli.kit_args``
(and ``args_cli.visualizer``) so you never hand-type the long ``--kit_args="..."``.

What it sets when streaming (each fixed a real failure we hit):

* ``primaryStream/publicIp``        -- ICE candidate the client must reach; mode 2
  leaves it at 127.0.0.1, so media never connects over NAT/VPN (black screen).
* ``--no-window``                   -- else an OS window opens at the desktop res
  and mismatches the client-negotiated res ("Cannot stream video frame").
* ``app/livestream/allowResize`` + ``primaryStream/allowDynamicResize`` -- BOTH
  layers, like the official kit; only one => client gets one frame then drops.
* ``visualizer=kit``                -- the Kit visualizer pumps ``app.update()``
  which delivers frames; AppLauncher auto-injects it only for XR, not livestream,
  so without it the stream stalls after a few frames.
* ``runLoops/main/rateLimitEnabled=true`` (60 Hz) -- Isaac Lab's headless kit sets
  it false, so the loop runs flat-out (steps dumped at once, no stream cadence).
* ``primaryStream/enableEventTracing=false`` -- stop NvStreamer-*.etli log spam.

It is a no-op when livestreaming is disabled, so it is safe to call unconditionally.

Usage (one line, right after parse_args and before AppLauncher):

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    from livestream_support import apply_livestream_defaults
    apply_livestream_defaults(args_cli)
    app_launcher = AppLauncher(args_cli)

Then run:  python scripts/<your_script>.py --livestream 2
The advertised IP is auto-detected (host's private LAN IP). Override with:
    export LIVESTREAM_PUBLIC_IP=192.168.90.162

For demos that finish and close, guard a keep-alive loop with is_livestreaming()
so the final scene stays streamable:

    if args_cli.hold_open or is_livestreaming(args_cli):
        while simulation_app.is_running():
            sim.step()
"""

from __future__ import annotations

import ipaddress
import os
import socket
import subprocess

_STREAM = "/exts/omni.kit.livestream.app/primaryStream"

# Cached at apply time, BEFORE AppLauncher runs. AppLauncher pops "livestream"
# out of args_cli.__dict__ (app_launcher.py:266/759), so reading args_cli.livestream
# after launch is unreliable -- callers should rely on this cache instead.
_ACTIVE: bool | None = None


def _is_private(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return addr.is_private and not addr.is_loopback and not addr.is_link_local
    except ValueError:
        return False


def _primary_ip() -> str:
    """Auto-detect this host's private LAN IPv4 (the address a remote client uses).

    Prefers the source IP of the default route (sends no packets); falls back to
    scanning ``hostname -I`` for any RFC1918 address. Returns 127.0.0.1 only if
    nothing private is found.
    """
    # 1) Source IP for the default route -- normally the private LAN IP.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        if _is_private(ip):
            return ip
    except OSError:
        ip = ""
    finally:
        sock.close()

    # 2) Fall back to the first private address reported by the host.
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout
        for cand in out.split():
            if _is_private(cand):
                return cand
    except (OSError, subprocess.SubprocessError):
        pass

    return ip or "127.0.0.1"


def _livestream_enabled(args_cli) -> bool:
    """Mirror AppLauncher precedence: CLI --livestream over LIVESTREAM env."""
    val = getattr(args_cli, "livestream", -1)
    val = -1 if val is None else int(val)
    if val >= 1:
        return True
    if val == 0:
        return False
    return int(os.environ.get("LIVESTREAM", 0)) >= 1


def is_livestreaming(args_cli=None) -> bool:
    """Whether livestreaming is enabled.

    Returns the value cached by :func:`apply_livestream_defaults` (captured before
    AppLauncher strips the ``livestream`` arg). Falls back to inspecting ``args_cli``
    only if the cache was never set.
    """
    if _ACTIVE is not None:
        return _ACTIVE
    return _livestream_enabled(args_cli) if args_cli is not None else False


def apply_livestream_defaults(args_cli, public_ip: str | None = None) -> None:
    """Inject publicIp + allowDynamicResize into ``args_cli.kit_args`` when streaming.

    Args:
        args_cli: the parsed namespace (must already have AppLauncher args added).
        public_ip: override the advertised IP. Defaults to env ``LIVESTREAM_PUBLIC_IP``
            or the host's auto-detected primary LAN IP.
    """
    global _ACTIVE
    _ACTIVE = _livestream_enabled(args_cli)
    if not _ACTIVE:
        return

    # Livestreaming needs the Kit visualizer: it is what pumps app.update() so the
    # WebRTC encoder actually delivers frames. AppLauncher only auto-injects a Kit
    # visualizer for XR, not livestream (simulation_context.py), so without --viz kit
    # the client gets a few frames then the stream stalls and drops. Add it unless
    # the user already requested a visualizer set.
    viz = getattr(args_cli, "visualizer", None)
    if viz is None:
        args_cli.visualizer = ["kit"]
    elif "kit" not in viz:
        args_cli.visualizer = list(viz) + ["kit"]

    public_ip = public_ip or os.environ.get("LIVESTREAM_PUBLIC_IP") or _primary_ip()

    extra = [
        # Run without an OS window. The streaming app otherwise opens a window at
        # the desktop resolution (e.g. 1440x900) that differs from the resolution
        # the client negotiated (1920x1080) -> "Cannot stream video frame" -> black.
        # The official isaacsim.exp.full.streaming app passes this too.
        "--no-window",
        # ICE candidate the remote client must reach (NAT/VPN-friendly).
        f"--{_STREAM}/publicIp={public_ip}",
        # Resize must be enabled at BOTH layers, exactly like the stable official
        # isaacsim.exp.full.streaming.kit. With only primaryStream/allowDynamicResize
        # (and not app/livestream/allowResize) the resize is half-wired: the client
        # gets one frame then the stream drops. Both -> stable.
        "--/app/livestream/allowResize=true",
        f"--{_STREAM}/allowDynamicResize=true",
        # Re-enable the main run-loop rate limit (60 Hz), like the official
        # isaacsim.exp.full / uidoc streaming kits. Isaac Lab's headless experience
        # sets rateLimitEnabled=false, so the loop runs flat-out -> steps dumped at
        # once and the stream has no steady cadence. Cap it for real-time playback.
        "--/app/runLoops/main/rateLimitEnabled=true",
        "--/app/runLoops/main/rateLimitFrequency=60",
        # Stop NvStreamer-*.etli trace logs piling up in the working dir.
        f"--{_STREAM}/enableEventTracing=false",
    ]
    # AppLauncher splits kit_args on whitespace, so a space-joined string is fine.
    existing = (getattr(args_cli, "kit_args", "") or "").strip()
    args_cli.kit_args = (existing + " " + " ".join(extra)).strip()

    print(
        f"[livestream] enabled -> client connects to {public_ip} "
        "(signal 49100 / stream 47998), no-window, resize on, viz=kit, 60fps cap",
        flush=True,
    )

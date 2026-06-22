# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Squad Dispatch — live dispatch console for the Go2 squad (Isaac Sim extension).

Shows each dog's position and the dispatch content (zones/targets/formations) in
the viewport (so it streams over WebRTC) plus an omni.ui control panel. Fed by the
running sim through :mod:`devicenexus.squad.dispatch.runtime`.
"""

from .extension import SquadDispatchExtension  # noqa: F401
from .runtime import DispatchHandle, get_handle, set_handle  # noqa: F401

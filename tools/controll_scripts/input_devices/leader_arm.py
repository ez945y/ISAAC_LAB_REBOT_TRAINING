# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Leader Arm Socket Input Device.

Reads data from a socket connection (port 5359 by default).
The data is sent by lerobot-teleoperate_port which reads the physical leader arm.

Supports two connection modes:
- Client mode (default): Connects to a remote socket server
- Server mode: Runs a socket server and waits for connections

Supports two data modes:
- Joint mode: Receives normalized joint positions (0~1)
- EE mode: Receives end-effector pose (x, y, z, qw, qx, qy, qz)

Joint mode data format (JSON):
{
    "mode": "joint",
    "shoulder_pan": float,   # Normalized position [0, 1]
    "shoulder_lift": float,
    "elbow_flex": float,
    "wrist_flex": float,
    "wrist_roll": float,
    "gripper": float
}

EE mode data format (JSON):
{
    "x": float,      # End-effector X position in meters
    "y": float,      # End-effector Y position in meters
    "z": float,      # End-effector Z position in meters
    "qw": float,     # Quaternion w
    "qx": float,     # Quaternion x
    "qy": float,     # Quaternion y
    "qz": float,     # Quaternion z
    "gripper": float # Gripper position [0, 100]
}
"""

import json
import logging
import socket
import threading
import time
from typing import Optional, Tuple

import torch

from .base import BaseInputDevice

logger = logging.getLogger(__name__)


class LeaderArmInputDevice(BaseInputDevice):
    """Leader Arm Socket Input Device

    Receives end-effector pose from a socket connection and converts it
    to the target pose format used by the robot controller.

    The pose data comes from leader_arm_sender.py which reads a physical
    leader arm and computes forward kinematics.
    """

    def __init__(
        self,
        initial_pose: torch.Tensor,
        device: str,
        socket_host: str = '127.0.0.1',
        socket_port: int = 5359,
        server_mode: bool = False,
        gripper_min: float = 0.0,
        gripper_max: float = 100.0,
        position_scale: float = 1.0,
        position_offset: Optional[torch.Tensor] = None,
    ):
        """Initialize Leader Arm Socket Input Device"""
        # Set essential attributes early for safe cleanup
        self._server_socket = None
        self._client_socket = None
        self._receive_thread = None
        self._running = False

        self._device = device
        self._socket_host = socket_host
        self._socket_port = socket_port
        self._server_mode = server_mode
        self._gripper_min = gripper_min
        self._gripper_max = gripper_max
        self._position_scale = position_scale
        self._position_offset = position_offset

        # Target state
        self._target_pose = initial_pose.clone()
        self._initial_pose = initial_pose.clone()
        self._gripper_pos = 1.0  # Initial fully open
        self._reset_requested = False

        # Joint positions (for joint mode)
        # Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
        self._joint_positions = torch.full((5,), 0.5, device=device, dtype=torch.float32)
        self._data_mode = 'joint'  # Default to joint (position) mode as requested

        # Socket state
        self._socket_buffer = ''
        self._connected = False
        self._reconnect_interval = 1.0  # seconds
        self._last_reconnect_attempt = 0.0
        self._last_data_time = 0.0

        # Latest received data
        self._latest_data: Optional[dict] = None
        self._data_lock = threading.Lock()

        # Start connection thread
        self._running = True
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()

        mode_str = 'server' if server_mode else 'client'
        logger.info(f'LeaderArmInputDevice initialized in {mode_str} mode on {socket_host}:{socket_port}')
        self._print_info()

    def _print_info(self) -> None:
        """Print device information."""
        print('\n' + '=' * 50)
        print('Leader Arm Socket Input Device')
        print('=' * 50)
        mode_str = 'SERVER (waiting)' if self._server_mode else 'CLIENT (connecting)'
        print(f'  Mode: {mode_str}')
        print(f'  Socket: {self._socket_host}:{self._socket_port}')
        print(f'  Gripper range: [{self._gripper_min}, {self._gripper_max}]')
        print(f'  Position scale: {self._position_scale}')
        print('  Controls:')
        print('    - Move the leader arm to control position')
        print('    - Gripper on leader arm controls gripper')
        print('=' * 50)

    def _start_server(self) -> bool:
        """Start socket server and wait for connection."""
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self._socket_host, self._socket_port))
            self._server_socket.listen(1)
            self._server_socket.settimeout(1.0)
            logger.info(f'Server listening on {self._socket_host}:{self._socket_port}')
            print(f'\n[INFO] Waiting for leader arm connection on port {self._socket_port}...')
            return True
        except OSError as e:
            logger.error(f'Failed to start server: {e}')
            return False

    def _accept_client(self) -> bool:
        """Accept a client connection."""
        try:
            client, addr = self._server_socket.accept()
            client.settimeout(0.1)
            self._client_socket = client
            self._connected = True
            self._socket_buffer = ''
            logger.info(f'Client connected from {addr}')
            print(f'\n[INFO] Leader arm connected from {addr}')
            return True
        except socket.timeout:
            return False
        except OSError as e:
            logger.debug(f'Accept failed: {e}')
            return False

    def _connect_to_server(self) -> bool:
        """Attempt to connect to the socket server (client mode)."""
        try:
            self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._client_socket.settimeout(2.0)
            self._client_socket.connect((self._socket_host, self._socket_port))
            self._client_socket.settimeout(0.1)  # Non-blocking reads
            self._connected = True
            self._socket_buffer = ''
            logger.info(f'Connected to leader arm sender at {self._socket_host}:{self._socket_port}')
            return True
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            logger.debug(f'Connection failed: {e}')
            if self._client_socket:
                try:
                    self._client_socket.close()
                except OSError:
                    pass
            self._client_socket = None
            self._connected = False
            return False

    def _disconnect(self) -> None:
        """Disconnect from the socket."""
        self._connected = False
        if hasattr(self, '_client_socket') and self._client_socket:
            try:
                self._client_socket.close()
            except OSError:
                pass
            self._client_socket = None

    def _receive_loop(self) -> None:
        """Background thread to receive data from socket."""
        # Start server if in server mode
        if self._server_mode:
            if not self._start_server():
                logger.error('Failed to start server, exiting receive loop')
                return

        while self._running:
            # Handle connection based on mode
            if not self._connected:
                current_time = time.time()
                if current_time - self._last_reconnect_attempt >= self._reconnect_interval:
                    self._last_reconnect_attempt = current_time

                    if self._server_mode:
                        self._accept_client()
                    else:
                        self._connect_to_server()

                if not self._connected:
                    time.sleep(0.1)
                    continue

            # Try to receive data
            try:
                data = self._client_socket.recv(4096)
                if not data:
                    print('\n[WARNING] Leader arm connection closed by remote (empty data).')
                    self._disconnect()
                    continue

                self._socket_buffer += data.decode('utf-8')

                # Process complete JSON lines
                while '\n' in self._socket_buffer:
                    line, self._socket_buffer = self._socket_buffer.split('\n', 1)
                    if line.strip():
                        try:
                            parsed = json.loads(line)
                            with self._data_lock:
                                self._latest_data = parsed
                                self._last_data_time = time.time()
                        except json.JSONDecodeError as e:
                            print(f'\n[DEBUG] JSON decode error: {e}')

            except socket.timeout:
                continue
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                print(f'\n[WARNING] Leader arm socket error: {e}')
                self._disconnect()
            except Exception as e:
                print(f'\n[ERROR] Unexpected error in receive loop: {e}')
                self._disconnect()

    def update(self) -> Tuple[torch.Tensor, float, bool]:
        """Update input state

        Returns:
            Tuple[torch.Tensor, float, bool]:
                - target_pose: Absolute target pose
                - gripper_pos: Gripper position [0, 1]
                - reset_requested: Whether reset was requested
        """
        # Check reset request
        if self._reset_requested:
            self._target_pose = self._initial_pose.clone()
            self._gripper_pos = 1.0
            self._reset_requested = False
            return self._target_pose, self._gripper_pos, True

        # Read latest pose data
        pose, gripper = self._read_leader_pose()
        if pose is not None:
            self._target_pose = pose.clone()
        if gripper is not None:
            self._gripper_pos = gripper

        return self._target_pose.clone(), self._gripper_pos, False

    def _read_leader_pose(self) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """Read leader arm data from socket.

        Returns:
            Tuple[torch.Tensor | None, float | None]:
                - pose: End-effector pose [pos(3) + quat(4)], None if no update or joint mode
                - gripper: Gripper position [0, 1], None if no update
        """
        with self._data_lock:
            if self._latest_data is None:
                return None, None
            data = self._latest_data.copy()

        try:
            # Check data mode
            mode = data.get('mode', 'joint')
            if 'mode' not in data:
                 print(f"[DEBUG] 'mode' key missing in data: {data.keys()}")
            
            # TODO: Remove this debug print later
            # if getattr(self, "_debug_counter", 0) % 60 == 0:
            #     print(f"[DEBUG] Received data mode: {mode}, keys: {list(data.keys())}")
            # self._debug_counter = getattr(self, "_debug_counter", 0) + 1

            self._data_mode = mode

            if mode == 'joint':
                # Check if ordered joints list exists (more robust)
                if 'joints' in data and isinstance(data['joints'], list):
                    # joints[0:5] are arm, joints[5] is gripper
                    self._joint_positions = torch.tensor(
                        data['joints'][:5], device=self._device, dtype=torch.float32
                    )
                    gripper_normalized = max(0.0, min(1.0, data['joints'][5]))
                else:
                    # Fallback to individual keys
                    self._joint_positions = torch.tensor([
                        data.get('shoulder_pan', 0.5),
                        data.get('shoulder_lift', 0.5),
                        data.get('elbow_flex', 0.5),
                        data.get('wrist_flex', 1.0),
                        data.get('wrist_roll', 0.5),
                    ], device=self._device, dtype=torch.float32)
                    gripper_raw = data.get('gripper', 0.0)
                    gripper_normalized = max(0.0, min(1.0, gripper_raw))

                return None, gripper_normalized
            else:
                # EE mode: extract position and orientation
                x = data.get('x', 0.0)
                y = data.get('y', 0.0)
                z = data.get('z', 0.0)

                # Apply scale
                x *= self._position_scale
                y *= self._position_scale
                z *= self._position_scale

                # Apply offset if provided
                if self._position_offset is not None:
                    x += self._position_offset[0].item()
                    y += self._position_offset[1].item()
                    z += self._position_offset[2].item()

                # Extract quaternion (w, x, y, z format - Isaac convention)
                qw = data.get('qw', 1.0)
                qx = data.get('qx', 0.0)
                qy = data.get('qy', 0.0)
                qz = data.get('qz', 0.0)

                # Extract gripper if available in EE mode
                gripper_raw = data.get('gripper', 50.0)
                gripper_normalized = max(0.0, min(1.0, gripper_raw / 100.0))

                # Create pose tensor
                pose = torch.tensor(
                    [[x, y, z, qw, qx, qy, qz]],
                    device=self._device,
                    dtype=torch.float32
                )

                return pose, gripper_normalized

        except Exception as e:
            logger.error(f'Error parsing data: {e}')
            return None, None

    @property
    def data_mode(self) -> str:
        """Current data mode ('ee' or 'joint')."""
        return self._data_mode

    @property
    def joint_positions(self) -> torch.Tensor:
        """Current joint positions (normalized 0~1)."""
        return self._joint_positions.clone()

    @property
    def target_pose(self) -> torch.Tensor:
        """Current target pose."""
        return self._target_pose.clone()

    @property
    def gripper_pos(self) -> float:
        """Current gripper position."""
        return self._gripper_pos

    @property
    def is_connected(self) -> bool:
        """Whether connected to socket"""
        return self._connected

    @property
    def last_data_age(self) -> float:
        """Time since last data was received (seconds)"""
        if self._last_data_time == 0:
            return float('inf')
        return time.time() - self._last_data_time

    def reset_target(self, pose: torch.Tensor) -> None:
        """Reset target pose

        Args:
            pose: New target pose
        """
        self._target_pose = pose.clone()
        self._initial_pose = pose.clone()

    def sync_to_actual(self, actual_pose: torch.Tensor) -> None:
        """Sync target to actual pose (when target is unreachable)

        Args:
            actual_pose: Actual EE pose
        """
        # For leader arm input, we don't sync to actual
        # The leader arm position is the source of truth
        pass

    def request_reset(self) -> None:
        """Request scene reset"""
        self._reset_requested = True

    def close(self) -> None:
        """Close the socket connection and stop threads."""
        self._running = False
        self._disconnect()
        if hasattr(self, '_server_socket') and self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None
        if hasattr(self, '_receive_thread'):
            self._receive_thread.join(timeout=1.0)
        logger.info('LeaderArmInputDevice closed')

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

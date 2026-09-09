"""Small dependency-free utilities for the tracking reference sources."""

from __future__ import annotations

import json
import socket
import threading
from collections import deque
from typing import Any


class DictToClass:
    def __init__(self, data_dict: dict[str, Any]) -> None:
        for key, value in data_dict.items():
            setattr(self, key, value)


class MotionUDPServer(threading.Thread):
    """Non-blocking UDP server for offline motion-name commands."""

    def __init__(self, host: str = "127.0.0.1", port: int = 28562) -> None:
        super().__init__(daemon=True)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(0.2)
        self._messages: deque[str] = deque()
        self._lock = threading.Lock()
        self._running = True

    def run(self) -> None:
        while self._running:
            try:
                payload, _ = self._sock.recvfrom(1024)
            except TimeoutError:
                continue
            except OSError:
                break
            message = payload.decode("utf-8", errors="ignore").strip()
            if message:
                with self._lock:
                    self._messages.append(message)

    def pop_all(self) -> list[str]:
        with self._lock:
            messages = list(self._messages)
            self._messages.clear()
        return messages

    def stop(self) -> None:
        self._running = False
        self._sock.close()


class RuntimeReferenceUDPClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 28564) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._target = (host, port)

    def send(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._sock.sendto(data, self._target)

    def close(self) -> None:
        self._sock.close()

"""Stream a live MolmoSpaces camera to XRoboToolkit Remote Vision."""

from __future__ import annotations

import os
import queue
import shutil
import socket
import struct
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


HTVF_MAGIC = b"HTVF"
HTVF_VERSION = 1
HTVF_HEADER_SIZE = 24
HTVF_PAYLOAD_SIZE = 1176


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class CameraRequest:
    width: int
    height: int
    fps: int
    bitrate: int
    port: int
    camera: str
    ip: str
    transport: str = "TCP"
    low_latency: bool = True


def _read_i32_le(data: bytes | bytearray, offset: int) -> int:
    if offset + 4 > len(data):
        raise ValueError("int32 out of range")
    return struct.unpack_from("<i", data, offset)[0]


def _read_compact_string(data: bytes | bytearray, offset: int) -> tuple[str, int]:
    if offset >= len(data):
        raise ValueError("string length out of range")
    length = data[offset]
    offset += 1
    if offset + length > len(data):
        raise ValueError("string data out of range")
    value = bytes(data[offset : offset + length]).decode("utf-8", errors="replace")
    return value, offset + length


def parse_camera_request(payload: bytes | bytearray) -> CameraRequest:
    if len(payload) < 32 or payload[0] != 0xCA or payload[1] != 0xFE or payload[2] != 1:
        raise ValueError("invalid camera request")
    offset = 3
    width = _read_i32_le(payload, offset)
    height = _read_i32_le(payload, offset + 4)
    fps = _read_i32_le(payload, offset + 8)
    bitrate = _read_i32_le(payload, offset + 12)
    port = _read_i32_le(payload, offset + 24)
    offset += 28
    camera, offset = _read_compact_string(payload, offset)
    ip, offset = _read_compact_string(payload, offset)
    transport = "TCP"
    low_latency = True
    if offset < len(payload):
        transport, offset = _read_compact_string(payload, offset)
    if offset + 4 <= len(payload):
        low_latency = _read_i32_le(payload, offset) != 0
    return CameraRequest(
        width=width,
        height=height,
        fps=fps,
        bitrate=bitrate,
        port=port,
        camera=camera,
        ip=ip,
        transport=transport or "TCP",
        low_latency=low_latency,
    )


def _parse_inner_control_packet(body: bytes | bytearray) -> tuple[str, bytes]:
    if len(body) < 8:
        raise ValueError("control packet too small")
    offset = 0
    command_len = _read_i32_le(body, offset)
    offset += 4
    if command_len < 0 or offset + command_len + 4 > len(body):
        raise ValueError("invalid command length")
    command = bytes(body[offset : offset + command_len]).decode("utf-8", errors="replace").rstrip("\x00")
    offset += command_len
    data_len = _read_i32_le(body, offset)
    offset += 4
    if data_len < 0 or offset + data_len > len(body):
        raise ValueError("invalid payload length")
    return command, bytes(body[offset : offset + data_len])


def try_parse_control_packet(buffer: bytearray) -> tuple[int, str, bytes] | None:
    """Parse either XRoboToolkit's wrapped TCP frame or its raw body."""

    if len(buffer) < 8:
        return None

    wrapped_len = struct.unpack_from(">I", buffer, 0)[0]
    if 0 < wrapped_len <= 1024 * 1024:
        if len(buffer) < 4 + wrapped_len:
            return None
        body = bytes(buffer[4 : 4 + wrapped_len])
        command, payload = _parse_inner_control_packet(body)
        return 4 + wrapped_len, command, payload

    command_len = _read_i32_le(buffer, 0)
    if 0 <= command_len <= 1024:
        needed_header = 4 + command_len + 4
        if len(buffer) < needed_header:
            return None
        data_len = _read_i32_le(buffer, 4 + command_len)
        if data_len < 0 or data_len > 1024 * 1024:
            raise ValueError("invalid raw payload length")
        needed = needed_header + data_len
        if len(buffer) < needed:
            return None
        command, payload = _parse_inner_control_packet(buffer[:needed])
        return needed, command, payload

    raise ValueError("invalid control framing")


def _find_ffmpeg() -> str:
    configured = os.getenv("MOLMO_PICO_FFMPEG")
    if configured:
        return configured
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found; install ffmpeg or imageio-ffmpeg") from exc


def _to_uint8_rgb(frame: Any) -> np.ndarray | None:
    if not isinstance(frame, np.ndarray):
        return None
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return None
    frame = frame[..., :3]
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        max_value = float(np.nanmax(frame)) if frame.size else 1.0
        scale = 255.0 if max_value <= 1.5 else 1.0
        return np.clip(frame * scale, 0, 255).astype(np.uint8)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _first_observation(observation: Any) -> dict[str, Any]:
    if isinstance(observation, list) and observation:
        first = observation[0]
        return first if isinstance(first, dict) else {}
    if isinstance(observation, dict):
        return observation
    return {}


def _task_text(task: Any) -> str:
    if task is None:
        return ""
    try:
        return str(task.get_task_description())
    except Exception:
        return ""


def _resize_with_padding(frame_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    src_h, src_w = frame_rgb.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def _fit_text_to_width(text: str, max_width: int, font: int, scale: float, thickness: int) -> str:
    import cv2

    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    lo = 0
    hi = len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid].rstrip() + suffix
        if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + suffix


def _draw_task_overlay_rgb(frame_rgb: np.ndarray, task_text: str) -> np.ndarray:
    if not task_text:
        return frame_rgb
    if not _env_flag("MOLMO_PICO_TASK_OVERLAY", True):
        return frame_rgb

    import cv2

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    height, width = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.42, min(0.72, width / 900.0))
    thickness = 1 if width < 900 else 2
    margin = max(8, width // 80)
    bar_h = max(34, int(38 * scale + 18))
    label = "TASK: " + " ".join(task_text.split())
    label = _fit_text_to_width(label, width - 2 * margin, font, scale, thickness)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (width, min(height, bar_h)), (0, 0, 0), -1)
    frame_bgr = cv2.addWeighted(overlay, 0.72, frame_bgr, 0.28, 0)
    cv2.putText(
        frame_bgr,
        label,
        (margin, min(height - 8, int(bar_h * 0.68))),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _shift_frame_x(frame_rgb: np.ndarray, shift_px: int) -> np.ndarray:
    if shift_px == 0:
        return frame_rgb.copy()
    pad = abs(int(shift_px))
    padded = np.pad(frame_rgb, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    start_x = pad + int(shift_px)
    return padded[:, start_x : start_x + frame_rgb.shape[1]].copy()


def _compose_stereo_sbs(left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
    if left_rgb.shape[:2] != right_rgb.shape[:2]:
        right_rgb = _resize_with_padding(right_rgb, left_rgb.shape[1], left_rgb.shape[0])
    return np.concatenate((left_rgb, right_rgb), axis=1)


def _start_code_len(nal: bytes) -> int:
    if nal.startswith(b"\x00\x00\x00\x01"):
        return 4
    if nal.startswith(b"\x00\x00\x01"):
        return 3
    return 0


def _find_start_code(buffer: bytearray, start: int = 0) -> int:
    a = buffer.find(b"\x00\x00\x01", start)
    b = buffer.find(b"\x00\x00\x00\x01", start)
    if a < 0:
        return b
    if b < 0:
        return a
    return min(a, b)


class H264AccessUnitParser:
    """Split Annex-B H264 byte stream into access-unit chunks."""

    def __init__(self, emit: Callable[[bytes], None]) -> None:
        self._emit = emit
        self._buffer = bytearray()
        self._current: list[bytes] = []
        self._seen_vcl = False

    def feed(self, data: bytes) -> None:
        if not data:
            return
        self._buffer.extend(data)
        while True:
            first = _find_start_code(self._buffer, 0)
            if first < 0:
                if len(self._buffer) > 4:
                    del self._buffer[:-4]
                return
            if first > 0:
                del self._buffer[:first]
            next_start = _find_start_code(self._buffer, _start_code_len(self._buffer) or 3)
            if next_start < 0:
                return
            nal = bytes(self._buffer[:next_start])
            del self._buffer[:next_start]
            self._process_nal(nal)

    def flush(self) -> None:
        if self._buffer and _find_start_code(self._buffer, 0) == 0:
            self._process_nal(bytes(self._buffer))
        self._buffer.clear()
        self._emit_current()

    def _process_nal(self, nal: bytes) -> None:
        sc_len = _start_code_len(nal)
        if sc_len <= 0 or len(nal) <= sc_len:
            return
        nal_type = nal[sc_len] & 0x1F
        is_vcl = nal_type in (1, 5)
        starts_new = nal_type in (6, 7, 8, 9)

        if is_vcl and self._seen_vcl:
            self._emit_current()
        elif starts_new and self._seen_vcl:
            self._emit_current()

        self._current.append(nal)
        if is_vcl:
            self._seen_vcl = True

    def _emit_current(self) -> None:
        if self._current and self._seen_vcl:
            self._emit(b"".join(self._current))
        self._current = []
        self._seen_vcl = False


class EncodedFrameSender:
    def __init__(self, host: str, port: int, transport: str) -> None:
        self.host = host
        self.port = int(port)
        self.transport = "UDP" if transport.upper() == "UDP" else "TCP"
        self._sock: socket.socket | None = None
        self._frame_id = 0

    def connect(self) -> None:
        sock_type = socket.SOCK_DGRAM if self.transport == "UDP" else socket.SOCK_STREAM
        sock = socket.socket(socket.AF_INET, sock_type)
        if self.transport == "TCP":
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512 * 1024)
        sock.connect((self.host, self.port))
        self._sock = sock
        print(f"[VirtualCameraStreamer] Connected {self.transport} video stream to {self.host}:{self.port}")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def send(self, frame: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("stream socket is not connected")
        if self.transport == "UDP":
            self._send_udp(frame)
        else:
            self._sock.sendall(struct.pack(">I", len(frame)) + frame)

    def _send_udp(self, frame: bytes) -> None:
        assert self._sock is not None
        self._frame_id = (self._frame_id + 1) & 0xFFFFFFFF
        frag_count = max(1, (len(frame) + HTVF_PAYLOAD_SIZE - 1) // HTVF_PAYLOAD_SIZE)
        if frag_count > 0xFFFF:
            print(f"[VirtualCameraStreamer][drop] H264 frame too large for UDP: {len(frame)}")
            return
        for frag_id in range(frag_count):
            offset = frag_id * HTVF_PAYLOAD_SIZE
            payload = frame[offset : offset + HTVF_PAYLOAD_SIZE]
            header = bytearray(HTVF_HEADER_SIZE)
            header[:4] = HTVF_MAGIC
            header[4] = HTVF_VERSION
            header[5] = 0
            struct.pack_into(">H", header, 6, HTVF_HEADER_SIZE)
            struct.pack_into(">I", header, 8, self._frame_id)
            struct.pack_into(">H", header, 12, frag_id)
            struct.pack_into(">H", header, 14, frag_count)
            struct.pack_into(">I", header, 16, len(frame))
            struct.pack_into(">H", header, 20, len(payload))
            struct.pack_into(">H", header, 22, 0)
            self._sock.send(header + payload)


class H264EncoderSession:
    def __init__(self, request: CameraRequest, *, forced_transport: str | None = None) -> None:
        force_size = _env_flag("MOLMO_PICO_FORCE_STREAM_SIZE", False)
        self._force_size = force_size
        if force_size:
            width = _env_int("MOLMO_PICO_STREAM_WIDTH", request.width or 540)
            height = _env_int("MOLMO_PICO_STREAM_HEIGHT", request.height or 960)
        else:
            width = request.width or _env_int("MOLMO_PICO_STREAM_WIDTH", 540)
            height = request.height or _env_int("MOLMO_PICO_STREAM_HEIGHT", 960)
        fps = _env_int("MOLMO_PICO_STREAM_FPS", request.fps or 25)
        bitrate = _env_int("MOLMO_PICO_STREAM_BITRATE", request.bitrate or 1_500_000)
        transport = (forced_transport or request.transport or "TCP").upper()
        if transport == "AUTO":
            transport = "TCP"
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self.bitrate = max(100_000, bitrate)
        self._request = request
        self._transport = transport
        self._sender = EncodedFrameSender(request.ip, request.port, transport)
        self._latest_frame: np.ndarray | None = None
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._process: subprocess.Popen | None = None
        self._writer_thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._send_queue: queue.Queue[bytes] = queue.Queue(maxsize=2)
        self._send_thread: threading.Thread | None = None
        self._sent_frames = 0
        self._last_send_monotonic: float | None = None

    def start(self) -> None:
        self._sender.connect()
        self._process = subprocess.Popen(
            self._ffmpeg_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._writer_thread = threading.Thread(target=self._writer_loop, name="molmo-pico-h264-writer", daemon=True)
        self._reader_thread = threading.Thread(target=self._reader_loop, name="molmo-pico-h264-reader", daemon=True)
        self._send_thread = threading.Thread(target=self._send_loop, name="molmo-pico-h264-sender", daemon=True)
        self._writer_thread.start()
        self._reader_thread.start()
        self._send_thread.start()
        if not self._force_size and (
            os.getenv("MOLMO_PICO_STREAM_WIDTH") or os.getenv("MOLMO_PICO_STREAM_HEIGHT")
        ):
            print(
                "[VirtualCameraStreamer] Ignoring MOLMO_PICO_STREAM_WIDTH/HEIGHT because "
                "MOLMO_PICO_FORCE_STREAM_SIZE is not enabled; matching PICO request size."
            )
        print(
            "[VirtualCameraStreamer] H264 encoder started "
            f"{self.width}x{self.height}@{self.fps} bitrate={self.bitrate} transport={self._transport}"
        )

    def stop(self) -> None:
        self._stop_event.set()
        process = self._process
        if process is not None:
            try:
                if process.stdin:
                    process.stdin.close()
            except Exception:
                pass
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except Exception:
                with suppress(Exception):
                    process.kill()
            self._process = None
        self._sender.close()

    def is_running(self) -> bool:
        if self._stop_event.is_set():
            return False
        return self._process is not None and self._process.poll() is None

    def is_ready(self) -> bool:
        return self.is_running() and self._sent_frames > 0

    def submit(self, frame_rgb: np.ndarray, *, task_text: str = "") -> None:
        if self._stop_event.is_set():
            return
        frame = _to_uint8_rgb(frame_rgb)
        if frame is None:
            return
        prepared = _resize_with_padding(frame, self.width, self.height)
        prepared = _draw_task_overlay_rgb(prepared, task_text)
        with self._latest_lock:
            self._latest_frame = prepared

    def _ffmpeg_cmd(self) -> list[str]:
        ffmpeg = _find_ffmpeg()
        return [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            str(self.bitrate),
            "-g",
            str(self.fps),
            "-keyint_min",
            str(self.fps),
            "-bf",
            "0",
            "-threads",
            "1",
            "-x264-params",
            "repeat-headers=1:scenecut=0",
            "-f",
            "h264",
            "pipe:1",
        ]

    def _writer_loop(self) -> None:
        assert self._process is not None and self._process.stdin is not None
        period = 1.0 / self.fps
        next_time = time.monotonic()
        black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        while not self._stop_event.is_set():
            with self._latest_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is None:
                frame = black
            try:
                self._process.stdin.write(frame.tobytes())
                self._process.stdin.flush()
            except Exception:
                self._stop_event.set()
                break
            next_time += period
            time.sleep(max(0.0, next_time - time.monotonic()))

    def _reader_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None

        def emit(frame: bytes) -> None:
            if self._stop_event.is_set():
                return
            try:
                self._send_queue.put_nowait(frame)
            except queue.Full:
                with suppress(Exception):
                    self._send_queue.get_nowait()
                with suppress(Exception):
                    self._send_queue.put_nowait(frame)

        parser = H264AccessUnitParser(emit)
        while not self._stop_event.is_set():
            data = self._process.stdout.read(4096)
            if not data:
                break
            parser.feed(data)
        parser.flush()

    def _send_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._send_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._sender.send(frame)
                self._sent_frames += 1
                self._last_send_monotonic = time.monotonic()
            except Exception as exc:
                print(f"[VirtualCameraStreamer][stopped] send failed: {exc}")
                self._stop_event.set()
                break


class VirtualCameraStreamer:
    """Control server plus live-frame bridge for XRoboToolkit Remote Vision."""

    def __init__(
        self,
        *,
        camera_name: str = "head_camera",
        left_camera_name: str | None = None,
        right_camera_name: str | None = None,
        video_mode: str = "mono",
        stereo_shift_px: int = 8,
        target_ip: str | None = None,
        prefer_control_ip: bool = False,
        host: str = "0.0.0.0",
        port: int = 13579,
        transport: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.left_camera_name = left_camera_name
        self.right_camera_name = right_camera_name
        self.video_mode = video_mode.lower()
        self.stereo_shift_px = int(stereo_shift_px)
        self.target_ip = target_ip
        self.prefer_control_ip = bool(prefer_control_ip)
        self.host = host
        self.port = int(port)
        self.transport = transport
        self.auto_open_on_control_connect = _env_flag("MOLMO_PICO_AUTO_OPEN_ON_CONNECT", False)
        self._server_socket: socket.socket | None = None
        self._server_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._session_lock = threading.Lock()
        self._session: H264EncoderSession | None = None
        self._task_text = ""

    def start(self) -> None:
        if self._server_thread is not None:
            return
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._server_socket.bind((self.host, self.port))
        except OSError as exc:
            with suppress(Exception):
                self._server_socket.close()
            self._server_socket = None
            if getattr(exc, "errno", None) in (98, 10048):
                raise RuntimeError(
                    f"PICO control port {self.host}:{self.port} is already in use. "
                    "Stop the old MolmoSpaces/PICO video process before starting a new run."
                ) from exc
            raise
        self._server_socket.listen(2)
        self._server_socket.settimeout(0.5)
        self._server_thread = threading.Thread(target=self._server_loop, name="molmo-pico-camera-control", daemon=True)
        self._server_thread.start()
        print(
            "[VirtualCameraStreamer] Control listening "
            f"on {self.host}:{self.port}; video_mode={self.video_mode}; "
            "PICO Remote Vision should connect to this PC IP"
        )

    def stop(self) -> None:
        self._stop_event.set()
        self._stop_session()
        if self._server_socket is not None:
            with suppress(Exception):
                self._server_socket.close()
            self._server_socket = None

    def has_active_session(self) -> bool:
        with self._session_lock:
            session = self._session
        return bool(session is not None and session.is_ready())

    def submit_observation(self, observation: Any, *, task: Any = None) -> None:
        obs = _first_observation(observation)
        task_text = _task_text(task)
        if task_text:
            self._task_text = task_text
        frame = self._frame_from_observation(obs)
        if frame is None:
            return
        with self._session_lock:
            session = self._session
        if session is not None:
            session.submit(frame, task_text=self._task_text)

    def _frame_from_observation(self, obs: dict[str, Any]) -> np.ndarray | None:
        if self.video_mode == "stereo_sbs":
            return self._stereo_frame_from_observation(obs)
        return _to_uint8_rgb(obs.get(self.camera_name))

    def _stereo_frame_from_observation(self, obs: dict[str, Any]) -> np.ndarray | None:
        left = _to_uint8_rgb(obs.get(self.left_camera_name)) if self.left_camera_name else None
        right = _to_uint8_rgb(obs.get(self.right_camera_name)) if self.right_camera_name else None
        if left is not None and right is not None:
            return _compose_stereo_sbs(left, right)

        base = _to_uint8_rgb(obs.get(self.camera_name))
        if base is None:
            return None

        # Fallback is synthetic stereo: enough to exercise the PICO stereo display path,
        # but it is not depth-correct like two independently rendered cameras.
        shift = max(0, self.stereo_shift_px)
        left = _shift_frame_x(base, shift)
        right = _shift_frame_x(base, -shift)
        return _compose_stereo_sbs(left, right)

    def _server_loop(self) -> None:
        assert self._server_socket is not None
        while not self._stop_event.is_set():
            try:
                client, addr = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(
                target=self._client_loop,
                args=(client, addr),
                name="molmo-pico-camera-client",
                daemon=True,
            )
            thread.start()

    def _client_loop(self, client: socket.socket, addr) -> None:
        print(f"[VirtualCameraStreamer] Control client connected: {addr[0]}:{addr[1]}")
        auto_open_timer = None
        if self.auto_open_on_control_connect:
            auto_open_timer = threading.Timer(1.0, self._auto_open_if_needed, args=(addr[0],))
            auto_open_timer.daemon = True
            auto_open_timer.start()
        buffer = bytearray()
        try:
            client.settimeout(0.5)
            while not self._stop_event.is_set():
                try:
                    data = client.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                buffer.extend(data)
                while True:
                    parsed = try_parse_control_packet(buffer)
                    if parsed is None:
                        break
                    consumed, command, payload = parsed
                    del buffer[:consumed]
                    self._handle_command(command, payload, addr[0])
        except Exception as exc:
            print(f"[VirtualCameraStreamer][warning] control client error: {exc}")
        finally:
            if auto_open_timer is not None:
                auto_open_timer.cancel()
            with suppress(Exception):
                client.close()
            print("[VirtualCameraStreamer] Control client disconnected")

    def _handle_command(self, command: str, payload: bytes, fallback_ip: str) -> None:
        if command == "OPEN_CAMERA":
            request = parse_camera_request(payload)
            original_ip = request.ip
            if self.target_ip:
                request.ip = self.target_ip
            elif self.prefer_control_ip:
                request.ip = fallback_ip
            elif not request.ip:
                request.ip = fallback_ip
            print(
                "[VirtualCameraStreamer] OPEN_CAMERA "
                f"camera={request.camera} target={request.ip}:{request.port} "
                f"request={request.width}x{request.height}@{request.fps} transport={request.transport} "
                f"pico_reported_ip={original_ip or 'none'} control_ip={fallback_ip}"
            )
            if self.video_mode == "stereo_sbs" and request.camera.upper() == "WEBCAM":
                print(
                    "[VirtualCameraStreamer][warning] stereo_sbs is active, but PICO requested WEBCAM. "
                    "Select PICO4U or ZEDMINI in Remote Vision for stereo shader splitting."
                )
            self._start_session(request)
        elif command == "CLOSE_CAMERA":
            self._stop_session()
        else:
            print(f"[VirtualCameraStreamer] Unknown command: {command}")

    def _auto_open_if_needed(self, fallback_ip: str) -> None:
        if self._stop_event.is_set():
            return
        with self._session_lock:
            session = self._session
        if session is not None and session.is_running():
            return

        request = CameraRequest(
            width=_env_int("MOLMO_PICO_AUTO_OPEN_WIDTH", 540),
            height=_env_int("MOLMO_PICO_AUTO_OPEN_HEIGHT", 960),
            fps=_env_int("MOLMO_PICO_AUTO_OPEN_FPS", 25),
            bitrate=_env_int("MOLMO_PICO_AUTO_OPEN_BITRATE", 1_500_000),
            port=_env_int("MOLMO_PICO_AUTO_OPEN_PORT", 12345),
            camera=os.getenv("MOLMO_PICO_AUTO_OPEN_CAMERA", "WEBCAM"),
            ip=self.target_ip or fallback_ip,
            transport=os.getenv("MOLMO_PICO_AUTO_OPEN_TRANSPORT", "TCP"),
        )
        print(
            "[VirtualCameraStreamer][AUTO_OPEN] No OPEN_CAMERA received after control connect; "
            f"trying {request.camera} target={request.ip}:{request.port} "
            f"{request.width}x{request.height}@{request.fps} transport={request.transport}"
        )
        try:
            self._start_session(request)
        except Exception as exc:
            print(f"[VirtualCameraStreamer][AUTO_OPEN_FAILED] {exc}")

    def _start_session(self, request: CameraRequest) -> None:
        self._stop_session()
        session = H264EncoderSession(request, forced_transport=self.transport)
        try:
            session.start()
        except Exception:
            session.stop()
            raise
        else:
            with self._session_lock:
                self._session = session

    def _stop_session(self) -> None:
        with self._session_lock:
            session = self._session
            self._session = None
        if session is not None:
            session.stop()

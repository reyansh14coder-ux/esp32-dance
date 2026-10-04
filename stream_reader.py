import socket
import threading
import time
import urllib.request

import cv2
import numpy as np

from config import CAPTURE_URL, STREAM_URL


class ESP32Stream:
    def __init__(self, url, timeout=6):
        self.url = url
        self.timeout = timeout
        self._resp = None

    def _open(self):
        req = urllib.request.Request(self.url, headers={"User-Agent": "esp32-dance"})
        self._resp = urllib.request.urlopen(req, timeout=self.timeout)

    def _read_exactly(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self._resp.read(n - len(buf))
            if not chunk:
                raise ConnectionError("stream closed")
            buf += chunk
        return buf

    def frames(self):
        while True:
            if self._resp is None:
                try:
                    self._open()
                except Exception as e:
                    yield None, f"connect error: {e}"
                    time.sleep(1)
                    continue
            try:
                frame = self._read_frame()
                if frame is None:
                    self.close()
                    yield None, "bad frame"
                    continue
                yield frame, None
            except (socket.timeout, TimeoutError) as e:
                self.close()
                yield None, f"timeout: {e}"
            except Exception as e:
                self.close()
                yield None, f"stream error: {e}"

    def _read_frame(self):
        header = b""
        while True:
            line = self._resp.readline()
            if not line:
                raise ConnectionError("stream closed")
            header += line
            if header.endswith(b"\r\n\r\n"):
                break
        length = None
        for part in header.split(b"\r\n"):
            if part.lower().startswith(b"content-length:"):
                length = int(part.split(b":")[1].strip())
        if length is None:
            return self._read_frame_by_eoi()
        data = self._read_exactly(length)
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    def _read_frame_by_eoi(self):
        data = b""
        while True:
            byte = self._resp.read(1)
            if not byte:
                raise ConnectionError("stream closed")
            data += byte
            if data.endswith(b"\xff\xd9"):
                break
            if len(data) > 5_000_000:
                raise ValueError("frame too large")
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    def close(self):
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
            self._resp = None


class WebcamStream:
    def __init__(self, url=0, timeout=10):
        self.url = url
        self.timeout = timeout
        self._cap = None

    def frames(self):
        while True:
            if self._cap is None:
                self._cap = cv2.VideoCapture(self.url)
                if not self._cap.isOpened():
                    self._cap = None
                    yield None, "cannot open source"
                    time.sleep(1)
                    continue
            ok, frame = self._cap.read()
            if not ok:
                self.close()
                yield None, "read failed"
                time.sleep(0.5)
                continue
            yield frame, None

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CaptureStream:
    def __init__(self, url, fps=10, timeout=8):
        self.url = url
        self.interval = 1.0 / fps
        self.timeout = timeout

    def frames(self):
        while True:
            t0 = time.time()
            try:
                data = urllib.request.urlopen(self.url, timeout=self.timeout).read()
                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    yield None, "bad jpeg from /capture"
                    continue
                yield frame, None
            except Exception as e:
                yield None, f"capture error: {e}"
            elapsed = time.time() - t0
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)

    def close(self):
        pass


class FrameGrabber:
    def __init__(self, source):
        self.stream = make_stream(source) if isinstance(source, str) else source
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.fps = 0.0
        self.last_error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        t0, n = time.time(), 0
        while not self._stop.is_set():
            for frame, err in self.stream.frames():
                if self._stop.is_set():
                    break
                if frame is None:
                    self.last_error = err
                    continue
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                n += 1
                now = time.time()
                if now - t0 >= 1.0:
                    self.fps = n / (now - t0)
                    t0, n = now, 0

    def wait(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._seq > 0:
                    return True
            time.sleep(0.1)
        return False

    def latest(self):
        with self._lock:
            if self._frame is None:
                return None, -1
            return self._frame.copy(), self._seq

    def stop(self):
        self._stop.set()
        try:
            self.stream.close()
        except Exception:
            pass


def pick_stream(preferred=None, use_camera=False, probe=6.0):
    if use_camera:
        urls = ["0"]
    else:
        urls = []
        for u in (preferred, STREAM_URL, CAPTURE_URL):
            if u and u not in urls:
                urls.append(u)
    for url in urls:
        print(f"[video] trying {url} ...")
        grabber = FrameGrabber(url).start()
        if grabber.wait(probe):
            print(f"[video] connected: {url}  ({grabber.fps:.1f} fps)")
            return grabber
        print(f"[video] no frames from {url}")
        grabber.stop()
        time.sleep(0.3)
    raise SystemExit("no video source reachable - check camera power / IP")


def make_stream(url):
    if isinstance(url, str) and (url.isdigit() or url.endswith((".mp4", ".avi", ".mov"))):
        return WebcamStream(int(url) if url.isdigit() else url)
    if isinstance(url, str) and (url.rstrip("/").endswith("capture") or "capture?" in url):
        return CaptureStream(url)
    return ESP32Stream(url)

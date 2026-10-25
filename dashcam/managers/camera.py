"""
Camera initialization and management
Adopted from yolo_webcam.py CaptureThread pattern:
  - VideoCapture dibuka DI DALAM thread (FFmpeg-safe)
  - Auto-reconnect dengan exponential backoff kalau kamera lepas/error
  - CAP_PROP_BUFFERSIZE=1 → selalu ambil frame terbaru (low-latency)
  - _ready Event untuk sinkronisasi — caller tunggu sampai kamera siap
  - Mendukung: index integer, /dev/videoN, file video, RTSP URL
"""
import cv2
import os
import queue
import threading
import time
import subprocess
import numpy as np
import logging
from typing import Optional, Tuple

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FFmpegCapture — untuk UVC/USB via FFmpeg pipe (tetap dipertahankan sebagai
# backend alternatif jika cv2.VideoCapture gagal pada device tertentu)
# ─────────────────────────────────────────────────────────────────────────────
class FFmpegCapture:
    """
    FFmpeg-based camera capture yang meniru interface cv2.VideoCapture.
    Lebih stabil dari GStreamer untuk kamera UVC (OsmoAction3, webcam, dll).
    """

    def __init__(self, device: str, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self._device = device
        self._frame_size = width * height * 3  # BGR24
        self._proc: Optional[subprocess.Popen] = None
        self._opened = False
        self._start()

    def _start(self):
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',
            '-f', 'v4l2',
            '-input_format', 'mjpeg',
            '-video_size', f'{self.width}x{self.height}',
            '-framerate', str(self.fps),
            '-i', self._device,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self._frame_size * 4
            )
            # Warm-up: beri FFmpeg waktu untuk buka device dan mulai stream
            time.sleep(0.8)
            self._opened = self._proc.poll() is None
        except Exception as e:
            print(f"⚠️ FFmpeg failed to start: {e}")
            self._opened = False

    def isOpened(self) -> bool:
        if not self._opened or self._proc is None:
            return False
        return self._proc.poll() is None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.isOpened():
            return False, None
        try:
            raw = self._proc.stdout.read(self._frame_size)
            if len(raw) != self._frame_size:
                return False, None
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
            return True, frame.copy()
        except Exception:
            return False, None

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:  return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:          return float(self.fps)
        if prop_id == cv2.CAP_PROP_BUFFERSIZE:   return 1.0
        return 0.0

    def set(self, prop_id: int, value) -> bool:
        return False  # FFmpeg params fixed at init time

    def release(self):
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
            finally:
                self._proc = None
        self._opened = False


# ─────────────────────────────────────────────────────────────────────────────
# CaptureThread — FFmpeg-safe, VideoCapture dibuka DI DALAM thread
# Diadopsi dari yolo_webcam.py CaptureThread v1.3.0
# ─────────────────────────────────────────────────────────────────────────────
class CaptureThread(threading.Thread):
    """
    Thread kamera yang membuka VideoCapture di dalam thread sendiri (FFmpeg-safe).
    Mendukung auto-reconnect dengan exponential backoff.
    Caller harus panggil wait_ready() sebelum membaca width/height/fps.
    Frame diambil via get_frame() atau langsung dari queue self.q.
    """

    def __init__(self, source, q: queue.Queue,
                 reconnect_delay: float = 3.0, max_reconnect: int = 0,
                 fps_limit: float = 0.0, skip: int = 0):
        super().__init__(daemon=True, name="CaptureThread")
        self.source          = source
        self.q               = q
        self.reconnect_delay = reconnect_delay
        self.max_reconnect   = max_reconnect
        self.fps_limit       = fps_limit
        self.skip            = skip
        self.stopped         = False
        self.width           = 0
        self.height          = 0
        self.fps             = 30.0
        self._ready          = threading.Event()
        self._error          = None

        src_str = str(source).strip()
        # Kamera lokal (index) dan stream URL keduanya mendukung reconnect
        self.is_stream = src_str.isdigit() or any(
            src_str.lower().startswith(p)
            for p in ("rtsp://", "rtmp://", "http://", "https://", "udp://",
                      "/dev/video")
        )
        self.is_rtsp = src_str.lower().startswith("rtsp://")

    def wait_ready(self, timeout: float = 15.0) -> bool:
        """
        Tunggu sampai kamera berhasil dibuka (atau error).
        Return True jika berhasil, False jika timeout/error.
        """
        if not self._ready.wait(timeout=timeout):
            log.error(f"⏱️ Timeout {timeout}s menunggu kamera: {self.source}")
            return False
        if self._error:
            log.error(f"❌ Kamera error: {self._error}")
            return False
        return True

    def _open_capture(self) -> cv2.VideoCapture:
        """
        Buka VideoCapture DI DALAM thread (FFmpeg-safe).
        Set buffer kecil agar selalu dapat frame terbaru.
        """
        src = self.source
        try:
            src = int(src)  # "0" → 0 (integer index)
        except (ValueError, TypeError):
            pass

        # Low-latency RTSP: paksa transport TCP + timeout
        if self.is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|stimeout;5000000"
            )

        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # buffer kecil → frame selalu terbaru
        return cap

    def run(self):
        reconnect_count = 0

        while not self.stopped:
            cap = self._open_capture()

            if not cap.isOpened():
                cap.release()
                if not self.is_stream:
                    # File video yang tidak ada → error langsung, jangan reconnect
                    self._error = f"Tidak bisa membuka source: {self.source}"
                    self._ready.set()
                    return

                reconnect_count += 1
                if self.max_reconnect > 0 and reconnect_count > self.max_reconnect:
                    self._error = (
                        f"Gagal connect setelah {self.max_reconnect} percobaan: "
                        f"{self.source}"
                    )
                    try:
                        self.q.put_nowait(None)  # sinyal stop ke consumer
                    except queue.Full:
                        pass
                    self._ready.set()
                    return

                delay = min(self.reconnect_delay * (1.5 ** (reconnect_count - 1)), 30.0)
                log.warning(
                    f"⚠️ Kamera belum bisa dibuka (percobaan {reconnect_count}). "
                    f"Reconnect dalam {delay:.1f}s ..."
                )
                time.sleep(delay)
                continue

            # Koneksi berhasil — ambil metadata (hanya sekali, saat koneksi pertama)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            f = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if not self._ready.is_set():
                self.width, self.height, self.fps = w, h, f
                log.info(f"📷 Kamera: {self.source} | {w}x{h} @ {f:.1f} FPS")
                print(f"✅ Camera opened: {self.source} | {w}x{h} @ {f:.1f} FPS")
                self._ready.set()

            # Hitung interval minimum jika ada fps_limit
            min_interval = (1.0 / self.fps_limit) if self.fps_limit > 0 else 0.0
            last_t = time.time()
            reconnect_count = 0  # reset counter setelah berhasil connect
            frame_idx = 0

            # ── Loop baca frame ──────────────────────────────────────────
            while not self.stopped:
                ret, frame = cap.read()

                if not ret or frame is None:
                    # Frame gagal — coba reconnect kalau ini stream/kamera
                    log.warning(f"⚠️ Frame gagal dari {self.source}, mencoba reconnect ...")
                    break  # keluar dari inner loop → outer loop akan reconnect

                # Frame skip: proses 1 dari skip+1 frame (0=off)
                if self.skip > 0 and (frame_idx % (self.skip + 1)) != 0:
                    frame_idx += 1
                    continue

                # FPS limit: buang frame jika terlalu cepat
                if min_interval > 0:
                    now = time.time()
                    elapsed = now - last_t
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                    last_t = time.time()

                frame_idx += 1

                # Taruh frame ke queue; kalau penuh, buang frame lama (keep terbaru)
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    try:
                        self.q.get_nowait()   # buang frame lama
                    except queue.Empty:
                        pass
                    try:
                        self.q.put_nowait(frame)
                    except queue.Full:
                        pass

            cap.release()

            # Kalau sudah diminta stop, keluar
            if self.stopped:
                break

            # Reconnect dengan backoff
            if not self.is_stream:
                # Source bukan stream (file video habis) → selesai normal
                try:
                    self.q.put_nowait(None)
                except queue.Full:
                    pass
                return

            reconnect_count += 1
            if self.max_reconnect > 0 and reconnect_count > self.max_reconnect:
                log.error(f"❌ Reconnect limit {self.max_reconnect} tercapai.")
                try:
                    self.q.put_nowait(None)
                except queue.Full:
                    pass
                return

            delay = min(self.reconnect_delay * (1.5 ** (reconnect_count - 1)), 30.0)
            log.warning(f"🔄 Reconnect #{reconnect_count} dalam {delay:.1f}s ...")
            time.sleep(delay)

    def stop(self):
        self.stopped = True


# ─────────────────────────────────────────────────────────────────────────────
# CameraManager — factory untuk membuat CaptureThread yang sesuai
# ─────────────────────────────────────────────────────────────────────────────
class CameraManager:
    """
    Factory untuk membuat CaptureThread kamera.
    Tidak lagi membuka VideoCapture di main thread.
    """

    @staticmethod
    def _get_uvc_capabilities(camera_id: int) -> Tuple[int, int, int]:
        """
        Auto-detect resolusi dan FPS dari device UVC via v4l2-ctl.
        Return (width, height, fps). Default (1280, 720, 30) jika gagal.
        """
        default = (1280, 720, 30)
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', f'/dev/video{camera_id}', '--list-formats-ext'],
                capture_output=True, text=True, timeout=5
            )
            import re
            sizes = re.findall(
                r'Size: Discrete (\d+)x(\d+).*?Interval: Discrete [\d.]+s \(([\d.]+) fps\)',
                result.stdout, re.DOTALL
            )
            if not sizes:
                return default
            for w, h, fps in sizes:
                if int(w) == 1280 and int(h) == 720:
                    return (int(w), int(h), int(float(fps)))
            w, h, fps = sizes[0]
            return (int(w), int(h), int(float(fps)))
        except Exception:
            return default

    @staticmethod
    def _is_uvc_device(camera_id: int) -> bool:
        """Cek apakah /dev/videoN adalah kamera UVC (USB Video Class)."""
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', f'/dev/video{camera_id}', '--info'],
                capture_output=True, text=True, timeout=5
            )
            return 'uvcvideo' in result.stdout.lower()
        except Exception:
            return False

    @staticmethod
    def _check_jetson() -> bool:
        """Cek apakah berjalan di Jetson dengan nvargus-daemon aktif."""
        try:
            result = subprocess.run(
                ['systemctl', 'is-active', '--quiet', 'nvargus-daemon'],
                capture_output=True, timeout=3
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def create_capture_thread(camera_id: int, camera_type: str,
                               frame_queue: queue.Queue,
                               reconnect_delay: float = 3.0,
                               max_reconnect: int = 0,
                               fps_limit: float = 0.0) -> 'CaptureThread':
        """
        Buat CaptureThread yang sesuai dengan tipe kamera.
        VideoCapture dibuka di dalam thread (FFmpeg-safe).

        Untuk USB/UVC: source = integer index atau /dev/videoN string.
        Untuk Jetson CSI: source = GStreamer pipeline string (dibuka via CAP_GSTREAMER).
        Untuk generic/standard: source = integer index.
        """
        is_uvc = CameraManager._is_uvc_device(camera_id)

        # Tentukan source string berdasarkan tipe kamera
        if camera_type == 'usb' or (camera_type == 'auto' and is_uvc):
            # USB/UVC: gunakan integer index agar cv2.VideoCapture pilih backend terbaik
            # (cv2 akan pakai V4L2 pada Linux, sudah cukup untuk UVC)
            source = camera_id
            log.info(f"📷 Kamera USB/UVC terdeteksi: camera_id={camera_id}")

        elif camera_type == 'jetson' or (
            camera_type == 'auto' and not is_uvc and CameraManager._check_jetson()
        ):
            # Jetson CSI: source = GStreamer pipeline
            # ponytail: resolusi hardcoded 1280x720@30, bisa dijadikan parameter
            source = (
                f"nvarguscamerasrc sensor-id={camera_id} ! "
                f"video/x-raw(memory:NVMM), width=1280, height=720, "
                f"framerate=30/1 ! nvvidconv ! video/x-raw, format=BGR ! "
                f"appsink drop=true max-buffers=1"
            )
            log.info(f"📷 Kamera Jetson CSI: sensor-id={camera_id}")

        else:
            # Standard / fallback: integer index
            source = camera_id
            log.info(f"📷 Kamera standard: camera_id={camera_id}")

        return CaptureThread(
            source=source,
            q=frame_queue,
            reconnect_delay=reconnect_delay,
            max_reconnect=max_reconnect,
            fps_limit=fps_limit,
        )

    @staticmethod
    def initialize(camera_id: int, camera_type: str) -> Optional[object]:
        """
        [DEPRECATED] Membuka kamera di main thread — tidak FFmpeg-safe.
        Gunakan create_capture_thread() sebagai gantinya.
        Dipertahankan untuk kompatibilitas backward (mode video file).
        """
        log.warning("CameraManager.initialize() deprecated — gunakan create_capture_thread()")
        try:
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Camera initialized (fallback): OpenCV {camera_id}")
                    return cap
                cap.release()
        except Exception:
            pass
        print("❌ Camera initialization failed")
        return None

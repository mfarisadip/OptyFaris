"""
Async video recording via FFmpeg pipe.

Dua mode:
  - 'mjpeg' (default): MJPEG → .avi  — crash-safe, untuk live camera recording
  - 'h264'           : H.264 → .mp4  — ukuran kecil, universal, untuk video analysis
                       Otomatis coba h264_nvenc (Jetson HW) → fallback libx264
"""
import cv2
import queue
import threading
import subprocess
import numpy as np
import time
import os
from typing import Optional, Tuple


class Cv2WriterThread:
    """cv2.VideoWriter di thread terpisah.

    Encode H.264 per frame (~6 ms @720p) tidak lagi memblokir thread utama
    pipeline video. Queue berbatas + put() blocking → backpressure: kalau
    encoder lebih lambat dari produksi frame, pipeline melambat (zero-drop),
    bukan membuang frame.
    """

    def __init__(self, output_path: str, fps: float, size: Tuple[int, int],
                 queue_size: int = 90):
        self.output_path = output_path
        self.fps = fps
        self.size = size
        self.frames_written = 0

        self._q = queue.Queue(maxsize=queue_size)
        self._writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, size)
        if not self._writer.isOpened():
            self._writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        self._thread = None
        if self._writer.isOpened():
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name='Cv2VideoWriter')
            self._thread.start()

    def is_opened(self) -> bool:
        return self._writer.isOpened()

    def _loop(self):
        while True:
            frame = self._q.get()
            if frame is None:      # poison pill
                break
            try:
                self._writer.write(frame)
                self.frames_written += 1
            except Exception:
                continue           # jangan mati → put() di producer tidak deadlock

    def write(self, frame: np.ndarray):
        """Blocking put — backpressure menjaga zero-drop."""
        self._q.put(frame)

    def release(self):
        if self._thread is None:
            return
        try:
            self._q.put(None, timeout=5)
        except queue.Full:
            pass
        self._thread.join(timeout=30)
        self._writer.release()


class VideoManager:
    """Handle video recording with async background writing via FFmpeg pipe"""

    def __init__(self, output_path: Optional[str], fps: float, size: Tuple[int, int],
                 codec: str = 'mjpeg', queue_size: int = 300):
        """
        Args:
            output_path : path file output (ekstensi .mp4 selalu dipass dari luar)
            fps         : frame rate output
            size        : (width, height)
            codec       : 'mjpeg' → AVI crash-safe (live camera)
                          'h264'  → MP4 H.264 (video analysis)
            queue_size  : ukuran frame buffer async
                          300 = camera mode (burst-safe)
                           60 = video mode (hemat RAM ~1.4GB)
        """
        self.output_path = output_path
        self.fps = fps
        self.size = size
        self.codec = codec.lower()
        self._proc = None

        # Async writing — queue_size dikonfigurasi dari luar sesuai mode
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.writer_thread = None
        self.is_writing = False

        # Stats
        self.frames_written = 0
        self.frames_dropped = 0
        self.first_frame_time = None
        self.last_frame_time = None

    # ------------------------------------------------------------------
    def initialize(self, clean_mode: bool = False) -> bool:
        """Start FFmpeg pipe writer. Mode ditentukan oleh self.codec."""
        if not self.output_path:
            return True
        if self.is_writing:
            return True

        if self.codec == 'h264':
            return self._initialize_h264()
        else:
            return self._initialize_mjpeg()

    def _initialize_mjpeg(self) -> bool:
        """MJPEG → .avi  — crash-safe, untuk live camera."""
        path = self.output_path.replace('.mp4', '.avi')
        w, h = self.size
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}', '-r', str(self.fps),
            '-i', 'pipe:0',
            '-c:v', 'mjpeg', '-q:v', '5', '-an',
            path
        ]
        return self._start_ffmpeg(cmd, label=f'MJPEG/AVI → {path}')

    def _initialize_h264(self) -> bool:
        """H.264 → .mp4  — coba nvenc (Jetson HW), videotoolbox (Mac HW), fallback libx264."""
        path = self.output_path  # tetap .mp4
        w, h = self.size

        base_args = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}', '-r', str(self.fps),
            '-i', 'pipe:0', '-an',
            '-movflags', '+faststart',   # moov atom di depan → MP4 langsung bisa diputar
        ]

        # --- Coba h264_videotoolbox (Apple HW encoder) ---
        if self._try_encoder_available('h264_videotoolbox'):
            cmd = base_args + [
                '-c:v', 'h264_videotoolbox',
                '-q:v', '65',         # ~kualitas CRF18
                path
            ]
            if self._start_ffmpeg(cmd, label=f'H.264 VideoToolbox/MP4 → {path}'):
                return True
            print('⚠️  VideoToolbox gagal, fallback ke libx264...')

        # --- Coba h264_nvenc (Jetson hardware encoder) ---
        if self._try_encoder_available('h264_nvenc'):
            cmd = base_args + [
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',      # p1=fastest … p7=best quality
                '-rc', 'vbr',         # Variable Bitrate (Constant Quality mode)
                '-cq', '18',          # CQ 18 ≈ CRF 18, hampir tidak bisa dibedakan dari asli
                '-b:v', '0',          # biarkan bitrate auto-adjust sesuai konten
                '-maxrate', '20M',    # ceiling agar tidak meledak di scene kompleks
                path
            ]
            if self._start_ffmpeg(cmd, label=f'H.264 NVENC CQ18/MP4 → {path}'):
                return True
            print('⚠️  NVENC gagal, fallback ke libx264...')

        # --- Fallback libx264 (software) ---
        cmd = base_args + [
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',             # CRF 18: kualitas tinggi, hampir lossless secara visual
            path
        ]
        return self._start_ffmpeg(cmd, label=f'H.264 libx264 CRF18/MP4 → {path}')

    def _try_encoder_available(self, encoder: str) -> bool:
        """Cek apakah encoder FFmpeg tertentu tersedia di sistem ini."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True, text=True, timeout=5
            )
            return encoder in result.stdout
        except Exception:
            return False

    def _start_ffmpeg(self, cmd: list, label: str) -> bool:
        """Start FFmpeg subprocess dan writer thread."""
        w, h = self.size
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=w * h * 3 * 4
            )
            # Tunggu sebentar — kalau FFmpeg langsung exit berarti error
            import time as _t; _t.sleep(0.3)
            if self._proc.poll() is not None:
                print(f'❌ FFmpeg exit early: {label}')
                return False

            self.is_writing = True
            self.writer_thread = threading.Thread(
                target=self._writer_loop, daemon=True, name='VideoWriterPipe')
            self.writer_thread.start()
            print(f'🎬 Video writer: {label}')
            return True
        except FileNotFoundError:
            print('❌ FFmpeg not found — cannot record video')
            return False
        except Exception as e:
            print(f'❌ Video writer failed: {e}')
            return False

    # ------------------------------------------------------------------
    def _writer_loop(self):
        """Background thread: pop frame dari queue → tulis ke FFmpeg stdin"""
        while self.is_writing or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is None:         # poison pill
                    break
                if self._proc and self._proc.stdin:
                    try:
                        self._proc.stdin.write(frame.tobytes())
                        self.frames_written += 1
                    except BrokenPipeError:
                        break
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                continue

    # ------------------------------------------------------------------
    def write_frame(self, frame: np.ndarray):
        """Enqueue frame untuk ditulis secara async (non-blocking)."""
        if not self.is_writing or self._proc is None:
            return

        now = time.time()
        if self.first_frame_time is None:
            self.first_frame_time = now
        self.last_frame_time = now

        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            self.frames_dropped += 1

    # ------------------------------------------------------------------
    def release(self):
        """Stop writer thread dan tutup FFmpeg pipe."""
        if not self.is_writing:
            return

        print("🛑 Stopping video writer...")
        self.is_writing = False

        # Poison pill
        try:
            self.frame_queue.put(None, timeout=1)
        except queue.Full:
            pass

        # Tunggu writer thread selesai flush
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=8)

        # Tutup stdin FFmpeg → FFmpeg akan finalize container
        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

        # Print stats
        total = self.frames_written + self.frames_dropped
        if total > 0:
            drop_pct = self.frames_dropped / total * 100
            print(f"📊 Video stats: {self.frames_written} written, "
                  f"{self.frames_dropped} dropped ({drop_pct:.1f}%)")
            if self.first_frame_time and self.last_frame_time and self.frames_written > 1:
                dur = self.last_frame_time - self.first_frame_time
                actual_fps = self.frames_written / dur
                print(f"⚙️  Configured FPS: {self.fps:.1f} | Actual FPS: {actual_fps:.1f}")

        print("✅ Video writer released")

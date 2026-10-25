"""
Main application orchestration
"""
import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages/")

import cv2
import time
import os
import queue
import threading
import numpy as np
from datetime import datetime
from typing import Dict, Any
from ultralytics import YOLO
import torch

from dashcam.models.detections import Detections

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from dashcam.models.detection_stats import DetectionStats
from dashcam.managers.gps import GPSManager
from dashcam.managers.camera import CameraManager
from dashcam.managers.video import VideoManager, Cv2WriterThread
from dashcam.managers.detection import DetectionManager
from dashcam.managers.device import DeviceManager
from dashcam.managers.oled import OLEDManager
from dashcam.managers.telegram_notifier import TelegramNotifier
from dashcam.core.detector import LowLatencyDetector
from dashcam.core.lane import LaneDetector, draw_lane
from dashcam.config.settings import parse_arguments


class DashcamApp:
    """Main dashcam application"""

    def __init__(self, args):
        """Initialize with parsed arguments"""
        # Setup paths
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        if args.source == 'camera':
            base_name = f"dashcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            base_name = os.path.splitext(os.path.basename(args.video_path))[0]

        video_path = os.path.join(output_dir, f"{base_name}_annotated.mp4") if args.save_video else None
        report_path = os.path.join(output_dir, f"{base_name}_report.txt")

        # Get device
        device_str, device_arg = DeviceManager.get_device(args.device, args.cuda_device)

        # Build configuration
        self.config = {
            'source': args.source,
            'video_path': args.video_path,
            'camera_id': args.camera_id,
            'camera_type': args.camera_type,
            'save_video': args.save_video,
            'show_cam': args.show_cam,
            'device': device_str,
            'device_arg': device_arg,
            'model': args.model,
            'conf': args.conf,
            'iou': args.iou,
            'imgsz': args.imgsz,
            'top_line': args.top_line,
            'bottom_line': args.bottom_line,
            'no_zone': args.no_zone,
            'vehicle_token': args.vehicle_token,
            'clean_video': args.clean_video,
            'save_screenshots': args.save_screenshots,
            'target_fps': args.target_fps,
            'buffer_size': args.buffer_size,
            'output_path': video_path,
            'report_path': report_path,
            'api_base': 'https://airoad.roots.web.id',
            # FP16 half-precision (default ON untuk CUDA, seperti yolo_tracker.py)
            'half': 'cuda' in device_str.lower() if device_str else False,
            # Class IDs untuk inference (seperti yolo_tracker.py — filter pothole di level YOLO)
            'detect_classes': None,
            'max_det': 100,
            'no_lane': args.no_lane,
            'lane_work_h': args.lane_work_h,
            'lane_skip': args.lane_skip,
            'lane_roi_top': args.lane_roi_top,
            'lane_roi_bottom': args.lane_roi_bottom,
            'lanes': args.lanes,
            'lane_yellow': not args.lane_white_only,
        }

        self.gps_manager = GPSManager()
        self.lane_detector = None if args.no_lane else LaneDetector(
            work_h=args.lane_work_h, skip=args.lane_skip,
            roi_top_y_frac=args.lane_roi_top, roi_bottom_y_frac=args.lane_roi_bottom,
            total_lanes=args.lanes, use_yellow=not args.lane_white_only,
        )
        self.oled_manager = OLEDManager()   # OLED display (auto-detect bus)
        self.tg_notifier  = TelegramNotifier()  # Telegram monitoring
        self.detection_manager = None
        self.video_manager = None
        self.camera = None
        self.stats = DetectionStats()

        # API configuration
        self.api_enabled = bool(self.config.get('vehicle_token') and REQUESTS_AVAILABLE)
        if self.api_enabled:
            print(f"✅ API enabled")
        elif self.config.get('vehicle_token'):
            print("⚠️ API token provided but requests not available")

    def initialize(self) -> bool:
        """Initialize all components"""
        # Initialize GPS
        if not self.gps_manager.start():
            print("⚠️ GPS initialization failed - continuing without GPS")
            # return False # Don't block app if GPS fails

        # Initialize OLED display
        self.oled_manager.start()

        # Initialize Telegram notifier
        self.tg_notifier.start()

        # Load model
        print(f"🔄 Loading model: {self.config['model']}")
        self.model = YOLO(self.config['model'], task='detect')

        # Get dynamic class names
        print(f"📋 Model classes: {list(self.model.names.values())}")

        # ── Filter kelas untuk inference (seperti yolo_tracker.py) ──────────
        # Program utama deteksi SEMUA kelas (pothole + crack + rutting),
        # jadi TIDAK restrict classes di level inference.
        # YOLO NMS per-class secara default, jadi tidak ada cross-class suppression.
        self.config['detect_classes'] = None

        # Initialize detection manager
        self.detection_manager = DetectionManager(self.model, self.config)

        # ── Filter kelas di level inference ─────────────────────────────────
        # Kirim daftar class ID yang termapping (pothole/crack) langsung ke
        # predict() → kelas tanpa mapping (mis. "Repair" di model universal)
        # dibuang SEBELUM NMS, bukan sesudahnya. Semantik output sama karena
        # filter valid_classes di DetectionManager membuang kelas itu juga.
        if self.detection_manager.valid_classes:
            self.config['detect_classes'] = sorted(self.detection_manager.valid_classes)
            print(f"🎯 Inference classes: {self.config['detect_classes']}")

        # GPU warmup — prevent degraded first ~100 frames
        is_cuda = 'cuda' in str(self.config.get('device', '')).lower()
        if is_cuda and torch.cuda.is_available():
            warmup_iters = 3
            print(f"🔥 GPU warmup ({warmup_iters}x) ...")
            try:
                dummy = np.zeros((self.config['imgsz'], self.config['imgsz'], 3), dtype=np.uint8)
                for _ in range(warmup_iters):
                    self.model.predict(source=dummy, imgsz=self.config['imgsz'],
                                       conf=self.config['conf'], verbose=False,
                                       device=self.config.get('device_arg'))
                torch.cuda.synchronize()
                print("✅ Warmup selesai.")
            except Exception as e:
                print(f"⚠️ Warmup gagal (diabaikan): {e}")

        # Initialize camera or video file
        if self.config['source'] == 'camera':
            # CaptureThread: VideoCapture dibuka di dalam thread (FFmpeg-safe),
            # frame masuk ke queue yang dikonsumsi LowLatencyDetector.
            cap_queue = queue.Queue(maxsize=2)
            self.cap_queue = cap_queue
            self.camera = CameraManager.create_capture_thread(
                self.config['camera_id'],
                self.config['camera_type'],
                cap_queue,
            )
            self.camera.start()
            if not self.camera.wait_ready():
                print("❌ Camera initialization failed")
                return False

            # Get camera properties (tersedia setelah wait_ready())
            w = self.camera.width
            h = self.camera.height
            fps = self.camera.fps or 30

            self.config['video_info'] = {'width': w, 'height': h, 'fps': fps, 'total_frames': None}

            # Initialize video manager if saving
            if self.config.get('save_video'):
                # Cap 30fps untuk camera — pipeline async bisa tidak stabil di FPS tinggi
                output_fps = min(fps, 30)
                self.video_manager = VideoManager(
                    self.config.get('output_path'),
                    output_fps,
                    (w, h),
                    codec='mjpeg',   # crash-safe untuk live recording
                    queue_size=300   # buffer besar untuk burst async camera
                )
                self.video_manager.initialize(self.config.get('clean_video'))
        
        elif self.config['source'] == 'video':
            # Initialize video file
            video_path = self.config['video_path']
            if not os.path.exists(video_path):
                print(f"❌ Video file not found: {video_path}")
                return False
            # Increase OpenCV ffmpeg read attempts (helps with some containers/multi-stream files)
            try:
                os.environ.setdefault('OPENCV_FFMPEG_READ_ATTEMPTS', '200000')
            except Exception:
                pass

            # Prefer explicit FFMPEG backend if available, fallback otherwise
            try:
                self.camera = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            except Exception:
                self.camera = cv2.VideoCapture(video_path)
            if not self.camera.isOpened():
                print(f"❌ Failed to open video file: {video_path}")
                return False
            
            # Get video properties
            w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"📹 Video loaded: {video_path}")
            print(f"   Resolution: {w}x{h}")
            print(f"   FPS: {fps}")
            print(f"   Total frames: {total_frames}")
            
            self.config['video_info'] = {'width': w, 'height': h, 'fps': fps, 'total_frames': total_frames}
            
            # Video mode: cv2.VideoWriter dibuat langsung di _run_video_sync()
            # Simpan fps & size ke config agar bisa diakses di sana
            self.config['source_fps']  = fps
            self.config['source_size'] = (w, h)
            self.video_manager = None  # tidak dipakai untuk video source


        return True

    def run(self):
        """Main run method"""
        print("\n" + "="*60)
        print("🚀 ULTRA-LOW LATENCY DASHCAM")
        print("="*60)

        if not self.initialize():
            print("❌ Initialization failed")
            return

        detector = None
        try:
            start_time = time.time()

            if self.config['source'] == 'video':
                # ── SYNCHRONOUS MODE: setiap frame diproses tanpa drop ──
                frame_count = self._run_video_sync()
            else:
                # ── ASYNC CAMERA MODE: multi-thread seperti sebelumnya ──
                detector = LowLatencyDetector(
                    self.model, self.config,
                    self.detection_manager, self.stats, self.gps_manager
                )
                detector.start(self.cap_queue, self.video_manager)

                print("\n🚀 Dashcam started")
                if self.config.get('show_cam'):
                    print("📺 Live view enabled — tekan 'q' untuk keluar")
                else:
                    print("📡 Monitoring for road damage...")
                print("⏹️ Press Ctrl+C to stop")

                frame_count = 0
                yolo_processed_at_start = 0
                last_fps_time = time.time()
                try:
                    while detector.running:
                        frame_count += 1
                        gps_pos = self.gps_manager.get_position()
                        gps_locked = (
                            gps_pos.get('last_update') is not None
                            and (gps_pos.get('latitude', 0.0) != 0.0
                                 or gps_pos.get('longitude', 0.0) != 0.0)
                            and (datetime.now() - gps_pos['last_update']).total_seconds() < 5
                        )
                        self.oled_manager.update(
                            camera_running=True,
                            gps_locked=gps_locked,
                            detection_counts=dict(self.stats.counts),
                            gps_position=gps_pos
                        )
                        self.tg_notifier.update(
                            camera_running=True,
                            gps_locked=gps_locked,
                            gps_position=gps_pos,
                            detection_counts=dict(self.stats.counts)
                        )

                        # ── Display from main thread (macOS requirement) ──
                        if self.config.get('show_cam'):
                            latest = detector.get_latest_frame()[0]
                            if latest is not None:
                                cv2.imshow('Dashcam Live View - Detections', latest)
                            else:
                                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                                cv2.putText(placeholder, 'Waiting...', (200, 240),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                cv2.imshow('Dashcam Live View - Detections', placeholder)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                detector.running = False
                                break

                        if frame_count % 150 == 0:
                            now = time.time()
                            elapsed = now - last_fps_time
                            if elapsed > 0:
                                yolo_delta = detector.yolo_processed_count - yolo_processed_at_start
                                real_fps = yolo_delta / elapsed
                                print(f"📊 YOLO FPS: {real_fps:.1f} | Detections: {self.stats.get_total_count()}")
                                last_fps_time = now
                                yolo_processed_at_start = detector.yolo_processed_count
                        time.sleep(0.001)
                    print("\n✅ Stream ended")
                except KeyboardInterrupt:
                    print("\n⚠️ Interrupted by user")
                finally:
                    if self.config.get('show_cam'):
                        cv2.destroyAllWindows()

            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            self._print_summary(avg_fps, frame_count, detector)

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if detector is not None:
                detector.stop()
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print("\n🛑 Shutting down...")
        if self.tg_notifier:
            self.tg_notifier.stop()
        if self.oled_manager:
            self.oled_manager.stop()
        if self.gps_manager:
            self.gps_manager.stop()
        if self.camera:
            if hasattr(self.camera, 'stop'):
                self.camera.stop()
                self.camera.join(timeout=3)
            else:
                self.camera.release()
        if self.video_manager:
            self.video_manager.release()
        print("✅ Cleanup complete")

    def _print_summary(self, avg_fps: float, frame_count: int, detector=None):
        """Print detection summary"""
        print("\n" + "="*60)
        print("📊 DASHCAM SUMMARY")
        print("="*60)

        if self.config['source'] == 'camera':
            print(f"📹 Source: Camera {self.config['camera_id']}")
            if detector:
                stats = detector.get_stats()
                print(f"📊 Camera FPS : {stats['camera_fps']:.1f}")
                print(f"🧠 YOLO FPS   : {stats['yolo_fps']:.1f}")
                print(f"📦 YOLO Proc : {stats['yolo_processed']}/{stats['camera_captured']} frames")
                if stats['dropped_by_buffer'] > 0:
                    print(f"⚠️  Dropped   : {stats['dropped_by_buffer']} ({stats['drop_rate']:.1f}%)")
                else:
                    print(f"✅ No drops   : 0 (0.0%)")
        else:
            print(f"📹 Source: Video file - {self.config['video_path']}")

        print(f"🖥️  Device: {self.config['device']}")
        print(f"⏱️  Processing time: {frame_count/avg_fps:.2f}s")
        print(f"🚀 Processing FPS: {avg_fps:.2f}")
        print(f"📊 Total Frames processed: {frame_count}")

        print(f"\n📈 Detection Counts:")
        for class_name, count in self.stats.counts.items():
            print(f"   {class_name}: {count}")
        print(f"\n📦 Total: {self.stats.get_total_count()}")

    # ------------------------------------------------------------------
    # SYNCHRONOUS VIDEO PROCESSING
    # ------------------------------------------------------------------

    def _run_video_sync(self) -> int:
        """
        Proses video file frame-by-frame secara synchronous.
        Inference YOLO jalan di worker thread; encode video di Cv2WriterThread.
        Queue berbatas + backpressure = zero drop.
        Returns: jumlah frame yang diproses.
        """
        total_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps   = self.config.get('source_fps', 30)
        source_size  = self.config.get('source_size', (1280, 720))
        frame_num = 0
        imgsz = self.config.get('imgsz', 640)

        # FPS tracking
        fps_start = time.time()
        fps_smooth = 0.0
        fps_alpha = 0.2

        # Video source is video — we compute timestamp from frame position
        is_video_source = (self.config.get('source') == 'video')

        print(f"\n📽️  Synchronous video mode (cv2.VideoWriter)")
        print(f"   Total frames : {total_frames}")
        print(f"   Source FPS   : {source_fps}")
        print(f"   Output size  : {source_size[0]}x{source_size[1]}")
        print(f"   Model input  : imgsz={imgsz}")
        print(f"   Press Ctrl+C to stop early\n")

        # --- Writer async: cv2.VideoWriter jalan di thread terpisah ---
        # Encode (~6ms/frame @720p) tidak lagi memblokir thread utama, jadi
        # throughput pipeline dibatasi inference YOLO, bukan encode.
        # Queue berbatas + blocking put = backpressure (zero-drop).
        writer = None
        if self.config.get('save_video') and self.config.get('output_path'):
            writer = Cv2WriterThread(
                self.config['output_path'], source_fps, source_size
            )
            if writer.is_opened():
                print(f"🎬 Output: {self.config['output_path']} @ {source_fps}fps (async writer)")
            else:
                print(f"❌ VideoWriter gagal dibuka: {self.config['output_path']}")
                writer = None

        if self.config.get('show_cam'):
            cv2.namedWindow('Video Analysis', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Video Analysis', 800, 600)

        # ── Worker: decode + YOLO inference (GPU tetap sibuk selama main thread
        #    meng-annotasi & menulis). Queue menahan (frame, frame_num, t_ms).
        # Support optional batching for video inference to increase GPU throughput.
        det_q = queue.Queue(maxsize=8)
        cap_error = []

        batch_size = max(1, int(self.config.get('batch_size', 1)))

        def _infer_worker():
            buffer = []
            while True:
                item = cap_q.get()
                if item is None:
                    # flush remaining
                    if buffer:
                        _process_batch(buffer)
                    break
                buffer.append(item)
                if len(buffer) >= batch_size:
                    _process_batch(buffer)
                    buffer = []

        def _process_batch(buffer_items):
            frames = [it[0] for it in buffer_items]
            indices = [it[1] for it in buffer_items]
            tms_list = [it[2] for it in buffer_items]
            try:
                if batch_size == 1:
                    results = [self.model.predict(
                        source=frames[0], imgsz=imgsz,
                        conf=self.config['conf'], iou=self.config['iou'],
                        device=self.config.get('device_arg'), quantize=16 if self.config.get('half', False) else None,
                        verbose=False, classes=self.config.get('detect_classes'),
                        max_det=self.config.get('max_det', 100), agnostic_nms=False,
                    )]
                else:
                    # ultralytics YOLO can accept list of images for batch
                    results = self.model.predict(
                        source=frames, imgsz=imgsz,
                        conf=self.config['conf'], iou=self.config['iou'],
                        device=self.config.get('device_arg'), quantize=16 if self.config.get('half', False) else None,
                        verbose=False, classes=self.config.get('detect_classes'),
                        max_det=self.config.get('max_det', 100), agnostic_nms=False,
                    )
                # Normalize results to list
                if not isinstance(results, list):
                    results = list(results)
                # results correspond to frames order
                for fr, idx, tms, res in zip(frames, indices, tms_list, results):
                    det_q.put((fr, idx, tms, Detections.from_ultralytics(res[0]) if res and len(res) > 0 else Detections.empty()))
            except Exception as e:
                cap_error.append(e)
                for fr, idx, tms in zip(frames, indices, tms_list):
                    det_q.put((fr, idx, tms, None))

        cap_q = queue.Queue(maxsize=8)
        infer_thread = threading.Thread(target=_infer_worker, daemon=True, name='yolo-infer')
        infer_thread.start()

        try:
            stop = False
            while not stop:
                iter_t0 = time.time()
                # Robust read: retry a few times on transient decode/read failures
                read_retry_limit = 8
                read_attempt = 0
                ret = False
                frame = None
                while read_attempt < read_retry_limit:
                    ret, frame = self.camera.read()
                    if ret:
                        break
                    read_attempt += 1
                    time.sleep(0.01)
                if not ret:
                    # If we reached EOF, break; otherwise log and continue
                    # Check if we're at end by comparing position to total frames
                    try:
                        pos = int(self.camera.get(cv2.CAP_PROP_POS_FRAMES))
                        total = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        if total > 0 and pos >= total:
                            print("\n\n✅ Semua frame selesai diproses")
                            break
                    except Exception:
                        pass
                    # transient failure not at EOF — warn and continue
                    print("[WARN] frame read failed repeatedly; skipping frame")
                    continue

                frame_num += 1
                h, w = frame.shape[:2]
                # PTS frame ini (ms) dari container — frame-akurat, VFR-safe
                t_ms = self.camera.get(cv2.CAP_PROP_POS_MSEC)

                # Feed ke worker (block saat penuh → backpressure, zero drop)
                cap_q.put((frame, frame_num, t_ms))

                # FPS calculation
                elapsed = time.time() - fps_start
                if elapsed > 0:
                    inst_fps = 1.0 / elapsed
                    fps_smooth = (fps_alpha * inst_fps + (1 - fps_alpha) * fps_smooth
                                  if fps_smooth > 0 else inst_fps)
                fps_start = time.time()

                # --- Ambil hasil YOLO frame INI (urutan terjaga oleh 2 queue) ---
                frame, done_num, t_ms, results = det_q.get()
                if cap_error:
                    raise cap_error[0]

                if results is not None:
                    detections = results
                else:
                    detections = Detections.empty()

                # Detections already at full frame resolution — no scaling needed

                # --- Filter kelas valid ---
                detections = self.detection_manager.filter_detections(detections)

                # --- Zone filter + Stats + Annotasi ---
                gps_data = self.gps_manager.get_position()

                # Video timestamp (for video source only — PTS asli dari container,
                # akurat utk VFR; fallback done_num/fps jika PTS tidak tersedia)
                video_ts = ""
                if is_video_source:
                    if t_ms > 0:
                        ms_total = int(round(t_ms))
                    elif source_fps > 0:
                        ms_total = int(round((done_num - 1) * 1000.0 / source_fps))
                    else:
                        ms_total = 0
                    hh, rem = divmod(ms_total, 3_600_000)
                    mm, rem = divmod(rem, 60_000)
                    ss, ms = divmod(rem, 1000)
                    video_ts = f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

                # --- Lane detection (ringan: 360p + cache tiap lane-skip frame) ---
                lane_result = self.lane_detector.detect(frame) if self.lane_detector is not None else None

                annotated, _ = self.detection_manager.process_detections(
                    detections, frame, frame.shape, done_num, gps_data, self.stats,
                    fps=fps_smooth, video_timestamp=video_ts, lane_result=lane_result
                )

                # --- Gambar garis lane (teks info sudah di overlay) ---
                if lane_result:
                    draw_lane(annotated, lane_result)

                # --- Tulis frame (enqueue non-blocking di thread utama;
                #     encode jalan di Cv2WriterThread, backpressure zero-drop) ---
                if writer:
                    writer.write(annotated)

                # --- Live view (opsional) — pace ke source FPS agar playback normal ---
                if self.config.get('show_cam'):
                    cv2.imshow('Video Analysis', annotated)
                    # sisa waktu frame sesuai FPS sumber (min 1ms untuk GUI event pump)
                    delay_ms = max(1, int(round(1000.0 / source_fps - (time.time() - iter_t0) * 1000)))
                    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                        print("\n⚠️ Dihentikan oleh user")
                        break

                # --- Update OLED & Telegram tiap 30 frame ---
                if done_num % 30 == 0:
                    gps_pos = self.gps_manager.get_position()
                    gps_locked = (
                        gps_pos.get('last_update') is not None
                        and (gps_pos.get('latitude', 0.0) != 0.0
                             or gps_pos.get('longitude', 0.0) != 0.0)
                        and (datetime.now() - gps_pos['last_update']).total_seconds() < 5
                    )
                    self.oled_manager.update(
                        camera_running=True, gps_locked=gps_locked,
                        detection_counts=dict(self.stats.counts), gps_position=gps_pos
                    )
                    self.tg_notifier.update(
                        camera_running=True, gps_locked=gps_locked,
                        gps_position=gps_pos, detection_counts=dict(self.stats.counts)
                    )

                # --- Progress bar (tiap 30 frame — I/O konsol tiap frame ikut
                #     membebani thread utama) ---
                if total_frames > 0 and (done_num % 30 == 0 or done_num == total_frames):
                    pct = done_num / total_frames
                    bar = '█' * int(30 * pct) + '░' * (30 - int(30 * pct))
                    total_det = self.stats.get_total_count()
                    print(
                        f"\r  [{bar}] {done_num}/{total_frames} "
                        f"({pct*100:.1f}%) | det:{total_det}",
                        end='', flush=True
                    )

        except KeyboardInterrupt:
            print("\n⚠️ Dihentikan oleh user")
        finally:
            cap_q.put(None)          # stop worker
            infer_thread.join(timeout=5)
            if writer:
                writer.release()
                print(f"\n✅ Video disimpan: {self.config.get('output_path')}")
                print(f"   Frames : {writer.frames_written} (0 drop)")
            if self.config.get('show_cam'):
                cv2.destroyAllWindows()

        return frame_num

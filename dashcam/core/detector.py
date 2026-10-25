"""
Low-latency detection engine with multi-threaded processing

Performance notes (v2.0):
  - Pure NumPy Detections (no supervision import)
  - Zone filter vectorized with NumPy
  - API crop uses small region (not full frame.copy())
  - Single frame.copy() in _draw_latest_detections_both for video + display
  - Shared pub-frame buffer: camera publishes once, video thread reads directly (no second queue copy)
"""
import cv2
import time
import queue
import threading
import torch
import numpy as np
from typing import Dict, Any

from dashcam.models.detections import Detections
from dashcam.core.lane import draw_lane, LaneDetector

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False


class LowLatencyDetector:
    """Ultra-low latency detector for headless dashcam with live view showing detections"""

    def __init__(self, model, config, detection_manager, stats, gps_manager):
        self.model = model
        self.config = config
        self.detection_manager = detection_manager
        self.stats = stats
        self.gps_manager = gps_manager

        self.frame_buffer     = queue.Queue(maxsize=config.get('buffer_size', 30))
        self.api_queue        = queue.Queue(maxsize=50)

        self.running = False
        self.latest_frame = None
        self.latest_frame_id = -1
        self.latest_frame_lock = threading.Lock()

        # Shared frame buffer: camera publishes here, video thread reads
        self._pub_frame = None
        self._pub_lock = threading.Lock()

        self.camera_frame_count  = 0
        self.yolo_queued_count   = 0
        self.yolo_processed_count = 0
        self.frame_count_lock    = threading.Lock()
        self._camera_count_lock  = threading.Lock()

        self.latest_detections = []
        self.latest_detections_lock = threading.Lock()
        self.latest_lane = None

        self.zone_top_y    = None
        self.zone_bottom_y = None

        self.cuda_stream = None
        if 'cuda' in str(config.get('device', '')).lower() and torch.cuda.is_available():
            self.cuda_stream = torch.cuda.Stream()

        self.target_fps = config.get('target_fps', 30)
        self.show_cam = config.get('show_cam', False)
        self.lane_detector = None if config.get('no_lane') else LaneDetector(
            work_h=config.get('lane_work_h', 360), skip=config.get('lane_skip', 2),
            roi_top_y_frac=config.get('lane_roi_top', 0.60),
            roi_bottom_y_frac=config.get('lane_roi_bottom', 1.0),
            total_lanes=config.get('lanes'),
            use_yellow=config.get('lane_yellow', True),
        )
        self.display_running = False
        self.video_manager = None
        self.start_time = None
        self.end_time = None

    def get_stats(self) -> Dict[str, Any]:
        elapsed = (self.end_time or time.time()) - (self.start_time or time.time())
        cc = self.camera_frame_count
        yq = self.yolo_queued_count
        yp = self.yolo_processed_count
        dropped = cc - yq
        return {
            'elapsed_time':     elapsed,
            'camera_captured':  cc,
            'yolo_queued':      yq,
            'yolo_processed':   yp,
            'dropped_by_buffer': dropped,
            'drop_rate': (dropped / cc * 100) if cc > 0 else 0,
            'camera_fps': cc / elapsed if elapsed > 0 else 0,
            'yolo_fps':   yp / elapsed if elapsed > 0 else 0,
        }

    # ── Thread 1: Camera capture ──────────────────────────────────────────
    # Membaca dari queue CaptureThread (FFmpeg-safe: VideoCapture dibuka
    # di dalam CaptureThread, bukan di sini).
    # None di queue = sinyal stop (kamera mati / reconnect limit habis).
    def camera_thread(self, cap_thread_queue: queue.Queue):
        while self.running:
            try:
                frame = cap_thread_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # None = sinyal stop dari CaptureThread
            if frame is None:
                self.running = False
                break

            with self._camera_count_lock:
                self.camera_frame_count += 1

            # Clean record: raw cam frame di sini, lepas dari YOLO/draw
            if self.config.get('clean_video') and self.video_manager:
                self.video_manager.write_frame(frame)

            with self.frame_count_lock:
                frame_id = self.yolo_queued_count

            item = {'frame': frame, 'frame_id': frame_id, 'timestamp': time.time()}
            try:
                self.frame_buffer.put_nowait(item)
                with self.frame_count_lock:
                    self.yolo_queued_count += 1
            except queue.Full:
                pass

            # Publish frame for video thread (no extra queue, no copy)
            with self._pub_lock:
                self._pub_frame = frame

    # ── Thread 2: YOLO processing ─────────────────────────────────────────
    def processing_thread(self):
        while self.running or not self.frame_buffer.empty():
            try:
                item = self.frame_buffer.get(timeout=0.05)
                self._process_single_frame(item)
            except queue.Empty:
                continue

    # ── Thread 3: Display (+ annotated record jika bukan clean) ────────────
    def annotated_video_thread(self):
        while self.running:
            with self._pub_lock:
                frame = self._pub_frame

            if frame is None:
                time.sleep(0.001)
                continue

            if self.config.get('clean_video'):
                # Record sudah di camera_thread; di sini cuma overlay live view
                display_frame = self._draw_latest_detections(frame, draw_lines=True)
            else:
                video_frame, display_frame = self._draw_latest_detections_both(frame)
                if self.video_manager:
                    self.video_manager.write_frame(video_frame)

            with self.latest_frame_lock:
                self.latest_frame = display_frame

    # ── Thread 4: API sender ───────────────────────────────────────────────
    def api_sender_thread(self):
        while self.running or not self.api_queue.empty():
            try:
                task = self.api_queue.get(timeout=0.5)
                if task is None:
                    break
                fn, args, kwargs = task
                try:
                    fn(*args, **kwargs)
                except Exception:
                    pass
                self.api_queue.task_done()
            except queue.Empty:
                continue

    # ── Display is handled from main thread (macOS requirement) ────────────
    # cv2.imshow / waitKey must run on the main thread.
    # See application.py main loop for the display polling.

    # ── Frame processing ──────────────────────────────────────────────────
    def _process_single_frame(self, item):
        frame    = item['frame']
        frame_id = item['frame_id']
        h, w     = frame.shape[:2]
        imgsz    = self.config.get('imgsz', 640)

        # Run inference
        predict_kwargs = dict(
            source=frame, conf=self.config['conf'], iou=self.config['iou'],
            imgsz=imgsz, verbose=False, quantize=16 if self.config.get('half', False) else None,
            classes=self.config.get('detect_classes'),
            max_det=self.config.get('max_det', 100), agnostic_nms=False,
        )
        if self.config.get('device_arg'):
            predict_kwargs['device'] = self.config['device_arg']

        if self.cuda_stream:
            with torch.cuda.stream(self.cuda_stream):
                try:
                    results = self.model.predict(**predict_kwargs)
                except Exception:
                    results = self.model.predict(**predict_kwargs)
        else:
            results = self.model.predict(**predict_kwargs)

        # Parse → pure NumPy Detections (no supervision)
        detections = Detections.from_ultralytics(results[0]) if results and len(results) > 0 else Detections.empty()

        # Filter + zone
        detections = self._filter_detections(detections, frame.shape)

        # Track
        if len(detections) > 0:
            detections = self.detection_manager.spatial_tracker.update_with_detections(detections)

        # Process stats + API
        gps_data = self.gps_manager.get_position()
        self._process_detections(detections, frame, frame_id, gps_data)

        # Update latest_detections for video overlay
        new_dets = []
        for i in range(len(detections)):
            cid  = int(detections.class_id[i])
            dtype = self.detection_manager.detection_mappings.get(cid, '')
            cname = dtype if dtype else self.model.names.get(cid, 'unknown').lower().replace(' ', '_')
            conf = float(detections.confidence[i])
            bbox = detections.xyxy[i]
            color = (0, 0, 255) if dtype == 'pothole' else (0, 255, 0) if dtype == 'crack' else (255, 165, 0)
            new_dets.append((bbox, cname, conf, color))

        with self.latest_detections_lock:
            self.latest_detections = new_dets
            self.zone_top_y    = int(h * self.config['top_line'])
            self.zone_bottom_y = int(h * self.config['bottom_line'])

        # Lane detection — jalan di thread YOLO ini (cache internal tiap lane_skip frame)
        lane_result = self.lane_detector.detect(frame) if self.lane_detector else None

        with self.latest_detections_lock:
            self.latest_lane = lane_result

        with self.latest_frame_lock:
            self.latest_frame_id = frame_id

        with self.frame_count_lock:
            self.yolo_processed_count += 1

    # ── Draw latest detections on raw frame ────────────────────────────────
    def _draw_latest_detections(self, frame: np.ndarray, draw_lines: bool = True) -> np.ndarray:
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        with self.latest_detections_lock:
            dets = list(self.latest_detections)
            lane = self.latest_lane
            draw_zone = not self.config.get('no_zone', False)
            top_y    = self.zone_top_y if draw_zone and self.zone_top_y is not None else None
            bottom_y = self.zone_bottom_y if draw_zone and self.zone_bottom_y is not None else None

        if draw_lines:
            if top_y is not None:
                cv2.line(annotated, (0, top_y), (w, top_y), (0, 255, 255), 2)
            if bottom_y is not None:
                cv2.line(annotated, (0, bottom_y), (w, bottom_y), (0, 255, 255), 2)

        draw_lane(annotated, lane)

        for bbox, class_name, conf, color in dets:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name.upper()} {conf:.2f}"
            lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    # ── Draw detections once, return both video + display frames ───────────
    def _draw_latest_detections_both(self, frame: np.ndarray) -> tuple:
        """Single copy + single draw pass. Returns (video_frame, display_frame)."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        with self.latest_detections_lock:
            dets = list(self.latest_detections)
            lane = self.latest_lane
            draw_zone = not self.config.get('no_zone', False)
            top_y    = self.zone_top_y if draw_zone and self.zone_top_y is not None else None
            bottom_y = self.zone_bottom_y if draw_zone and self.zone_bottom_y is not None else None

        # Lane overlay (in-place; satu copy overlay di dalam draw_lane)
        draw_lane(annotated, lane)

        # Draw detection boxes (shared by video and display)
        for bbox, class_name, conf, color in dets:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name.upper()} {conf:.2f}"
            lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Video frame = annotated without zone lines
        video_frame = annotated

        # Display frame = video + zone lines (in-place, no extra copy)
        if top_y is not None:
            cv2.line(annotated, (0, top_y), (w, top_y), (0, 255, 255), 2)
        if bottom_y is not None:
            cv2.line(annotated, (0, bottom_y), (w, bottom_y), (0, 255, 255), 2)
        display_frame = annotated

        return video_frame, display_frame

    # ── Filter detections (vectorized) ─────────────────────────────────────
    def _filter_detections(self, detections: Detections, frame_shape) -> Detections:
        if len(detections) == 0:
            return detections

        valid_mask = np.isin(detections.class_id, list(self.detection_manager.valid_classes))
        if not np.any(valid_mask):
            return Detections.empty()
        detections = detections[valid_mask]

        if self.config.get('no_zone', False):
            return detections

        h = frame_shape[0]
        top_y = int(h * self.config['top_line'])
        bottom_y = int(h * self.config['bottom_line'])

        # Vectorized zone filter
        centers_y = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) * 0.5
        mask = (centers_y >= top_y) & (centers_y <= bottom_y)
        return detections[mask] if np.any(mask) else Detections.empty()

    # ── Process detections (stats + API) ───────────────────────────────────
    def _process_detections(self, detections: Detections, frame, frame_id, gps_data):
        if len(detections) == 0:
            return

        timestamp = time.strftime('%M:%S')

        if detections.tracker_id is None:
            return

        for i in range(len(detections)):
            tracker_id = int(detections.tracker_id[i])
            if tracker_id <= 0 or tracker_id in self.stats.processed_trackers:
                continue

            class_id   = int(detections.class_id[i])
            confidence = float(detections.confidence[i])
            bbox       = detections.xyxy[i]
            center_y   = int((bbox[1] + bbox[3]) / 2)

            # Kategori output dari mapping kelas model — hanya 'pothole'/'crack'
            # (model universal: D00/D10/D20 → crack, D40 → pothole, Repair → tidak lolos filter)
            dtype = self.detection_manager.detection_mappings.get(class_id, '')
            class_name = dtype
            if dtype == 'pothole':
                self.detection_manager.pothole_count += 1
                count = self.detection_manager.pothole_count
            elif dtype == 'rutting':
                self.detection_manager.rutting_count += 1
                count = self.detection_manager.rutting_count
            elif dtype == 'crack':
                self.detection_manager.crack_count += 1
                count = self.detection_manager.crack_count
            else:
                continue

            info = {
                'frame': frame_id, 'position': center_y,
                'confidence': confidence, 'detection_type': 'detected_in_zone',
                'timestamp': timestamp,
            }
            self.stats.increment(class_name, tracker_id, info)

            emoji = self.detection_manager.emoji_map.get(class_name, '🔍')
            print(f"{emoji} {class_name.upper()} #{count} ID:{tracker_id} Y:{center_y}")

            # Analyze API — padded crop from raw frame
            self.detection_manager._queue_analyze(frame, bbox, frame_id)

            # Vehicle-entries — crop small region (NOT full frame.copy())
            if _REQUESTS and self.config.get('vehicle_token') and self.detection_manager._api_sender:
                bx = bbox.astype(int)
                cx1 = max(0, int(bx[0])); cy1 = max(0, int(bx[1]))
                cx2 = min(frame.shape[1], int(bx[2])); cy2 = min(frame.shape[0], int(bx[3]))
                if cx2 > cx1 and cy2 > cy1:
                    crop = frame[cy1:cy2, cx1:cx2].copy()
                    self.detection_manager._api_sender.submit((
                        crop, dtype, class_name, confidence, info, gps_data,
                        np.array([cx1, cy1, cx2, cy2]), timestamp,
                    ))

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def start(self, cap_queue: queue.Queue, video_manager=None):
        self.running = True
        self.latest_frame = None
        self.latest_frame_id = -1
        self.video_manager = video_manager
        self.start_time = time.time()
        self.end_time = None

        threads = [
            threading.Thread(target=self.camera_thread, args=(cap_queue,), daemon=True, name="CameraThread"),
            threading.Thread(target=self.processing_thread, daemon=True, name="DetectionThread"),
            threading.Thread(target=self.annotated_video_thread, daemon=True, name="AnnotatedVideoThread"),
            threading.Thread(target=self.api_sender_thread, daemon=True, name="APISenderThread"),
        ]

        self.threads = threads
        for t in threads:
            t.start()

        # Create display window on MAIN thread (macOS requirement)
        if self.show_cam:
            cv2.namedWindow('Dashcam Live View - Detections', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Dashcam Live View - Detections', 800, 600)

        print("✅ LowLatencyDetector started")
        print(f"🎯 Target FPS: {self.target_fps}")

    def stop(self):
        self.running = False
        self.display_running = False
        self.end_time = time.time()
        try:
            self.api_queue.put_nowait(None)
        except queue.Full:
            pass
        if hasattr(self, 'threads'):
            for t in self.threads:
                if t.is_alive():
                    t.join(timeout=2.0)

    def get_latest_frame(self):
        with self.latest_frame_lock:
            return self.latest_frame, self.latest_frame_id

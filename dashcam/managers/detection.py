"""
Detection processing, annotation, and API communication

Performance notes (v2.0):
  - Pure NumPy Detections (no supervision) → zero per-frame conversion overhead
  - Single frame.copy() per frame (was 3x before)
  - JPEG encode ONLY when API payload is queued (was unconditional before)
  - pytz timezone cached at init (was recreated every frame)
  - Zone filter vectorized with NumPy (was Python loop)
  - API calls use a single sender thread + queue (was 1 thread per call)
"""
import cv2
import time
import queue
import threading
import numpy as np
from typing import Dict, Tuple, Optional, Any
from ultralytics import YOLO

try:
    import pytz
    from datetime import datetime
    _WIB = pytz.timezone("Asia/Jakarta")
except ImportError:
    _WIB = None

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

from dashcam.models.detection_stats import DetectionStats
from dashcam.models.detections import Detections
from dashcam.core.lane import lane_of_point


# ─────────────────────────────────────────────────────────────────────────────
# Drawing constants (same palette as yolo_tracker.py)
# ─────────────────────────────────────────────────────────────────────────────
_PALETTE = [
    (255, 56,  56),  (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210,  49), (72,  249, 10),  (146, 204, 23),  (61,  219, 134),
    (26,  147, 52),  (0,   212, 187), (44,  153, 168),  (0,   194, 255),
    (52,  69,  147), (100, 115, 255), (0,   24,  236),  (132, 56,  255),
    (82,  0,   133), (203, 56,  255), (255, 149, 200),  (255, 55,  199),
]
_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SC   = 0.55
_FONT_TH   = 1
_BOX_TH    = 2
_ALPHA     = 0.25

_COLORS = {
    "pothole": (0, 0, 255),
    "crack":   (0, 255, 0),
    "rutting": (255, 165, 0),
}

# API submit constants
_API_FIELD     = "image"
_API_SOURCE_ID = "camera_1"
_API_TIMEOUT   = 5.0
_API_JPEG_Q    = 90
_API_QUEUE_MAX = 64


# ─────────────────────────────────────────────────────────────────────────────
# ApiSenderThread — single background thread, non-blocking
# ─────────────────────────────────────────────────────────────────────────────
class _ApiSender(threading.Thread):
    """Single daemon thread that drains a queue and POSTs crops to the API.
    Replaces the old pattern of spawning 1 thread per detection.
    """
    def __init__(self, url: str, token: str):
        super().__init__(daemon=True, name="ApiSender")
        self.url   = url
        self.token = token
        self.q     = queue.Queue(maxsize=_API_QUEUE_MAX)
        self.stopped = False
        self.sent   = 0
        self.failed = 0

    def submit(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(item)
            except queue.Empty:
                pass

    def run(self):
        while not self.stopped:
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            self._do_send(item)

    def _do_send(self, item):
        crop, detection_type, class_name, confidence, info, gps_data, bbox, video_time = item
        try:
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), _API_JPEG_Q])
            if not ok:
                self.failed += 1
                return

            wib_now = datetime.now(_WIB).astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if _WIB else ""

            form = {
                "detection_type": detection_type,
                "class":          class_name,
                "confidence":     str(confidence),
                "timestamp":      wib_now,
                "sequence":       str(info["frame"]),
                "frame_index":    str(info["frame"]),
                "latitude":       str(gps_data.get("latitude", 0)),
                "longitude":      str(gps_data.get("longitude", 0)),
                "video_time":     video_time,
                "bbox_x1":        str(int(bbox[0])),
                "bbox_y1":        str(int(bbox[1])),
                "bbox_x2":        str(int(bbox[2])),
                "bbox_y2":        str(int(bbox[3])),
            }
            files  = {_API_FIELD: ("detection.jpg", buf.tobytes(), "image/jpeg")}
            headers = {"X-Vehicle-Token": self.token}

            r = requests.post(
                f"{self.url}/api/vehicle-entries",
                data=form, files=files, headers=headers, timeout=_API_TIMEOUT,
            )
            if 200 <= r.status_code < 300:
                self.sent += 1
            else:
                self.failed += 1
        except Exception:
            self.failed += 1

    def stop(self):
        self.stopped = True
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass


class _AnalyzeSender(threading.Thread):
    """Single daemon thread POSTing padded crops to /analyze."""

    def __init__(self, url: str, source: str, name: str):
        super().__init__(daemon=True, name="AnalyzeSender")
        self.url = url
        self.source = source
        self.name = name
        self.q = queue.Queue(maxsize=_API_QUEUE_MAX)
        self.stopped = False

    def submit(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(item)
            except queue.Empty:
                pass

    def run(self):
        while not self.stopped:
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            self._do_send(item)

    def _do_send(self, item):
        crop, bbox_str, frame_height, frame_id = item
        try:
            if crop is None or crop.size == 0:
                print("   ⚠️ Analyze: empty crop, skip")
                return
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), _API_JPEG_Q])
            if not ok:
                print("   ⚠️ Analyze: encode failed")
                return
            form = {
                "bbox": bbox_str,
                "frame_height": str(frame_height),
                "source": self.source,
                "frame_id": f"frame-{int(frame_id):04d}",
                "name": self.name,
            }
            files = {"image": ("crop.jpg", buf.tobytes(), "image/jpeg")}
            r = requests.post(self.url, data=form, files=files, timeout=_API_TIMEOUT)
            if 200 <= r.status_code < 300:
                print(f"   ✅ Analyze: {form['frame_id']} bbox={bbox_str}")
            else:
                print(f"   ⚠️ Analyze: Error {r.status_code}")
                for k, v in form.items():
                    print(f"      {k}: {v}")
        except Exception as e:
            print(f"   ⚠️ Analyze Error: {e}")

    def stop(self):
        self.stopped = True
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# SimpleSpatialTracker — Euclidean distance (same as yolo_tracker.py)
# ─────────────────────────────────────────────────────────────────────────────
class SimpleSpatialTracker:
    def __init__(self, max_distance: int = 150, max_lost: int = 10):
        self.max_distance = max_distance
        self.max_lost     = max_lost
        self.tracks       = {}
        self.next_id      = 1

    def update_with_detections(self, detections: Detections) -> Detections:
        if len(detections) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["lost_count"] += 1
                if self.tracks[tid]["lost_count"] > self.max_lost:
                    del self.tracks[tid]
            return detections

        n = len(detections)
        new_tracker_ids = np.full(n, -1, dtype=int)
        unmatched_dets  = list(range(n))

        # Vectorized center & area computation
        xyxy = detections.xyxy
        det_cx = ((xyxy[:, 0] + xyxy[:, 2]) / 2.0).astype(np.float32)
        det_cy = ((xyxy[:, 1] + xyxy[:, 3]) / 2.0).astype(np.float32)
        det_area = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])).astype(np.float32)

        matches = []
        for tid, track in self.tracks.items():
            tcx, tcy = track["center"]
            t_cls    = track["class_id"]
            t_area   = track["area"]

            for d_idx in unmatched_dets:
                if detections.class_id[d_idx] != t_cls:
                    continue
                area_ratio = det_area[d_idx] / (t_area + 1e-6)
                if area_ratio < 0.2 or area_ratio > 5.0:
                    continue
                dcy = det_cy[d_idx]
                if dcy < tcy - 10:
                    continue
                dcx = det_cx[d_idx]
                if abs(dcx - tcx) > 60:
                    continue
                dist = np.hypot(dcx - tcx, dcy - tcy)
                if dist < self.max_distance:
                    matches.append((dist, tid, d_idx))

        matches.sort(key=lambda x: x[0])

        matched_tids  = set()
        matched_didxs = set()

        for dist, tid, d_idx in matches:
            if tid not in matched_tids and d_idx not in matched_didxs:
                new_tracker_ids[d_idx]  = tid
                self.tracks[tid]["bbox"]       = detections.xyxy[d_idx]
                self.tracks[tid]["center"]     = (float(det_cx[d_idx]), float(det_cy[d_idx]))
                self.tracks[tid]["area"]       = float(det_area[d_idx])
                self.tracks[tid]["lost_count"] = 0
                matched_tids.add(tid)
                matched_didxs.add(d_idx)

        for tid in list(self.tracks.keys()):
            if tid not in matched_tids:
                self.tracks[tid]["lost_count"] += 1
                if self.tracks[tid]["lost_count"] > self.max_lost:
                    del self.tracks[tid]

        for d_idx in unmatched_dets:
            if d_idx not in matched_didxs:
                tid = self.next_id
                self.next_id += 1
                new_tracker_ids[d_idx] = tid
                self.tracks[tid] = {
                    "bbox":       detections.xyxy[d_idx],
                    "center":     (float(det_cx[d_idx]), float(det_cy[d_idx])),
                    "area":       float(det_area[d_idx]),
                    "lost_count": 0,
                    "class_id":   int(detections.class_id[d_idx]),
                }

        detections.tracker_id = new_tracker_ids
        return detections


# ─────────────────────────────────────────────────────────────────────────────
# DetectionManager
# ─────────────────────────────────────────────────────────────────────────────
class DetectionManager:

    def __init__(self, model: YOLO, config: Dict[str, Any]):
        self.model  = model
        self.config = config

        self.valid_classes     = set()
        self.detection_mappings = {}
        self.spatial_tracker   = SimpleSpatialTracker(max_distance=150, max_lost=10)

        self.emoji_map = {
            "pothole": "🕳️", "potholes": "🕳️",
            "alligator_cracking": "🐊", "alligator_crack": "🐊",
            "lateral_cracking": "↔️",  "lateral_crack": "↔️",
            "longitudinal_cracking": "↕️", "longitudinal_crack": "↕️",
            "transverse_crack": "↔️",  "transverse_cracking": "↔️",
            "rutting": "🛤️", "crack": "〰️",
        }

        self.pothole_count = 0
        self.alligator_cracking_count = 0
        self.lateral_cracking_count   = 0
        self.longitudinal_cracking_count = 0
        self.rutting_count = 0
        self.crack_count   = 0

        # Pre-compute zone boundaries (float) — computed once per frame in process_detections
        self._zone_top    = 0.0
        self._zone_bottom = 0.0

        # API sender thread (single thread, non-blocking)
        self._api_sender: Optional[_ApiSender] = None
        api_base = config.get("api_base", "https://airoad.roots.web.id")
        token    = config.get("vehicle_token", "")
        if _REQUESTS and token:
            self._api_sender = _ApiSender(api_base, token)
            self._api_sender.start()

        self._analyze_sender: Optional[_AnalyzeSender] = None
        if _REQUESTS and config.get("analyze"):
            self._analyze_sender = _AnalyzeSender(
                config.get("analyze_url", "http://192.168.18.22:8000/analyze"),
                config.get("analyze_source", "camera_0"),
                config.get("analyze_name", "dashcam/session"),
            )
            self._analyze_sender.start()

        self._initialize_classes()

    # ── class init ────────────────────────────────────────────────────────
    # Kode kelas model universal (RDDI) → kategori output.
    # Output dipaksa hanya 'pothole' dan 'crack':
    #   D00 (longitudinal) / D10 (transverse) / D20 (alligator) → crack
    #   D40 (pothole)                                           → pothole
    #   Repair (repaired area) & kelas lain tanpa mapping       → diabaikan
    _UNIVERSAL_CODES = {
        "d00": "crack",
        "d10": "crack",
        "d20": "crack",
        "d40": "pothole",
    }

    def _initialize_classes(self):
        for cid, cname in self.model.names.items():
            n = cname.lower()

            # Model universal: kode RDDI (D00/D10/D20/D40)
            dtype = next((d for code, d in self._UNIVERSAL_CODES.items() if code in n), None)
            if dtype:
                self.valid_classes.add(cid)
                self.detection_mappings[cid] = dtype
                continue

            # Fallback: model lama dengan nama deskriptif (pothole, *crack*, dll)
            if "pothole" in n or "potholes" in n:
                self.valid_classes.add(cid)
                self.detection_mappings[cid] = "pothole"
            elif any(k in n for k in ("crack", "alligator", "transverse", "longitudinal", "lateral", "rutting")):
                self.valid_classes.add(cid)
                self.detection_mappings[cid] = "crack"
        # "Repair" dan kelas tanpa mapping tidak masuk valid_classes
        # → otomatis dibuang oleh filter_detections / _filter_detections

    @staticmethod
    def _norm(name: str) -> str:
        return name.replace(" ", "_")

    @staticmethod
    def _crop_with_padding(frame: np.ndarray, bbox_xyxy, pad_ratio: float = 0.5):
        """Crop with 50% pad; return (crop, 'x,y,w,h') or (None, None)."""
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None, None
        bbox_str = f"{x1},{y1},{w},{h}"
        frame_h, frame_w = frame.shape[:2]
        pad_w, pad_h = int(w * pad_ratio), int(h * pad_ratio)
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(frame_w, x2 + pad_w)
        cy2 = min(frame_h, y2 + pad_h)
        if cx2 <= cx1 or cy2 <= cy1:
            return None, None
        return frame[cy1:cy2, cx1:cx2].copy(), bbox_str

    def _queue_analyze(self, frame: np.ndarray, bbox_xyxy, frame_id: int):
        if not self._analyze_sender:
            return
        crop, bbox_str = self._crop_with_padding(frame, bbox_xyxy)
        if crop is None:
            return
        self._analyze_sender.submit((crop, bbox_str, frame.shape[0], frame_id))

    # ── filter detections (vectorized) ────────────────────────────────────
    def filter_detections(self, det: Detections) -> Detections:
        if len(det) == 0:
            return Detections.empty()
        mask = np.isin(det.class_id, list(self.valid_classes))
        return det[mask] if np.any(mask) else Detections.empty()

    def _filter_by_zone(self, det: Detections, h: int) -> Detections:
        if len(det) == 0 or self.config.get("no_zone", False):
            return det
        top_y = self._zone_top
        bot_y = self._zone_bottom
        centers_y = (det.xyxy[:, 1] + det.xyxy[:, 3]) * 0.5
        mask = (centers_y >= top_y) & (centers_y <= bot_y)
        return det[mask] if np.any(mask) else Detections.empty()

    # ── main processing (video mode — single copy, conditional encode) ────
    def process_detections(self, detections: Detections, frame: np.ndarray,
                           frame_shape: Tuple[int, int, int],
                           frame_id: int, gps_data: Dict[str, Any],
                           stats: DetectionStats,
                           fps: float = 0, video_timestamp: str = "",
                           lane_result: dict = None
                           ) -> Tuple[np.ndarray, Optional[bytes]]:
        h, w = frame_shape[:2]

        # Pre-compute zone boundaries once
        self._zone_top    = h * self.config.get("top_line", 0.35)
        self._zone_bottom = h * self.config.get("bottom_line", 0.95)

        no_zone = self.config.get("no_zone", False)

        # Zone filter
        if len(detections) > 0 and not no_zone:
            detections = self._filter_by_zone(detections, h)

        # Tracker update
        if len(detections) > 0:
            detections = self.spatial_tracker.update_with_detections(detections)

        api_items = []

        # ── Single frame.copy() per frame (was 2x): build the translucent
        # overlay directly from frame, then blend → annotated. Borders, labels
        # and zone lines are drawn opaque on top afterward. This mirrors the
        # batched-overlay technique in tools/yolo_tracker.py.
        if len(detections) > 0:
            overlay = frame.copy()
            for i in range(len(detections)):
                bbox  = detections.xyxy[i].astype(int)
                cid   = int(detections.class_id[i])
                dtype = self.detection_mappings.get(cid, "")
                if not dtype:
                    continue
                color = _COLORS.get(dtype, (255, 255, 0))

                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # Blend creates `annotated` as a new array (single alloc, not a 2nd copy)
            annotated = cv2.addWeighted(overlay, _ALPHA, frame, 1 - _ALPHA, 0)
        else:
            # Tanpa deteksi: pakai frame langsung (tanpa copy ~2.7MB @720p).
            # Aman karena cap.read() di mode video selalu mengembalikan array
            # baru dan frame tidak dimutasi pihak lain setelah titik ini.
            annotated = frame

        # Zone lines (opaque, drawn on top of the blended frame)
        if not no_zone:
            top_y = int(self._zone_top)
            bot_y = int(self._zone_bottom)
            cv2.line(annotated, (0, top_y), (w, top_y), (0, 255, 255), 2)
            cv2.line(annotated, (0, bot_y), (w, bot_y), (0, 255, 255), 2)

        # Draw borders + labels on the blended frame
        if len(detections) > 0:
            for i in range(len(detections)):
                bbox  = detections.xyxy[i].astype(int)
                cid   = int(detections.class_id[i])
                dtype = self.detection_mappings.get(cid, "")
                if not dtype:
                    continue
                cname = dtype  # output unified: hanya 'crack' / 'pothole'
                conf  = float(detections.confidence[i])
                color = _COLORS.get(dtype, (255, 255, 0))
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # Border
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, _BOX_TH)

                # Label
                tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                label = f"#{tid} {cname} {conf:.2f}" if tid >= 0 else f"{cname} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SC, _FONT_TH)
                ly = max(y1, th + 6)
                cv2.rectangle(annotated, (x1, ly - th - 6), (x1 + tw + 6, ly + 2), color, -1)
                cv2.putText(annotated, label, (x1 + 3, ly - 2), _FONT, _FONT_SC,
                            (255, 255, 255), _FONT_TH, cv2.LINE_AA)

                # Jalur ke-n / total — baris kedua di bawah nama pothole/crack
                lane_no = lane_of_point(lane_result, (x1 + x2) * 0.5, y2)
                lane_txt = f"Lane {lane_no}/{lane_result['total_lanes']}" if lane_no else ""
                if lane_txt:
                    (lw, lh), _ = cv2.getTextSize(lane_txt, _FONT, _FONT_SC, _FONT_TH)
                    cv2.rectangle(annotated, (x1, ly + 2), (x1 + lw + 6, ly + lh + 10), color, -1)
                    cv2.putText(annotated, lane_txt, (x1 + 3, ly + lh + 6), _FONT, _FONT_SC,
                                (255, 255, 255), _FONT_TH, cv2.LINE_AA)

                # ── Count + API queue (skip already-processed trackers) ───
                tracker_id = tid
                if tracker_id > 0 and tracker_id not in stats.processed_trackers:
                    count = 0
                    if dtype == "pothole":
                        self.pothole_count += 1
                        count = self.pothole_count
                    elif dtype == "rutting":
                        self.rutting_count += 1
                        count = self.rutting_count
                    elif dtype == "crack":
                        self.crack_count += 1
                        count = self.crack_count

                    info = {
                        "frame": frame_id,
                        "position": int((bbox[1] + bbox[3]) / 2),
                        "confidence": conf,
                        "detection_type": "detected_in_zone",
                        "timestamp": time.strftime("%M:%S"),
                    }
                    if lane_no:
                        info["lane"] = lane_no
                        info["total_lanes"] = lane_result['total_lanes']
                    stats.increment(cname, tracker_id, info)
                    emoji = self.emoji_map.get(cname, "🔍")
                    print(f"{emoji} {cname.upper()} #{count} (ID:#{tracker_id}) "
                          f"at {info['timestamp']} (Y:{info['position']}, Conf:{conf:.2f})"
                          + (f" {lane_txt}" if lane_txt else ""))

                    # Crop for API — small region, not full frame
                    crop_x1 = max(0, x1); crop_y1 = max(0, y1)
                    crop_x2 = min(w, x2); crop_y2 = min(h, y2)
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1 and self._api_sender:
                        api_items.append((
                            frame[crop_y1:crop_y2, crop_x1:crop_x2].copy(),
                            dtype, cname, conf, info, gps_data,
                            np.array([crop_x1, crop_y1, crop_x2, crop_y2]),
                            info["timestamp"],
                        ))

                    self._queue_analyze(frame, bbox, frame_id)

        # Overlay text (always)
        self._draw_overlay(annotated, frame_id, fps, video_timestamp, lane_result)

        # ── JPEG encode ONLY when there are API items to send ─────────────
        encoded = None
        if api_items and self._api_sender:
            for item in api_items:
                self._api_sender.submit(item)
            # Encode once for the last crop (or skip entirely — crops are small)
            # Actually: crop is already small, encode inside ApiSender is cheap
            # So we skip encoding the full annotated frame here

        return annotated, encoded

    # ── overlay text (cached timezone) ────────────────────────────────────
    def _draw_overlay(self, frame: np.ndarray, frame_id: int,
                      fps: float, video_timestamp: str, lane_result: dict = None):
        total_cracks = self.crack_count + self.rutting_count
        y = 25

        if fps > 0:
            col = (0, 230, 80) if fps >= 15 else (0, 200, 255) if fps >= 5 else (0, 60, 255)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), _FONT, 0.6, col, 2)
            y += 25

        if video_timestamp:
            t_text = f"Time: {video_timestamp}"
        elif _WIB:
            t_text = f"Time: {datetime.now(_WIB).strftime('%H:%M:%S WIB')}"
        else:
            t_text = f"Time: {time.strftime('%H:%M:%S')}"
        cv2.putText(frame, t_text, (10, y), _FONT, 0.6, (255, 255, 255), 2)
        y += 25

        cv2.putText(frame, f"Potholes: {self.pothole_count}", (10, y), _FONT, 0.6, (0, 255, 255), 2)
        y += 25
        cv2.putText(frame, f"Cracks: {total_cracks}", (10, y), _FONT, 0.6, (255, 165, 0), 2)
        y += 25

        # Lane info (di atas Device:)
        if lane_result:
            conf = lane_result.get('confidence', 0)
            lcol = (0, 255, 0) if conf > 0.3 else (0, 0, 255)
            cv2.putText(frame, f"Lane: {lane_result['lane_number']}/{lane_result['total_lanes']} ({int(conf*100)}%)",
                        (10, y), _FONT, 0.6, lcol, 2)
            y += 25
            active = lane_result.get('active_lane', False)
            acol = (0, 255, 0) if active else (0, 0, 255)
            cv2.putText(frame, f"Active Lane: {'YES' if active else 'NO'}",
                        (10, y), _FONT, 0.6, acol, 2)
            y += 25

        dev = self.config.get("device", "auto").upper()
        dcol = (0, 255, 0) if "cuda" in dev.lower() else (0, 165, 0)
        cv2.putText(frame, f"Device: {dev}", (10, y), _FONT, 0.6, dcol, 2)
        y += 25

        if self.config.get("save_video"):
            cv2.putText(frame, "RECORDING", (10, y), _FONT, 0.6, (0, 0, 255), 2)

    # ── cleanup ───────────────────────────────────────────────────────────
    def stop(self):
        if self._api_sender:
            self._api_sender.stop()

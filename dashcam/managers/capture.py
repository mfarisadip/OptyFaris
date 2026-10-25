"""
Dataset capture: save images (jpg) + YOLO labels on detection.

Layout (under output root):
  images/  one .jpg  per frame containing >=1 valid detection (raw frame)
  labels/  one .txt  per captured frame, YOLO format:
             class_id cx cy w h   (cx,cy,w,h normalized to [0,1])
           class mapping: crack -> 0, pothole -> 1 ; rutting excluded.

Writes run on a background thread (non-blocking to YOLO) and are atomic
(temp file + os.replace) so a crash never leaves a partial/corrupt file.
"""
import os
import queue
import time
import threading
import cv2
import numpy as np
from typing import Dict, Optional, Any


# Keywords that classify a model class name into the 2-class scheme.
_CRACK_KEYS = ("crack", "alligator", "transverse", "longitudinal", "lateral")
_POTHOLE_KEYS = ("pothole",)
_RUTTING_KEYS = ("rutting",)


def _build_class_map(model) -> Dict[int, Optional[int]]:
    """Map model class_id -> 0 (crack), 1 (pothole) or None (skip)."""
    cmap: Dict[int, Optional[int]] = {}
    for cid, cname in model.names.items():
        n = str(cname).lower()
        if any(k in n for k in _RUTTING_KEYS):
            cmap[int(cid)] = None
            continue
        if any(k in n for k in _POTHOLE_KEYS):
            cmap[int(cid)] = 1
            continue
        if any(k in n for k in _CRACK_KEYS):
            cmap[int(cid)] = 0
            continue
        cmap[int(cid)] = None
    return cmap


class CaptureManager:
    """Background-threaded, atomic dataset capture for detections."""

    def __init__(self, store_dir: str, model, prefix: str = "frame", queue_size: int = 256):
        self.images_dir = os.path.join(store_dir, "images")
        self.labels_dir = os.path.join(store_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        self.class_map = _build_class_map(model)
        self.prefix = prefix
        self._captured = set()   # tracker_ids already captured this session (dedup)
        self._saved = 0

        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._running = True
        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="CaptureWriter"
        )
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────
    def maybe_save(self, frame: np.ndarray, detections, frame_id: int):
        """Enqueue a capture only when a NEW object (tracker_id) appears.

        Deduplicates by tracker_id so a persistent object is saved once, not every frame.
        The saved frame's label still lists ALL currently-visible saveable boxes.
        frame is copied here so the shared/published frame is never mutated.
        """
        if detections is None or len(detections) == 0:
            return
        tids = detections.tracker_id
        all_idxs = [i for i in range(len(detections))
                   if self.class_map.get(int(detections.class_id[i])) is not None]
        if not all_idxs:
            return
        new_idxs = [i for i in all_idxs
                    if tids is not None and int(tids[i]) > 0
                    and int(tids[i]) not in self._captured]
        if not new_idxs:
            return  # only already-seen objects visible → skip (removes duplicates)
        ts = time.strftime("%H%M%S_%f")[:-3]  # HHMMSSmmm (capture time)
        try:
            self._queue.put_nowait((frame.copy(), detections, all_idxs, frame_id, ts))
            for i in new_idxs:
                self._captured.add(int(tids[i]))  # mark AFTER successful enqueue
        except queue.Full:
            # Drop this capture rather than stall the YOLO pipeline; retried next frame.
            pass

    def stop(self):
        """Drain the queue, flush, and join the writer thread."""
        self._running = False
        try:
            self._queue.put(None, timeout=1)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        print(f"📸 Capture dataset: {self._saved} frame(s) saved "
              f"({self.images_dir})")

    # ── writer thread ─────────────────────────────────────────────────────
    def _writer_loop(self):
        while self._running or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            self._write_item(item)
            self._queue.task_done()

    def _write_item(self, item):
        frame, detections, idxs, frame_id, ts = item
        if frame is None or frame.size == 0:
            return

        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return

        # Select (class_id, normalized box) for each saveable detection.
        boxes = []
        for i in idxs:
            cid = int(detections.class_id[i])
            save_id = self.class_map.get(cid)
            if save_id is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in detections.xyxy[i]]
            cx = (x1 + x2) / (2.0 * w)
            cy = (y1 + y2) / (2.0 * h)
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            boxes.append(f"{save_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not boxes:
            return

        # Unique per frame: prefix + capture timestamp + monotonic frame_id.
        name = f"{self.prefix}_{ts}_{int(frame_id):06d}"

        # ── Image: encode once, atomic write ──
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return  # do not write a label without its image
        img_tmp = os.path.join(self.images_dir, f"{name}.jpg.tmp")
        img_final = os.path.join(self.images_dir, f"{name}.jpg")
        try:
            with open(img_tmp, "wb") as f:
                f.write(buf.tobytes())
                f.flush()
                os.fsync(f.fileno())
            os.replace(img_tmp, img_final)
        except OSError:
            if os.path.exists(img_tmp):
                os.remove(img_tmp)
            return

        # ── Label: atomic write ──
        lbl_tmp = os.path.join(self.labels_dir, f"{name}.txt.tmp")
        lbl_final = os.path.join(self.labels_dir, f"{name}.txt")
        try:
            with open(lbl_tmp, "w") as f:
                f.write("\n".join(boxes) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(lbl_tmp, lbl_final)
        except OSError:
            if os.path.exists(lbl_tmp):
                os.remove(lbl_tmp)
            return

        self._saved += 1

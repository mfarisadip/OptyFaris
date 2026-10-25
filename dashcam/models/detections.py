"""
Lightweight pure-NumPy Detections class — replaces sv.Detections.

Zero overhead vs supervision:
  - No import of supervision (pulls many dependencies)
  - from_ultralytics() reads result.boxes directly — no double conversion
  - __getitem__ (slicing) does not allocate heavy objects
  - No internal validation/marshalling per frame
"""
import numpy as np


class Detections:
    __slots__ = ("xyxy", "confidence", "class_id", "tracker_id")

    def __init__(self, xyxy, confidence, class_id, tracker_id=None):
        self.xyxy       = xyxy        # (N, 4) float32  — x1 y1 x2 y2
        self.confidence = confidence  # (N,)   float32
        self.class_id   = class_id    # (N,)   int32
        self.tracker_id = tracker_id  # (N,)   int64  | None

    def __len__(self):
        return len(self.xyxy)

    def __getitem__(self, idx):
        """Boolean mask or integer index → new Detections subset."""
        tid = self.tracker_id[idx] if self.tracker_id is not None else None
        return Detections(
            xyxy       = self.xyxy[idx],
            confidence = self.confidence[idx],
            class_id   = self.class_id[idx],
            tracker_id = tid,
        )

    @staticmethod
    def empty():
        return Detections(
            xyxy       = np.empty((0, 4), dtype=np.float32),
            confidence = np.empty((0,),   dtype=np.float32),
            class_id   = np.empty((0,),   dtype=np.int32),
        )

    @staticmethod
    def from_ultralytics(result):
        """Parse YOLO result.boxes directly — no supervision layer."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty()
        xyxy       = boxes.xyxy.cpu().numpy().astype(np.float32)
        confidence = boxes.conf.cpu().numpy().astype(np.float32)
        class_id   = boxes.cls.cpu().numpy().astype(np.int32)
        return Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

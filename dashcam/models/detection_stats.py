"""
Detection statistics tracking
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any


@dataclass
class DetectionStats:
    """Track detection statistics"""
    counts: Dict[str, int] = field(default_factory=dict)
    processed_trackers: Set[int] = field(default_factory=set)
    log: List[Dict[str, Any]] = field(default_factory=list)

    def increment(self, class_name: str, tracker_id: int, info: Dict[str, Any]):
        """Increment detection count"""
        if class_name not in self.counts:
            self.counts[class_name] = 0
        self.counts[class_name] += 1

        if tracker_id not in self.processed_trackers:
            self.processed_trackers.add(tracker_id)
            self.log.append({**info, 'type': class_name, 'tracker_id': tracker_id})

    def get_total_count(self):
        """Get total detection count"""
        return sum(self.counts.values())

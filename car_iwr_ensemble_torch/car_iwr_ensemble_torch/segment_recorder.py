import os
import pickle
import time
from typing import Any, List


class SegmentRecorder:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.segment_idx = 0
        self.buffer: List[Any] = []
        os.makedirs(self.save_dir, exist_ok=True)

    def add(self, payload: Any) -> None:
        self.buffer.append(payload)

    def save(self) -> None:
        if not self.buffer:
            return
        file_name = f"segment_{self.segment_idx:08d}.pkl"
        file_path = os.path.join(self.save_dir, file_name)
        payload = {
            "segment_idx": self.segment_idx,
            "timestamp": time.time(),
            "data": self.buffer,
        }
        with open(file_path, "wb") as f:
            pickle.dump(payload, f)
        self.segment_idx += 1
        self.buffer = []

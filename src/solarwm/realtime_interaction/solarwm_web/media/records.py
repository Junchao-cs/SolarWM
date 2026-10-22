from pathlib import Path
import threading

SOURCE_FPS = 16
PREVIEW_FORMATS = {"full": (1344, 768, "native"), "half": (672, 384, "half")}


class StreamRecord:
    def __init__(self, sequence: int, part: Path, final: Path):
        self.sequence, self.part, self.final = sequence, part, final
        self.done = threading.Event()

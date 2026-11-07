"""Read-only, incremental compiler progress. Never touches CUDA or model state."""
import json
import math
from pathlib import Path
import statistics
import threading
import time


class PreparationProgress:
    def __init__(self, root, deployment, ranks=8):
        self.root, self.deployment, self.ranks = Path(root), deployment, ranks
        self.offsets = [0] * ranks
        self.entered = [{} for _ in range(ranks)]
        self.finished = [{} for _ in range(ranks)]
        self.durations = [{} for _ in range(ranks)]
        self.latest = None
        self.lock = threading.Lock()
        self.cached, self.checked = None, -math.inf

    def _read(self, rank):
        path = self.root / f"compiler-rank{rank}" / f"rank-{rank}.jsonl"
        if not path.exists():
            return True
        if path.stat().st_size < self.offsets[rank]:
            raise ValueError("compiler evidence was truncated")
        with path.open("rb") as source:
            source.seek(self.offsets[rank])
            limit = self.offsets[rank] + 2 * 1024 * 1024
            while source.tell() < limit:
                start = source.tell()
                line = source.readline(4 * 1024 * 1024 + 1)
                if not line:
                    return True
                if len(line) > 4 * 1024 * 1024:
                    raise ValueError("compiler event exceeds reader limit")
                if not line.endswith(b"\n"):
                    return False  # Writer has not finished this record yet.
                event = json.loads(line)
                self.offsets[rank] = source.tell()
                if event.get("rank") != rank:
                    raise ValueError("compiler rank identity mismatch")
                if event.get("event") not in ("backend_enter", "inductor_return"):
                    continue
                phase = event.get("phase")
                if (not isinstance(phase, list) or len(phase) != 2
                        or type(phase[0]) is not int or not 0 <= phase[0] < 6
                        or type(phase[1]) is not bool):
                    raise ValueError("unknown compiler phase")
                key = phase[0] * 2 + int(phase[1])
                stamp = event["monotonic"]
                if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                    raise ValueError("invalid compiler event time")
                self.latest = max(self.latest or stamp, stamp)
                if event["event"] == "backend_enter":
                    if event.get("count") != 1 or key in self.entered[rank]:
                        raise ValueError("unexpected repeated compiler phase")
                    self.entered[rank][key] = stamp
                else:
                    seconds = event["seconds"]
                    if key not in self.entered[rank] or key in self.finished[rank]:
                        raise ValueError("unmatched compiler completion")
                    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                        raise ValueError("invalid compiler duration")
                    self.finished[rank][key] = stamp
                    self.durations[rank][key] = seconds
            return source.tell() >= path.stat().st_size

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            if self.cached is not None and now - self.checked < 2:
                return dict(self.cached)
            row = dict(deployment_id=self.deployment, observed_at=time.time(),
                       total=12, ranks=self.ranks, error=None, estimate_scope="dit_compilation_only")
            try:
                caught_up = all([self._read(rank) for rank in range(self.ranks)])
                completed = set.intersection(*(set(x) for x in self.finished))
                pending = set.union(*(set(x) for x in self.entered)) - completed
                current = min(pending) if pending else None
                starts = [x[current] for x in self.entered if current in x] if current is not None else []
                elapsed = max(0, now - min(starts)) if starts else 0
                durations = [max(x[key] for x in self.durations) for key in sorted(completed)]
                completion_times = [max(x[key] for x in self.finished) for key in sorted(completed)]
                intervals = [b-a for a, b in zip(completion_times, completion_times[1:]) if b > a]
                # Include observed tracing/transition gaps, not just backend time.
                typical = max(statistics.median(durations), statistics.median(intervals or durations)) if durations else None
                remaining = None
                if caught_up and len(completed) >= 2 and len(completed) < 12:
                    remaining = max(0, (12-len(completed))*typical-elapsed)
                overdue = bool(typical and elapsed > typical * 1.75)
                row.update(completed=len(completed), caught_up=caught_up,
                    completed_by_rank=[len(x) for x in self.finished],
                    current_phase=current + 1 if current is not None else None,
                    operation=("refresh" if current % 2 else "denoise") if current is not None else None,
                    phase_elapsed_s=elapsed, last_event_age_s=max(0, now-self.latest) if self.latest else None,
                    remaining_low_s=remaining*.75 if remaining is not None and not overdue else None,
                    remaining_high_s=remaining*1.35 if remaining is not None and not overdue else None,
                    overdue=overdue)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(error=str(error), caught_up=False)
            self.cached, self.checked = row, now
            return dict(row)

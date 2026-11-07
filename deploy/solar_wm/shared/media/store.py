"""Resumable complete-frame media store, no model imports."""
import collections, hashlib, json, re, struct, threading, time
from pathlib import Path
MAX_BYTES = 2 * 1024 * 1024

class TraceIndex:

    def __init__(self, path, capacity=8192):
        self.path = path
        self.position = 0
        self.rows = collections.deque(maxlen=capacity)
        self.serial = self.offset = self.frames = 0
        self.done = False
        self.lock = threading.Lock()

    def update(self):
        with self.path.open('rb') as source:
            source.seek(self.position)
            while True:
                line = source.readline(65537)
                if not line or not line.endswith(b'\n'):
                    if len(line) > 65536:
                        raise ValueError('oversized trace row')
                    break
                row = json.loads(line)
                if row.get('event') == 'fragment':
                    if row['serial'] != self.serial or row['offset'] != self.offset or row['first'] != self.frames or (not 0 <= row['count'] <= 128) or (not 0 < row['length'] <= 32 * 1024 * 1024):
                        raise ValueError('source fragment continuity differs')
                    self.rows.append(row)
                    self.serial += 1
                    self.offset += row['length']
                    self.frames += row['count']
                elif row.get('event') == 'finish':
                    if row.get('rc') != 0 or row.get('error') or row['output_frames'] != self.frames:
                        raise ValueError('source encoder did not finish cleanly')
                    self.done = True
                self.position = source.tell()

    def batch(self, cursor, limit):
        with self.lock:
            self.update()
            if cursor > self.serial:
                raise ValueError('cursor beyond produced fragments')
            if self.rows and cursor < self.rows[0]['serial']:
                rows = []
                with self.path.open('rb') as source:
                    for line in source:
                        if not line.endswith(b'\n'):
                            break
                        row = json.loads(line)
                        if row.get('event') == 'fragment' and row['serial'] >= cursor:
                            rows.append(row)
                            if len(rows) == limit:
                                break
            else:
                rows = [r for r in self.rows if r['serial'] >= cursor][:limit]
            if rows and rows[0]['serial'] != cursor:
                raise ValueError('requested source cursor missing')
            return (rows, self.serial, self.done)

class MediaStore:

    def __init__(self, root, deployment):
        self.root = root.resolve()
        self.deployment = deployment
        self.indices = collections.OrderedDict()
        self.lock = threading.Lock()

    def fragments(self, scene, sequence, cursor, limit):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', scene):
            raise ValueError('invalid scene')
        if not 1 <= sequence <= 10 ** 9 or not 0 <= cursor <= 10 ** 12 or (not 1 <= limit <= 64):
            raise ValueError('invalid sequence/cursor/batch')
        stem = f'{scene}-upgraded-streaming-session-{sequence}-live'
        trace = self.root / (stem + '.delivery.jsonl')
        if trace.is_symlink():
            raise ValueError('source trace symlink refused')
        with self.lock:
            index = self.indices.get(stem)
            if index is None:
                index = self.indices[stem] = TraceIndex(trace)
            self.indices.move_to_end(stem)
            while len(self.indices) > 2:
                self.indices.popitem(last=False)
        rows, produced, done = index.batch(cursor, limit)
        part = self.root / ('.' + stem + '.part.mp4')
        final = self.root / (stem + '.mp4')
        try:
            source = part.open('rb')
        except FileNotFoundError:
            source = final.open('rb')
        packets = []
        with source:
            if Path(source.name).is_symlink():
                raise ValueError('source media symlink refused')
            for row in rows:
                if packets and sum(map(len, packets)) + row['length'] > MAX_BYTES:
                    break
                source.seek(row['offset'])
                data = source.read(row['length'])
                if len(data) != row['length']:
                    raise ValueError('published fragment is incomplete')
                meta = dict(row, deployment_id=self.deployment, session=sequence, scene=scene, send_ms=time.monotonic_ns() / 1000000.0, sha256=hashlib.sha256(data).hexdigest())
                encoded = json.dumps(meta, separators=(',', ':')).encode()
                packets.append(struct.pack('>II', len(encoded), len(data)) + encoded + data)
        next_cursor = cursor + len(packets)
        return (b''.join(packets), next_cursor, bool(done and next_cursor == produced))

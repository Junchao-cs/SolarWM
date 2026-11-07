"""Camera-only controls; no changes to model, precision or captured shapes."""
import math
from collections import deque
from dataclasses import replace

KEY_ORDER = 'qweasdijkl'

def speed(value, *, maximum=10):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(f'speed must be a finite number from 0 to {maximum}')
    return float(value)

class NavigationCamera:
    """Preserve native planar camera exactly at 1x; add local-Y elevation."""
    def __init__(self, native):
        self.native = native
        self.base = native.config
        self.vertical = deque()
        self.vertical_velocity = 0.0

    @property
    def c2w(self):
        return self.native.c2w

    def prepare(self, actions, index, move=1., turn=1.):
        move, turn = speed(move, maximum=10), speed(turn, maximum=3)
        if self.vertical:
            raise RuntimeError('unconsumed camera steps')
        if len(actions) != (4 if index == 0 else 5):
            raise ValueError('camera action count differs')
        previous = 0 if index == 0 else index * 5 - 1
        for i, keys in enumerate(actions):
            if type(keys) is not list or any(type(k) is not str or k not in KEY_ORDER or len(k)!=1 for k in keys):
                raise ValueError('invalid navigation key')
            direction = int('q' in keys) - int('e' in keys)  # OpenCV local Y points down.
            self.vertical.extend([direction] * (1,4,4,4,4)[(previous+i)%5])
        self.native.config = replace(self.base,
            move_speed_per_second=self.base.move_speed_per_second*move,
            pitch_speed_degrees_per_second=self.base.pitch_speed_degrees_per_second*turn,
            yaw_speed_degrees_per_second=self.base.yaw_speed_degrees_per_second*turn)
        if move == 0:
            self.native.linear_velocity_local[:] = 0
            self.vertical_velocity = 0.
        if turn == 0:
            self.native.pitch_speed_degrees_per_second = 0.
            self.native.yaw_speed_degrees_per_second = 0.

    def step(self, keys):
        if not self.vertical:
            raise RuntimeError('camera sampling exceeded prepared packet')
        direction = self.vertical.popleft()
        result = self.native.step(keys)
        config = self.native.config
        target = direction * config.move_speed_per_second
        delta = (config.move_speed_per_second / config.frames_per_second /
                 config.acceleration_seconds) if config.acceleration_seconds else float('inf')
        difference = target - self.vertical_velocity
        self.vertical_velocity += max(-delta, min(delta, difference))
        if self.vertical_velocity:
            self.native.c2w[:3,3] += self.native.c2w[:3,1] * (self.vertical_velocity/config.frames_per_second)
            result = self.native.c2w.astype('float32', copy=True)
        return result

class NavigationStateMixin:
    def __init__(self, *args, **kwargs):
        self.move_speed = self.turn_speed = 1.
        super().__init__(*args, **kwargs)

    def snapshot(self):
        row = super().snapshot()
        with self.lock:
            row.update(navigation_controls=True, move_speed=self.move_speed,
                       turn_speed=self.turn_speed,
                       held_keys=''.join(k for k in KEY_ORDER if k in self.held))
        return row

    def set_speeds(self, move, turn):
        move, turn = speed(move), speed(turn)
        with self.lock:
            self.move_speed, self.turn_speed = move, turn
        self.publish('status', self.snapshot())

    def set_key(self, key, down):
        if type(key) is not str or len(key)!=1 or key not in KEY_ORDER or type(down) is not bool:
            raise ValueError('expected one navigation key and boolean')
        with self.lock:
            if not self.active or self.stop_requested:
                raise RuntimeError('no active navigation session')
            if down:self.held.add(key)
            else:self.held.discard(key)
        self.publish('status', self.snapshot())

    def all_idle(self):
        with self.lock:
            self.held.clear();self.transitions.clear()
        self.publish('status', self.snapshot())

    def next_actions(self, count):
        if type(count) is not int or count not in (4,5):raise ValueError('invalid action count')
        with self.lock:
            self.transitions.clear()
            return [[k for k in KEY_ORDER if k in self.held] for _ in range(count)]

    def set_pause(self, paused):
        if paused:self.all_idle()
        return super().set_pause(paused)

"""Request-scoped long-lived heap isolation, with automatic GC kept enabled.

Only freeze objects already alive at request entry. Never retain a frozen heap
across a scene switch or eviction; unfreeze in finally at request end. This is
CPU object-lifetime policy, not a change to model math or tensor ownership.
"""
import gc
import json
import os
import time

class RequestGC:

    def __init__(self, log, context, collector=gc):
        self.gc = collector
        self.log = log
        self.context = context
        self.active = False
        self.frozen = False
        self.started = None
        self.events = []
        self.callback = self.observe
        self.gc.callbacks.append(self.callback)

    def emit(self, row):
        row = dict(row, wall=time.time(), pid=os.getpid(), **self.context())
        self.events.append(row)
        with self.log.open('a') as f:
            f.write(json.dumps(row) + '\n')

    def observe(self, phase, info):
        if info.get('generation') != 2:
            return
        if phase == 'start':
            self.started = time.perf_counter()
        elif self.started is not None:
            elapsed = time.perf_counter() - self.started
            self.started = None
            self.emit(dict(event='generation2', seconds=elapsed, active=self.active, frozen=self.frozen, collected=info['collected'], uncollectable=info['uncollectable']))

    def begin(self, mode):
        if self.active:
            raise RuntimeError('nested GC owner')
        if mode not in ('observe', 'freeze'):
            raise ValueError('unknown GC mode')
        if not self.gc.isenabled():
            raise RuntimeError('automatic GC must stay enabled')
        if self.gc.get_freeze_count():
            raise RuntimeError('foreign frozen heap owner')
        self.active = True
        if mode == 'freeze':
            try:
                self.gc.collect(2)
                self.gc.freeze()
                self.frozen = True
            except BaseException:
                self.active = False
                raise
        self.emit(dict(event='begin', mode=mode, frozen_count=self.gc.get_freeze_count()))

    def end(self):
        if not self.active:
            return
        if self.frozen:
            self.gc.unfreeze()
        self.frozen = False
        self.active = False
        self.emit(dict(event='end', frozen_count=self.gc.get_freeze_count()))

    def close(self):
        self.end()
        if self.callback in self.gc.callbacks:
            self.gc.callbacks.remove(self.callback)

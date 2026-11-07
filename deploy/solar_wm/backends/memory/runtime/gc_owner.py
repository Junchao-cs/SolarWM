"""Explicit whole-process GC owner for this standalone Memory worker.

The installed interpreter itself starts with a nonempty permanent generation.
Thus nonzero freeze_count is not proof of a competing application owner. This
adapter requires explicit exclusive-process ownership and never disables GC.
"""
from deploy.solar_wm.backends.memory.runtime.gc_base import RequestGC as Parent

class RequestGC(Parent):

    def __init__(self, *args, exclusive_process=False, **kwargs):
        if exclusive_process is not True:
            raise RuntimeError('exclusive worker GC ownership required')
        super().__init__(*args, **kwargs)

    def begin(self, mode):
        if self.active:
            raise RuntimeError('nested GC owner')
        if mode not in ('observe', 'freeze'):
            raise ValueError('unknown GC mode')
        if not self.gc.isenabled():
            raise RuntimeError('automatic GC must stay enabled')
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

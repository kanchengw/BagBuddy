"""
multiprocess Python 3.12+ compatibility patch
"""
import multiprocess.resource_tracker as _rt
import os as _os

# Patch _stop_locked if it exists (some versions have it)
if hasattr(_rt.ResourceTracker, '_stop_locked'):
    _orig_sl = _rt.ResourceTracker._stop_locked
    def _patched_sl(self, close=_os.close, waitpid=_os.waitpid, ws=_os.waitstatus_to_exitcode):
        try:
            return _orig_sl(self, close, waitpid, ws)
        except AttributeError:
            pass
        if self._fd is None:
            return
        if self._pid is None:
            return
        close(self._fd)
        self._fd = None
        waitpid(self._pid, 0)
        self._pid = None
    _rt.ResourceTracker._stop_locked = _patched_sl

# Patch __del__ if it exists (some versions have it)
if hasattr(_rt.ResourceTracker, '__del__'):
    _orig_del = _rt.ResourceTracker.__del__
    def _patched_del(self):
        try:
            _orig_del(self)
        except Exception:
            pass
    _rt.ResourceTracker.__del__ = _patched_del

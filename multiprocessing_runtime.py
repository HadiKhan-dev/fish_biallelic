"""Shared process-pool runtime for the scientific pipeline.

The pipeline uses a forkserver so workers start from a lightweight,
single-threaded process rather than forking large or threaded parents. The
fallback to 'fork' preserves support for platforms without forkserver.
"""

import multiprocessing as _mp
import multiprocessing.pool as _mp_pool
from contextlib import contextmanager


try:
    forkserver_context = _mp.get_context("forkserver")
except (ValueError, AttributeError):
    forkserver_context = _mp.get_context("fork")


class ForkserverPool(_mp_pool.Pool):
    """A standard process pool bound to the shared forkserver context."""

    def __init__(self, *args, **kwargs):
        kwargs["context"] = forkserver_context
        super().__init__(*args, **kwargs)


class NonDaemonicProcess(forkserver_context.Process):
    """Forkserver process allowed to create child processes."""

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NonDaemonicForkserverContext(type(forkserver_context)):
    Process = NonDaemonicProcess


class NonDaemonicForkserverPool(_mp_pool.Pool):
    """Forkserver pool whose workers may create nested child pools."""

    def __init__(self, *args, **kwargs):
        kwargs["context"] = NonDaemonicForkserverContext()
        super().__init__(*args, **kwargs)


@contextmanager
def main_module_guard():
    """Prevent workers from re-executing the caller's entry script.

    The original __main__.__file__ and __main__.__spec__ values are restored
    on normal and exceptional exits. This intentionally preserves the existing
    convention that an absent or None __file__ remains absent after restoration.
    """

    import sys

    main_module = sys.modules.get("__main__")
    saved_main_file = getattr(main_module, "__file__", None)
    saved_main_spec = getattr(main_module, "__spec__", None)
    if main_module is not None:
        if hasattr(main_module, "__file__"):
            del main_module.__file__
        main_module.__spec__ = None
    try:
        yield
    finally:
        if main_module is not None:
            if saved_main_file is not None:
                main_module.__file__ = saved_main_file
            main_module.__spec__ = saved_main_spec


@contextmanager
def safe_forkserver_pool(processes, initializer=None, initargs=()):
    """Yield a ForkserverPool while guarding __main__."""

    with main_module_guard():
        with ForkserverPool(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
        ) as pool:
            yield pool

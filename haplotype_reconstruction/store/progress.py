"""Keep the partial progress of one long work unit, resumable inside the unit and removed when it publishes.

A unit (a selection block, a join batch, a refinement component) publishes one record when it completes; work in
flight when its process ends (a node expiry, a stopped run) used to be redone from the start. Its partial progress
is kept beside its container, under the unit's own key and identity, in one of two forms:

- a state snapshot (selection blocks, join batches): the unit's complete state at one of its own boundaries (a
  search round, a sweep), written when at least ``interval`` seconds of the unit's run have passed since its start
  or its last snapshot, atomically replacing the previous one (codec.write_payload: temporary file, fsync, rename).
  Nested science layers each contribute a frame: an enclosing layer registers a reader of its live state
  (:meth:`UnitProgress.frame`) and the innermost loop offers its own at each boundary (:meth:`boundary`); a
  resumed layer takes its frame back (:meth:`take`) and continues from it;
- keyed records (refinement components): every completed sweep and fit, appended to a per-unit container
  (:meth:`records`); the refinement resumes them by label.

The location is ``<container directory>/.<container name>.progress/<digest>.*``, the digest binding the unit's key
and progress identity: the unit's derivation identity (its code closure, parameters and inputs) and this module's
code. A unit with another identity never finds it; a snapshot that cannot be read is discarded with a warning (the
unit then starts afresh, with the same result). Only the process holding the unit's claim writes its progress,
and the unit removes it right after its own record is published (a stop between the two leaves files that no
later run reads: the published record wins). Snapshots are pickles of the science objects' exact state (the
store codec), trusted internal state read back only by the same code.

Progress is not a scientific record: resuming from it reproduces the uninterrupted unit's published records
(each science module documents what its frames hold and which execution counters count only what ran), and it
never enters an identity. The interval bounds the work an interruption loses and the snapshot traffic; units
shorter than it never write anything.
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from ..core.digest import json_digest
from . import codec
from .container import CheckpointFile, forget_index

__all__ = ["UnitProgress", "PROGRESS_SECONDS", "progress_directory"]

# Default minimum time between two snapshots of one unit (seconds). A snapshot of the largest measured units
# (giant joins and selection blocks; lab418/ckpt/RESULTS_CKPT.md) costs well under a second, so five minutes keeps
# their overhead far below 1 % while an interruption loses at most about five minutes of a unit's work. The run
# setting execution.progress_minutes overrides it (engine.runtime.RunSettings.progress_seconds).
PROGRESS_SECONDS = 300.0

# Bumped whenever the envelope or the frame protocol changes (enters the progress identity with this module's code).
PROGRESS_SCHEMA = 1


def progress_directory(container_path: str | os.PathLike) -> str:
    """The progress directory of one container: a dot-prefixed sibling that container scans ignore."""
    path = os.path.abspath(os.fspath(container_path))
    return os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.progress")


def _module_code() -> str:
    from .identity import code_dependency_identity

    return json_digest(code_dependency_identity(("haplotype_reconstruction.store.progress",)))


class UnitProgress:
    """The partial progress of one unit (see the module docstring). Created by the claim holder of the unit.

    ``container``: the path of the container the unit publishes to; ``key``: the unit's record key there;
    ``identity``: the unit's derivation identity. ``interval``: seconds between snapshots (0: at every boundary).
    Counters ``saved``, ``saved_bytes``, ``save_seconds`` and ``restored`` observe this process's snapshot work.
    """

    def __init__(
        self,
        container: str | os.PathLike,
        key: str,
        identity: str,
        *,
        interval: float = PROGRESS_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.container = os.path.abspath(os.fspath(container))
        self.key = str(key)
        self.identity = json_digest(dict(unit=str(identity), schema=PROGRESS_SCHEMA, code=_module_code()))
        stem = json_digest(dict(key=self.key, identity=self.identity))[:40]
        directory = progress_directory(self.container)
        self.state_path = os.path.join(directory, f"{stem}.state{codec.PAYLOAD_SUFFIX}")
        self.records_path = os.path.join(directory, f"{stem}.ckpt")
        self.interval = max(0.0, float(interval))
        self._clock = clock
        self._last = clock()
        self._frames: dict[str, Callable[[], Any]] = {}
        self._restored: dict[str, Any] | None = None
        self._loaded = False
        self._written = False
        self.saved = 0
        self.saved_bytes = 0
        self.save_seconds = 0.0
        self.restored = 0

    def __repr__(self) -> str:
        return f"UnitProgress({self.key!r})"

    # -- snapshots ---------------------------------------------------------------------------------

    def _load(self) -> None:
        self._loaded = True
        try:
            envelope = codec.read_payload(self.state_path)
        except FileNotFoundError:
            return
        except (OSError, ValueError, EOFError, pickle.UnpicklingError, AttributeError, ImportError) as error:
            warnings.warn(
                f"{self.state_path}: unreadable progress of {self.key} discarded ({error}); the unit restarts",
                RuntimeWarning,
                stacklevel=3,
            )
            self._remove(self.state_path)
            return
        if not isinstance(envelope, dict) or envelope.get("identity") != self.identity:
            self._remove(self.state_path)
            return
        self._restored = dict(envelope["frames"])
        self._written = True  # an earlier attempt's file: removed with this unit's progress
        self.restored += 1

    def take(self, name: str) -> Any | None:
        """The frame ``name`` of the saved snapshot, once (None when there is none): a resuming layer's state."""
        if not self._loaded:
            self._load()
        if self._restored is None:
            return None
        return self._restored.pop(name, None)

    def resuming(self) -> bool:
        """Whether a saved snapshot still holds frames no layer has taken."""
        if not self._loaded:
            self._load()
        return bool(self._restored)

    @contextmanager
    def frame(self, name: str, state: Callable[[], Any]) -> Iterator[None]:
        """Register an enclosing layer's live state reader for the snapshots taken inside this block."""
        if name in self._frames:
            raise ValueError(f"progress frame {name!r} is already open")
        self._frames[name] = state
        try:
            yield
        finally:
            self._frames.pop(name, None)

    def due(self) -> bool:
        return self._clock() - self._last >= self.interval

    def boundary(self, name: str, state: Callable[[], Any]) -> bool:
        """At a boundary of the innermost layer: when due, snapshot every open frame and ``name`` (``state()``)."""
        if not self.due():
            return False
        frames = {key: reader() for key, reader in self._frames.items()}
        frames[name] = state()
        self.save(frames)
        return True

    def save(self, frames: dict[str, Any]) -> None:
        """Replace the unit's snapshot by ``frames`` (atomic and durable) and restart the interval."""
        started = time.perf_counter()
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        size = codec.write_payload(
            self.state_path, dict(identity=self.identity, key=self.key, frames=frames), clevel=codec.FAST_CLEVEL
        )
        self._written = True
        self.saved += 1
        self.saved_bytes += int(size)
        self.save_seconds += time.perf_counter() - started
        self._last = self._clock()

    # -- keyed records -----------------------------------------------------------------------------

    def records(self, *, name: str = "") -> CheckpointFile:
        """The unit's own append-only container for keyed progress records (created by its first record; one found
        from an earlier attempt counts as restored)."""
        if not self._written and os.path.exists(self.records_path):
            self.restored += 1
        self._written = True
        return CheckpointFile(self.records_path, kind="contig", name=name, nthreads=1, clevel=codec.FAST_CLEVEL)

    # -- cleanup -----------------------------------------------------------------------------------

    def clear(self) -> None:
        """Remove the unit's progress once its record is published (nothing to do when none was written)."""
        if not self._written:
            return
        for path in (self.state_path, self.records_path):
            self._remove(path)
        # A snapshot write a stopped process left unfinished (codec.write_payload's temporary file).
        directory, stem = os.path.split(self.state_path)
        try:
            names = os.listdir(directory)
        except FileNotFoundError:
            names = []
        for name in names:
            if name.startswith(stem + ".") and name.endswith(".tmp"):
                self._remove(os.path.join(directory, name))
        forget_index(self.records_path)
        self._written = False

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

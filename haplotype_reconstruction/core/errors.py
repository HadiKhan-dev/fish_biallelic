"""Distinguish input, settings, resource and scientific invariant failures at task boundaries.

These categories define the CLI exit-code contract.
They carry ordinary exception messages and impose no logging or recovery policy.
"""

__all__ = ["SettingsError", "InputError", "ResourceError", "InvariantError"]


class SettingsError(ValueError):
    """Resolved settings cannot describe the requested run (CLI exit code 2)."""


class InputError(ValueError):
    """Observations or a supplied map violate their input contract (CLI exit code 3)."""


class ResourceError(RuntimeError):
    """The available resources cannot execute a task (CLI exit code 4)."""


class InvariantError(RuntimeError):
    """An internal scientific or execution invariant failed (CLI exit code 1)."""

"""Small process-memory release primitives shared by numerical stages."""

import ctypes


try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None


def malloc_trim():
    """Ask glibc to return free heap pages to the operating system."""
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


# Existing modules historically expose this private spelling.
_malloc_trim = malloc_trim

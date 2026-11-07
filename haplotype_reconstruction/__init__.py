"""Reconstruct founder haplotypes, pedigrees, phase and recombination with a lazy public API.

Importing the package loads no numerical libraries and changes no runtime policy.
The public functions are resolved from interface.api only when first requested.
"""

__all__ = ["run", "load_settings", "open_results", "errors", "__version__"]


def __getattr__(name: str):
    if name == "__version__":
        from ._version import __version__

        return __version__
    if name in {"run", "load_settings", "open_results", "errors"}:
        from .interface import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

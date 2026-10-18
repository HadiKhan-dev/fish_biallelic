"""Founder-haplotype reconstruction from low-coverage sequencing of experimental crosses.

One pipeline: evidence -> discovery -> local selection -> assembly -> painting -> pedigree -> founder refit ->
family phase -> recombination (see docs/). Importing the package configures the numerical libraries before any
numerical module loads (execution.bootstrap.configure_imports); nothing else happens at import.
"""
from .execution.bootstrap import configure_imports as _configure_imports

__version__ = "2.1.0"

_configure_imports()

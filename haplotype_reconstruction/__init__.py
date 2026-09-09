"""Founder-haplotype reconstruction from low-coverage experimental crosses."""
from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parent
__version__ = "1.0.0"
from .core.environment import force_single_threaded_numeric_libraries
force_single_threaded_numeric_libraries()

"""Simulation founder templates: built from a real cross's VCF (`haplotypes templates`: windowed founder discovery
and the naive linker, builder.py and linker.py) or synthetic for the end-to-end example (`haplotypes example`,
example.py). Templates generate simulated founders; the pipeline never reads them."""
from .builder import build_templates, templates_command
from .example import create_example, example_command

__all__ = ["build_templates", "create_example", "example_command", "templates_command"]

"""Derive the initialized configuration publication from the bound manifest fields."""

from ...core.digest import json_digest


def identity(scientific):
    """Hash the canonical scientific init document independently of invocation metadata."""
    return json_digest(scientific)


def expected(manifest, resolved):
    """Bind only the scientific configuration fields initialized by resolve."""
    return identity({key: manifest[key] for key in resolved.manifest()})

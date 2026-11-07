"""Bind stored results to their static code dependency closure and canonical inputs.

Canonical value digests are relocated unchanged to core.digest (P0 ENG-06). Source
walking retains the 3.5 path/NUL/bytes encoding. 4.0 code identities follow local
imports, including package initializers; callers seed the graph with stage adapters,
input construction and decoding modules as well as science. No module is imported
while computing a closure. Once a process has fixed its installed sources (the engine runtime
does when it starts), each installed closure is hashed once; before that, and for any other
root, closures are hashed on every call, so edits are seen.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ..core.digest import array_digest, canonical_json, json_digest, jsonable

__all__ = [
    "PACKAGE_ROOT",
    "CodeIdentity",
    "canonical_json",
    "jsonable",
    "json_digest",
    "array_digest",
    "add_sources",
    "fix_installed_sources",
    "dependency_closure",
    "code_dependency_identity",
    "code_identity",
    "source_digest",
]

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CodeIdentity:
    """The digest and inspectable module closure that produced a result."""

    digest: str
    modules: tuple[str, ...]


def add_sources(digest, directory: Path, root: Path = PACKAGE_ROOT) -> None:
    """Feed sorted source paths and bytes to a hasher using the original 3.5 encoding."""
    for current, subdirectories, files in os.walk(directory):
        subdirectories[:] = sorted(name for name in subdirectories if name != "__pycache__")
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(current, name)
                digest.update(str(os.path.relpath(path, root)).encode("utf-8", "surrogateescape") + b"\0")
                with open(path, "rb") as handle:
                    digest.update(handle.read())


def _module_path(module: str, root: Path) -> Path | None:
    parts = module.split(".")
    if parts[0] != root.name:
        return None
    stem = root.joinpath(*parts[1:])
    if stem.with_suffix(".py").is_file():
        return stem.with_suffix(".py")
    if (stem / "__init__.py").is_file():
        return stem / "__init__.py"
    return None


def _resolve_relative(name: str, package: str) -> str:
    """Resolve a literal relative import without importing modules or a loader."""
    level = len(name) - len(name.lstrip("."))
    if not level:
        return name
    parts = package.split(".")
    if level > len(parts):
        raise ImportError("relative import beyond top-level package")
    return ".".join(parts[: len(parts) - level + 1] + ([name[level:]] if name[level:] else []))


@functools.cache
def _import_records(text: str, restricted: bool, filename: str) -> tuple[tuple, ...]:
    """The import statements and literal import_module calls of one source text, in traversal order.

    A pure function of the text (parsing dominated every closure: about 1 s per stage root set, recomputed for
    every result reader). Keyed by content, so an edited source is parsed again. ``restricted`` skips function
    bodies (the root initializer's lazy exports).
    """
    tree = ast.parse(text, filename=filename)
    if restricted:
        nodes: list[ast.AST] = []
        todo: list[ast.AST] = [tree]
        while todo:
            node = todo.pop()
            nodes.append(node)
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                todo.extend(ast.iter_child_nodes(node))
    else:
        nodes = list(ast.walk(tree))
    records: list[tuple] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            records.append(("import", tuple(alias.name for alias in node.names)))
        elif isinstance(node, ast.ImportFrom):
            records.append(("from", node.module or "", node.level, tuple(alias.name for alias in node.names)))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            records.append(("call", node.args[0].value))
    return tuple(records)


def dependency_closure(modules: Iterable[str], *, package_root: Path = PACKAGE_ROOT) -> tuple[str, ...]:
    """Return the sorted local import closure, without executing package code.

    Roots may be fully qualified or relative to the package. Explicit package roots
    include their source modules (the 3.5 package-hashing behavior). Static imports in all
    branches count. Literal import_module calls count; dynamically selected modules
    must be supplied as roots by the stage. Unknown roots raise ValueError.

    The root initializer is always hashed. When reached only as an ancestor, its
    lazy public-export function bodies are not executed by importing a stage and
    are not followed. Otherwise every scientific key would include interface.run
    and every unrelated stage. The release label lives in the root
    initializer's lazy _version import, outside scientific closures. An explicit root import or root request follows
    those bodies normally. All non-root modules retain full static traversal.
    """
    root = Path(package_root)
    pending = {
        name if name.startswith(root.name + ".") or name == root.name else f"{root.name}.{name}" for name in modules
    }
    for name in pending:
        if _module_path(name, root) is None:
            raise ValueError(f"no source module {name}")
    # A package root denotes its code, not just its often-empty initializer.
    for name in tuple(pending):
        path = cast(Path, _module_path(name, root))
        if path.name == "__init__.py":
            for source in path.parent.rglob("*.py"):
                parts = list(source.relative_to(root).with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts.pop()
                pending.add(".".join([root.name, *parts]))
    root_exports = root.name in pending
    found = set()
    paths: dict[str, Path | None] = {}

    def path_of(name: str) -> Path | None:
        # The tree does not change during one closure; each name is resolved once per call.
        if name not in paths:
            paths[name] = _module_path(name, root)
        return paths[name]

    while pending:
        module = pending.pop()
        if module in found:
            continue
        candidate = path_of(module)
        if candidate is None:
            continue
        path = candidate
        found.add(module)
        parts = module.split(".")
        pending.update(".".join(parts[:i]) for i in range(1, len(parts)))
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        # Only root export bodies are exempt: ordinary lazy science imports
        # still belong to the calling stage's transitive source identity.
        restricted = module == root.name and not root_exports
        for kind, *data in _import_records(path.read_text(), restricted, str(path)):
            aliases: tuple[str, ...] = ()
            if kind == "import":
                names = list(data[0])
            elif kind == "from":
                stem, level, aliases = data
                if level:
                    stem = _resolve_relative("." * level + stem, package)
                names = [stem] + [f"{stem}.{alias}" for alias in aliases if alias != "*"]
            else:
                name = data[0]
                names = [_resolve_relative(name, package) if name.startswith(".") else name]
            root_attribute = root.name in names and not (
                kind == "from" and all(path_of(f"{root.name}.{alias}") is not None for alias in aliases)
            )
            if root_attribute and not root_exports:
                root_exports = True
                found.discard(root.name)
            pending.update(name for name in names if name not in found and path_of(name) is not None)
    return tuple(sorted(found))


# Whether this process's installed sources are fixed for its lifetime (fix_installed_sources).
_SOURCES_FIXED = False


def fix_installed_sources() -> None:
    """Declare the installed package's sources fixed for the rest of this process's life.

    The engine runtime calls this when it starts, and its forkserver and workers when they set up: from then on the
    process (and the workers it forks) run the code imported under one runtime until it exits, and editing that
    running snapshot is unsupported (runtime._check_declarations). Installed closures are then hashed once."""
    global _SOURCES_FIXED
    _SOURCES_FIXED = True


def code_dependency_identity(modules: Iterable[str], *, package_root: Path = PACKAGE_ROOT) -> CodeIdentity:
    """Hash a complete static dependency closure, retaining its module list for review.

    In a process with fixed installed sources (a run's coordinator, forkserver and workers), a closure of the
    installed package is hashed once: every derivation of every stage task reads its code identity, and parsing and
    hashing a closure's sources each time dominated planning steps and readers. Elsewhere (a reader or notebook with
    no runtime, whose sources may be edited between calls) and for any other package root, such as a fixture tree,
    the closure is hashed on every call."""
    root = Path(package_root)
    if root == PACKAGE_ROOT and _SOURCES_FIXED:
        return _installed_identity(frozenset(modules))
    return _hashed_closure(modules, root)


@functools.cache
def _installed_identity(modules: frozenset[str]) -> CodeIdentity:
    return _hashed_closure(modules, PACKAGE_ROOT)


def _hashed_closure(modules: Iterable[str], root: Path) -> CodeIdentity:
    closure = dependency_closure(modules, package_root=root)
    digest = hashlib.sha256()
    for module in closure:
        path = cast(Path, _module_path(module, root))
        digest.update(str(path.relative_to(root)).encode("utf-8", "surrogateescape") + b"\0")
        digest.update(path.read_bytes())
    return CodeIdentity(digest.hexdigest(), closure)


def code_identity(*modules: str, package_root: Path = PACKAGE_ROOT) -> str:
    """Return the dependency-closure digest for explicit stage/input/codec roots."""
    return code_dependency_identity(modules, package_root=package_root).digest


def source_digest() -> str:
    """Hash the whole package for run provenance using the original source walker."""
    digest = hashlib.sha256()
    add_sources(digest, PACKAGE_ROOT)
    return digest.hexdigest()

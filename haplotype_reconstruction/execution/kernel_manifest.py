"""Release warm-up signatures, generated from reference runs with this candidate on PYTHONPATH.

Run ``python -m haplotype_reconstruction.execution.kernel_manifest STORE/manifest.pkl ...``
when preparing a release. The JSON carries exact Numba argument types, not machine code. Its readable type
strings are documentation; the trusted pickled types preserve literals, defaults, layouts and named tuples.
"""
from __future__ import annotations

import argparse
import base64
import functools
import importlib
import json
from pathlib import Path
import pickle

RELEASE_MANIFEST = Path(__file__).with_name("kernel_manifest.json")


def read_release_manifest(path: Path = RELEASE_MANIFEST) -> tuple[dict, dict]:
    """Read packaged requests, skipping signatures unavailable in this dependency/code version."""
    try:
        data = json.loads(path.read_text())
        if data["format"] != 1:
            raise ValueError("unknown release manifest format")
    except Exception as error:
        print(f"  [kernel cache] release manifest unreadable ({type(error).__name__}); ignored", flush=True)
        return {}, {}
    overloads, seconds = {}, {}
    skipped = 0
    for entry in data["overloads"]:
        function = entry["module"], entry["function"]
        signatures = set()
        for signature in entry["signatures"]:
            try:
                signatures.add(pickle.loads(base64.b85decode(signature["pickle"])))
            except Exception:
                skipped += 1
        if signatures:
            overloads.setdefault(function, set()).update(signatures)
            seconds[function] = entry.get("seconds", 0.0)
    if skipped:
        print(f"  [kernel cache] {skipped} release signatures unavailable in this dependency/code version; "
              "skipped", flush=True)
    return overloads, seconds


def generate(manifests: list[Path], output: Path = RELEASE_MANIFEST) -> dict:
    """Union learned manifests and keep dispatchers importable in the current candidate.

    Inputs are trusted reference-run manifests, with no dependency on their original source path or CPU.
    Compile validity is checked by the existing warm-up task; a changed signature is an ordinary cache miss.
    """
    import numba
    from numba.core.serialize import dumps
    from .kernel_cache import _write_atomically

    overloads, seconds = {}, {}
    for path in manifests:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        for function, signatures in data["overloads"].items():
            overloads.setdefault(function, set()).update(signatures)
        for function, elapsed in data.get("seconds", {}).items():
            seconds[function] = max(seconds.get(function, 0.0), elapsed)
    entries, skipped = [], []
    for (module, function), signatures in sorted(overloads.items()):
        try:
            if not module.startswith("haplotype_reconstruction."):
                raise ImportError("not a package kernel")
            dispatcher = functools.reduce(getattr, function.split("."), importlib.import_module(module))
            if not callable(getattr(dispatcher, "compile", None)):
                raise TypeError("not a Numba dispatcher")
        except (ImportError, AttributeError, TypeError) as error:
            skipped.append(dict(module=module, function=function, reason=type(error).__name__))
            continue
        entries.append(dict(module=module, function=function, signatures=[
            dict(text=str(signature), pickle=base64.b85encode(dumps(signature)).decode("ascii"))
            for signature in sorted(signatures, key=str)], seconds=seconds.get((module, function), 0.0)))
    payload = dict(format=1, numba_version=numba.__version__, overloads=entries)
    _write_atomically(output, (json.dumps(payload, indent=2) + "\n").encode())
    return dict(functions=len(entries), overloads=sum(len(entry["signatures"]) for entry in entries),
                skipped=skipped, output=str(output))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+", help="trusted reference-run manifest.pkl files")
    parser.add_argument("--output", type=Path, default=RELEASE_MANIFEST,
                        help="generated package JSON (default: this candidate's execution/kernel_manifest.json)")
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.manifests, args.output), indent=2))


if __name__ == "__main__":
    main()

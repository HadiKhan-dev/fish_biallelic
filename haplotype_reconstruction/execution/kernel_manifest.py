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



def check(manifests: list[Path], release: Path = RELEASE_MANIFEST) -> dict:
    """Fail release validation when observed signatures are absent from the packaged requests.

    Accept learned cache manifests and the compile-event manifests written by start_audit. The latter
    include kernels that cannot persist to disk. This checks reference coverage, not future inputs.
    """
    # Unlike runtime warm-up, release validation must not silently ignore an unreadable list.
    data = json.loads(release.read_text())
    if data["format"] != 1:
        raise ValueError("unknown release manifest format")
    listed = {}
    for entry in data["overloads"]:
        key = entry["module"], entry["function"]
        listed.setdefault(key, set()).update(
            pickle.loads(base64.b85decode(s["pickle"])) for s in entry["signatures"])
    observed = {}
    for path in manifests:
        with path.open("rb") as handle:
            for key, signatures in pickle.load(handle)["overloads"].items():
                observed.setdefault(key, set()).update(signatures)
    missing = [dict(module=key[0], function=key[1], signature=str(sig))
               for key in sorted(observed)
               for sig in sorted(observed[key] - listed.get(key, set()), key=str)]
    return dict(observed=sum(map(len, observed.values())), missing=missing, release=str(release))


_AUDIT_STARTED = False


def start_audit() -> None:
    """Opt-in release receipt: save successful compiler events, including non-cacheable kernels.

    HAPLOTYPES_KERNEL_AUDIT names an attempt-local directory. Called in the driver and each scientific
    worker; one atomic manifest per process avoids competing writers. Normal runs install no listener.
    """
    import os
    global _AUDIT_STARTED
    directory = os.environ.get("HAPLOTYPES_KERNEL_AUDIT")
    if not directory or _AUDIT_STARTED:
        return
    import socket
    from numba.core import event
    from numba.core.serialize import dumps
    from .kernel_cache import _write_atomically
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    class CompileAudit(event.Listener):
        def __init__(self):
            self.overloads = {}

        def on_start(self, event):
            pass

        def on_end(self, event):
            dispatcher, args = event.data["dispatcher"], event.data["args"]
            function = dispatcher.py_func
            if not function.__module__.startswith("haplotype_reconstruction."):
                return
            # The dispatcher adds the overload before the successful end event. Failed typing and
            # literal-specialisation retries have no overload and are not actual compiled signatures.
            if args not in dispatcher.overloads:
                return
            key = function.__module__, function.__qualname__
            signatures = self.overloads.setdefault(key, set())
            if args in signatures:
                return
            signatures.add(args)
            _write_atomically(root / f"{socket.gethostname()}-{os.getpid()}.pkl",
                              dumps(dict(overloads=self.overloads, seconds={})))

    event.register("numba:compile", CompileAudit())
    _AUDIT_STARTED = True


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+", help="trusted reference-run manifest.pkl files")
    parser.add_argument("--output", type=Path, default=RELEASE_MANIFEST,
                        help="generated package JSON (default: this candidate's execution/kernel_manifest.json)")
    parser.add_argument("--check", action="store_true",
                        help="check observed manifests against --output; do not regenerate it")
    args = parser.parse_args(argv)
    result = check(args.manifests, args.output) if args.check else generate(args.manifests, args.output)
    print(json.dumps(result, indent=2))
    if args.check and (result["missing"] or not result["observed"]):
        raise SystemExit("Release kernel coverage failed: missing signatures or empty observations")


if __name__ == "__main__":
    main()

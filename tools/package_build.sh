#!/bin/bash
# Build from a private staged tree, then enforce IN5's contents and portability gates.
# Usage: HR_PACKAGE_PY=python tools/package_build.sh OUTPUT_DIRECTORY
# The selected interpreter must already have build, setuptools and wheel installed.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?Usage: tools/package_build.sh OUTPUT_DIRECTORY}
py=${HR_PACKAGE_PY:-python}
mkdir -p "$out"
out=$(cd "$out" && pwd)
exec 9>"$out/.package.lock"
flock -n 9 || exit 0
export PYTHONDONTWRITEBYTECODE=1
stage="$out/source"
"$py" - "$root" "$stage" <<'PY'
from pathlib import Path
import glob
import shutil
import sys
root, stage = map(Path, sys.argv[1:])
if stage.exists():
    raise SystemExit('build staging directory already exists; use a new output directory')
stage.mkdir()
for directory in ('haplotype_reconstruction','docs'):
    if (root/directory).is_dir():
        shutil.copytree(root/directory,stage/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbi','*.nbc'))
# MANIFEST's explicit includes are the distribution's portable fast-test selection.
for line in (root/'MANIFEST.in').read_text().splitlines():
    words=line.split()
    if words and words[0]=='include':
        for pattern in words[1:]:
            paths=glob.glob(str(root/pattern))
            if not paths:
                raise SystemExit('required distribution input missing: '+pattern)
            for item in paths:
                source=Path(item)
                destination=stage/source.relative_to(root)
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,destination)
PY
printf 'build_host=%s\nbuild_utc=%s\n' "$(hostname)" "$(date -u +%FT%TZ)" > "$out/build-location.txt"
"$py" -m build --no-isolation --outdir "$out/dist" "$stage" > "$out/build.log" 2>&1
"$py" "$stage/tests/packaging/test_import_scan.py" --root "$stage" --dist "$out/dist" --receipt "$out/packaging_receipt.json"
printf 'Wheel, sdist and distribution gates passed: %s\n' "$out/packaging_receipt.json"

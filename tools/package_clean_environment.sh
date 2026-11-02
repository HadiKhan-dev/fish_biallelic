#!/bin/bash
# IN5: build now; run the complete clean-install comparison only on a frozen candidate.
# Usage: HR_REFERENCE_PY=python3.14 HR_PACKAGE_PY=python tools/package_clean_environment.sh SCRATCH --build-only
# Full frozen-candidate gate: tools/package_clean_environment.sh SCRATCH --full
# Pip is used only inside fresh scratch venvs. Set normal pip index/proxy options externally.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?Usage: tools/package_clean_environment.sh SCRATCH [--build-only|--full]}
mode=${2:---build-only}
[[ "$mode" == --build-only || "$mode" == --full ]] || { echo 'choose --build-only or --full' >&2; exit 2; }
ref_py=${HR_REFERENCE_PY:-python3.14}
mkdir -p "$out"
out=$(cd "$out" && pwd)
exec 8>"$out/.clean-environment.lock"
flock -n 8 || exit 0
# Module entry probes must not find an editable source tree through the current directory.
cd "$out"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export NUMBA_DISABLE_INTEL_SVML=1 NUMBA_THREADING_LAYER=omp NUMBA_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONHOME
phase=build
receipt="$out/build/packaging_receipt.json"
finish() {
    local status=$?
    "$ref_py" - "$receipt" "$status" "$phase" "$mode" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
path=Path(sys.argv[1]);status=int(sys.argv[2]);phase,mode=sys.argv[3:]
value=json.loads(path.read_text()) if path.exists() else {}
value['clean_environment']=dict(status='PASS' if status==0 and mode=='--full' else 'not_run_build_only' if status==0 else 'FAIL',exit_code=status,phase=phase,host=platform.node(),utc=datetime.now(timezone.utc).isoformat())
path.parent.mkdir(parents=True,exist_ok=True)
temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
PY
    exit "$status"
}
trap finish EXIT
if [[ "$mode" == --full ]]; then
    [[ ! -e "$out/build-tools" ]] || { echo 'use a new scratch directory' >&2; exit 2; }
    "$ref_py" -m venv "$out/build-tools"
    build_py="$out/build-tools/bin/python"
    "$build_py" -m pip install --timeout 30 --retries 2 build 'setuptools>=84' wheel > "$out/build-tools-install.log" 2>&1
else
    build_py=${HR_PACKAGE_PY:-python}
fi
HR_PACKAGE_PY="$build_py" "$root/tools/package_build.sh" "$out/build"
[[ "$mode" == --full ]] || exit 0
phase=install
cat > "$out/installed_probe.py" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
from importlib import metadata

mode=sys.argv[1]
if mode=='versions':
    import haplotype_reconstruction as hr
    package=Path(hr.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix)), (package,sys.prefix)
    assert not {'numpy','numba','scipy'} & sys.modules.keys()
    pins={line.split('==')[0]:line.split('==')[1] for line in Path(sys.argv[2]).read_text().splitlines() if line and not line.startswith('#')}
    actual={name:metadata.version(name) for name in pins}
    assert actual==pins,(actual,pins)
    Path(sys.argv[3]).write_text(json.dumps(dict(python=platform.python_version(),versions=actual,package=str(package)),indent=2)+'\n')
elif mode=='records':
    import numpy as np
    import haplotype_reconstruction as hr
    from haplotype_reconstruction.core.digest import array_digest
    from haplotype_reconstruction.store.store import CheckpointStore
    root=Path(sys.argv[2])
    results=hr.open_results(root)
    assert len(results.contigs)==3 and results.manifest
    store=CheckpointStore(root/'checkpoints')
    manifest=store.manifest('run')
    files={'run':store.run,**{c:store.contig(c) for c in manifest['contigs']}}
    def neutral(value):
        if isinstance(value,np.ndarray):
            if value.dtype.hasobject:
                return ['array',value.dtype.str,list(value.shape),neutral(value.tolist())]
            return ['array',value.dtype.str,list(value.shape),array_digest(value)]
        if isinstance(value,np.generic): return neutral(value.item())
        if isinstance(value,dict): return ['mapping',sorted([[neutral(k),neutral(v)] for k,v in value.items()],key=lambda x:json.dumps(x[0]))]
        if isinstance(value,(list,tuple)): return [type(value).__name__,[neutral(v) for v in value]]
        if isinstance(value,bytes): return ['bytes',value.hex()]
        if isinstance(value,float): return ['float',value.hex()]
        if value is None or isinstance(value,(str,int,bool)): return value
        raise TypeError(type(value))
    import importlib.util
    spec=importlib.util.spec_from_file_location('publication_timing',os.environ['HR_PUBLICATION_TIMING'])
    timing=importlib.util.module_from_spec(spec);spec.loader.exec_module(timing)
    records={};leaves={}
    for scope,file in files.items():
        for key in file.keys():
            record=file.get(key)
            if isinstance(record,dict) and {'derivation','payload','publication_schema'} <= record.keys() and record.get('publication_kind')!='output':
                encoded=json.dumps(neutral(record),sort_keys=True,separators=(',',':')).encode()
                records[scope+'::'+key]=hashlib.sha256(encoded).hexdigest()
                leaves[scope+'::'+key]=dict(timing.leaf_digests(record,neutral))
    assert records,'no scientific publications compared'
    Path(sys.argv[3]).write_text(json.dumps(dict(contigs=list(results.contigs),scientific_records=records,scientific_leaves=leaves),indent=2)+'\n')
elif mode=='cache':
    base=Path(sys.argv[2]);base.mkdir(parents=True,exist_ok=True)
    blocked=base/'blocked';blocked.write_text('not a directory')
    from haplotype_reconstruction.engine.runtime import start
    start(2,cache_directory=blocked/'cache')
    import numpy as np
    import haplotype_reconstruction as hr
    from haplotype_reconstruction.engine.workers import create_shared_array,attach_shared_array,close_shared_memory
    values=np.arange(1<<22,dtype=np.uint8)
    owner,reference=create_shared_array(values,copy_threads=2)
    handles=[owner]
    try:
        attached,view=attach_shared_array(reference);handles.append(attached)
        np.testing.assert_array_equal(view,values)
    finally:
        close_shared_memory(handles,unlink=True)
    package=Path(hr.__file__).parent
    assert not list(package.rglob('*.nbi')) and not list(package.rglob('*.nbc'))
    assert blocked.is_file()
else:
    raise SystemExit('unknown probe mode')
PY
wheel=$(find "$out/build/dist" -maxdepth 1 -name '*.whl' -print)
sdist=$(find "$out/build/dist" -maxdepth 1 -name '*.tar.gz' -print)
for kind in wheel sdist; do
    case_root="$out/$kind"
    mkdir -p "$case_root"
    "$ref_py" -m venv "$case_root/venv"
    py="$case_root/venv/bin/python"
    artifact=$wheel
    [[ "$kind" == wheel ]] || artifact=$sdist
    env -u PYTHONPATH "$py" -m pip install --timeout 30 --retries 2 -c "$root/requirements-tested.txt" "$artifact" > "$case_root/install.log" 2>&1
    env -u PYTHONPATH "$py" -m pip check > "$case_root/pip-check.txt"
    env -u PYTHONPATH "$py" "$out/installed_probe.py" versions "$root/requirements-tested.txt" "$case_root/versions.json"
    env -u PYTHONPATH "$case_root/venv/bin/haplotypes" --version > "$case_root/cli-version.txt"
    env -u PYTHONPATH "$py" -m haplotype_reconstruction --version >> "$case_root/cli-version.txt"
done
phase=example
cd "$out"
env -u PYTHONPATH "$out/wheel/venv/bin/haplotypes" example "$out/example" > "$out/example.log" 2>&1
# One shared deterministic input location prevents path differences in source descriptors.
for kind in source wheel sdist; do
    phase="$kind-run"
    case_root="$out/$kind"
    mkdir -p "$case_root"
    if [[ "$kind" == source ]]; then
        py="$out/wheel/venv/bin/python"
        command=(env PYTHONPATH="$root" "$py" -m haplotype_reconstruction)
        probe=(env PYTHONPATH="$root" "$py")
    else
        py="$case_root/venv/bin/python"
        command=(env -u PYTHONPATH "$case_root/venv/bin/haplotypes")
        probe=(env -u PYTHONPATH "$py")
    fi
    export NUMBA_CACHE_DIR="$case_root/numba" XDG_CACHE_HOME="$case_root/cache"
    for step in init run; do
        timeout -k 10s "${HR_PACKAGE_COMMAND_SECONDS:-1800}s" "${command[@]}" "$step" "$out/example/simulation.toml" --output "$case_root/run" --scratch "$case_root/scratch" --cores 4 > "$case_root/$step.log" 2>&1
    done
    timeout -k 10s "${HR_PACKAGE_COMMAND_SECONDS:-1800}s" "${command[@]}" evaluate "$case_root/run" --cores 4 > "$case_root/evaluate.log" 2>&1
    env HR_PUBLICATION_TIMING="$root/tests/oracle/publication_timing.py" "${probe[@]}" "$out/installed_probe.py" records "$case_root/run" "$case_root/records.json"
    if [[ "$kind" != source ]]; then
        timeout -k 10s 120s "${probe[@]}" "$out/installed_probe.py" cache "$case_root/cache-refusal" > "$case_root/cache-refusal.log" 2>&1
    fi
done
phase=compare
"$ref_py" - "$out" "$root" <<'PY'
import json
import subprocess
from pathlib import Path
import sys
out=Path(sys.argv[1]);root=Path(sys.argv[2]);reference=json.loads((out/'source/records.json').read_text())
differences={}
for kind in ('wheel','sdist'):
    candidate=json.loads((out/kind/'records.json').read_text())
    old,new=reference['scientific_records'],candidate['scientific_records']
    differences[kind]=dict(missing=sorted(old.keys()-new.keys()),extra=sorted(new.keys()-old.keys()),changed=sorted(key for key in old.keys()&new.keys() if old[key]!=new[key]),contigs_equal=candidate['contigs']==reference['contigs'])
    # Receipt 26 (cmp40): strict on every leaf except the timings the reviewed field maps declare.
    run=subprocess.run([sys.executable,str(root/'tests/oracle/publication_timing.py'),'compare',str(out/'source/records.json'),str(out/kind/'records.json'),str(out/kind/'timing_projection.json')],capture_output=True,text=True)
    differences[kind]['projection']=dict(returncode=run.returncode,summary=run.stdout.strip(),stderr=run.stderr[-2000:],report=f'{kind}/timing_projection.json')
(out/'record_differences.json').write_text(json.dumps(differences,indent=2)+'\n')
assert all(row['projection']['returncode']==0 and not row['missing'] and not row['extra'] and row['contigs_equal'] for row in differences.values()), 'scientific publications differ beyond the declared timings; inspect record_differences.json and <route>/timing_projection.json'
receipt=json.loads((out/'build/packaging_receipt.json').read_text())
receipt['record_comparison']=dict(status='PASS',records=len(reference['scientific_records']),contigs=reference['contigs'],routes=['source','wheel','sdist'],excludes='operational logs, physical append tokens, output-only path records, and execution timings declared in the reviewed field maps (tests/oracle/publication_timing.toml; receipt 26)')
receipt['install_locations']={kind:json.loads((out/kind/'versions.json').read_text()) for kind in ('wheel','sdist')}
(out/'build/packaging_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
phase=complete
printf 'Clean wheel and sdist installs match the source run: %s\n' "$receipt"

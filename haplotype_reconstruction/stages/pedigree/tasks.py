"""Declare the pedigree graph and its independently runnable checkpoint cuts."""

from __future__ import annotations

from ...core.errors import InputError, SettingsError
from ...store.protocol import NumericalPolicy
from ...store.store import CheckpointStore
from ..context import StageRun
from ..framework import RecordKey, ResourceRequest, Task
from ..relatedness.working_map import working_maps
from . import map_scale, prepare
from .records import run_parameters

CONTIG_STEPS = ("prepare", "map-inputs", "map-frames")
RUN_STEPS = ("contamination", "refit-priors", "map-pairs", "gather")


def _prepare(root, manifest, maps, contig, parameters, policy, *, threads=1):
    store = CheckpointStore(root, nthreads=threads)
    maps = working_maps(store, manifest, maps)
    result = prepare.prepare_stored(root, manifest, maps.for_contig(contig), parameters, policy, threads=threads)
    return result.identity


def task(key, context: StageRun, *, gather_only=False) -> Task:
    """Declare the selected checkpoint publication or the full pedigree expansion."""
    resources = ResourceRequest(max_cores=context.cores)
    if key.unit == "run":
        return Task(key, prepare.pipeline_start,
                    args=(context.root, context.maps, context.manifest, context.cores),
                    kwargs=dict(gather_only=gather_only), resources=resources, thread_argument="threads")
    manifest = context.manifest
    if manifest is None:
        raise InputError("pedigree requires the bound run manifest")
    base = (context.root, manifest)
    parameters = run_parameters(manifest)
    policy = NumericalPolicy(**manifest["numerical_policy"])
    if key.unit == "prepare":
        action, args, product = _prepare, (*base, context.maps, key.scope, parameters, policy), prepare.KEY
    elif key.unit == "map-inputs":
        action, args, product = map_scale.scatter_inputs, (*base, key.scope), map_scale.INPUTS
    elif key.unit == "map-pairs":
        action, args, product = map_scale.scatter_pairs, (*base, parameters, policy), map_scale.PAIRS
    elif key.unit == "map-frames":
        action, args, product = map_scale.scatter_frames, (*base, key.scope, parameters, policy), map_scale.FRAMES
    else:
        raise SettingsError(f"unsupported pedigree task {key.unit}")
    return Task(key, action, args, resources=resources, thread_argument="threads",
                outputs=(RecordKey(key.scope, product),))

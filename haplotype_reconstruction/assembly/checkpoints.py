"""Assembly and block-feedback records in a contig's checkpoint file.

Every assembly of a contig (the final assembly under ``assembly``; the block
feedback context assemblies under ``feedback/l1/assembly`` and
``feedback/l2/assembly``) writes under its key prefix P of the contig's
CheckpointFile (core.checkpoints):

``identity/P``                       its exact scientific identity, bound once
``P/preprocess``                     founder completion result (prepared blocks)
``P/l<k>/plan``                      hierarchy level k at its start: {"batches":
                                     [[first input, end input, digest or None]]}
``joins/<digest>``                   one joined hierarchy batch, keyed by the
                                     SHA-256 of its exact inputs
                                     (hierarchy._batch_input_digest):
                                     {"status", "super_block"}; shared by every
                                     assembly of the contig
``P/l<k>/join``                      level k's manifest at its end: {"schema",
                                     "level", "batches", "components",
                                     "diagnostic"}
``P/l<k>/refine/<n:05d>``            founder refinement of level k's component n
                                     (paths on the prepared blocks,
                                     founder/checkpoints.py)
``P/l<k>/refine/<n:05d>/sweep/<fit>``  a long component's converged-sweep states
``P/l<k>/refine``                    level k's refinement: {"diagnostics",
                                     "components": count}
``P/final``                          the final count step: {"level",
                                     "diagnostics"}

A level's hierarchy output is its manifest's batches in order: a joined batch
is the super-block of its join record (a batch whose inputs an earlier
assembly joined reuses that join); a pass-through batch (a single input block,
or no informative sample) has no digest, costs nothing to recompute and is
stored inline as its blocks. A refinement level's output is its component
records; ``P/final`` copies no blocks either (they are the last level's).
Records hold no identity of their own: the section's bound identity covers
them, and join digests cover their inputs exactly.

Block feedback round R (``initial``, ``l1``, ``l2``) writes under
``feedback/R`` (discovery/path_blocks.py): ``identity/feedback/R``,
``feedback/R/blocks/<k:06d>`` (the round's selected blocks, 64 per record in
completion order: its product), ``feedback/R/proposals`` (rounds l1, l2: the
context refits) and ``feedback/R/selected`` (completion: chunk count and the
blocks kept as discovered, no block copies).
"""
from __future__ import annotations

import copy
import json
import re

from ..core import checkpoints as core_checkpoints
from ..core.haplotypes import BlockResults


FINAL_ASSEMBLY = "assembly"


FEEDBACK = "feedback"


FEEDBACK_ROUNDS = ("initial", "l1", "l2")


JOINS = "joins/"


LEVEL_SCHEMA = "assembly-level-v1"


FEEDBACK_STAGES = {f"feedback_{name}": name for name in FEEDBACK_ROUNDS}


def feedback_round(level):
    """Name of block feedback round ``level`` (0, 1, 2)."""
    return FEEDBACK_ROUNDS[level]


class AssemblyCheckpoints:
    """The records of one assembly (or feedback round) under ``prefix`` of a contig file.

    Cheap and picklable: founder refinement workers record their sweep states
    through scoped copies. ``bind`` binds the section to one exact identity
    (``identity/<prefix>``) before anything is written. Records are written at
    the fast compression level, with ``nthreads`` compression threads (the
    file's default when None).
    """

    def __init__(self, file, prefix, *, nthreads=None):
        if not prefix or prefix.startswith("/") or prefix.endswith("/"):
            raise ValueError(f"invalid assembly checkpoint prefix: {prefix!r}")
        self.file = file
        self.prefix = prefix
        self.nthreads = nthreads
        # Canonical JSON of the bound identity (batch queue stage names use it).
        self.identity = None

    def __repr__(self):
        return f"AssemblyCheckpoints({self.file.label!r}, {self.prefix!r})"

    def bind(self, identity):
        """Bind this section to ``identity``; a different bound identity is an error."""
        try:
            self.file.put_once(f"identity/{self.prefix}", identity)
        except core_checkpoints.CheckpointMismatch:
            raise RuntimeError(
                f"{self.file.label}: checkpoint identity '{self.prefix}' does not match the "
                "current model/configuration; use a new output root") from None
        self.identity = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False)

    def scoped(self, phase, *, nthreads=None):
        """The records under ``<prefix>/<phase>``, written under this binding."""
        scope = copy.copy(self)
        scope.prefix = self.key(phase)
        if nthreads is not None:
            scope.nthreads = nthreads
        return scope

    def key(self, phase):
        return f"{self.prefix}/{phase}"

    def has(self, phase):
        return self.file.has(self.key(phase))

    def load(self, phase):
        """The record of ``phase``, or None when it is absent."""
        key = self.key(phase)
        return self.file.get(key) if self.file.has(key) else None

    def save(self, phase, payload):
        self._require_bound()
        return self.file.put(self.key(phase), payload, nthreads=self.nthreads,
                             clevel=core_checkpoints.FAST_CLEVEL)

    def joins(self, digests):
        """{digest: join record} of those batch input digests some assembly joined."""
        present = set(self.file.keys(JOINS))
        return {digest: self.file.get(JOINS + digest) for digest in digests
                if JOINS + digest in present}

    def save_join(self, digest, result):
        """Record one joined batch (written by the coordinating process as it completes)."""
        self._require_bound()
        return self.file.put(JOINS + digest, result, nthreads=1,
                             clevel=core_checkpoints.FAST_CLEVEL)

    def _require_bound(self):
        if self.identity is None:
            raise RuntimeError(f"{self!r} must be bound before it records anything")


def level_blocks(checkpoints, manifest):
    """A hierarchy level's output blocks from its manifest, as the joins produced them."""
    if manifest.get("schema") != LEVEL_SCHEMA:
        raise ValueError(f"{checkpoints!r}: unrecognized hierarchy level manifest")
    blocks = []
    for batch in manifest["batches"]:
        if isinstance(batch, str):
            blocks.append(checkpoints.file.get(JOINS + batch)["super_block"])
        else:
            blocks.extend(batch)
    return BlockResults(blocks)


def _component_keys(file, prefix, level):
    pattern = re.compile(re.escape(f"{prefix}/l{level}/refine/") + r"\d+")
    return tuple(key for key in file.keys(f"{prefix}/l{level}/refine/")
                 if pattern.fullmatch(key))


def _final_level(file, prefix):
    """The last refined level of a completed assembly (its final count step)."""
    pattern = re.compile(re.escape(f"{prefix}/l") + r"(\d+)/refine")
    return max(int(match.group(1)) for key in file.keys(f"{prefix}/l")
               if (match := pattern.fullmatch(key)))


def _stage_source(file, stage):
    """(kind, prefix, level) of a founder stage's product, or None when it is absent."""
    if stage in FEEDBACK_STAGES:
        prefix = f"{FEEDBACK}/{FEEDBACK_STAGES[stage]}"
        return ("feedback", prefix, None) if file.has(f"{prefix}/selected") else None
    prefix = FINAL_ASSEMBLY
    if stage == "founder_refinement":
        if not file.has(f"{prefix}/final"):
            return None
        return "refine", prefix, _final_level(file, prefix)
    match = re.fullmatch(r"assembly_l(\d)", stage)
    if match is None:
        raise ValueError(f"unknown founder stage: {stage}")
    level = int(match.group(1))
    # The refined level when refinement ran, else the hierarchy level.
    if file.has(f"{prefix}/l{level}/refine"):
        return "refine", prefix, level
    if file.has(f"{prefix}/l{level}/join"):
        return "join", prefix, level
    return None


def stage_keys(file, stage):
    """The record keys stage_blocks decodes for ``stage``, or None when it is absent.

    Read from record keys (and a hierarchy level's small plan), without
    decoding products: for presence checks and memory planning
    (RecordInfo.decoded_bytes). ``block_discovery`` is covered too.
    """
    from ..discovery import checkpoints as discovery_checkpoints
    if stage == "block_discovery":
        return discovery_checkpoints.block_keys(file)
    source = _stage_source(file, stage)
    if source is None:
        return None
    kind, prefix, level = source
    if kind == "feedback":
        if not discovery_checkpoints.complete(file):
            return None
        return (*file.keys(f"{prefix}/blocks/"), f"{prefix}/selected",
                *file.keys(discovery_checkpoints.CHUNKS), discovery_checkpoints.PLAN)
    if kind == "join":
        plan = file.get(f"{prefix}/l{level}/plan")["batches"]
        return (f"{prefix}/l{level}/join",
                *(JOINS + digest for _start, _end, digest in plan if digest is not None))
    return (*_component_keys(file, prefix, level), f"{prefix}/l{level}/refine",
            f"{prefix}/preprocess")


def stage_blocks(file, stage):
    """The founder panels of a contig's ``stage`` as that stage produced them, or None.

    Stages: feedback_initial, feedback_l1, feedback_l2 (selected local panels),
    assembly_l1 .. assembly_l4 (the final assembly's refined level k, else its
    hierarchy level k) and founder_refinement (the final count step's panels).
    Discovery blocks come from discovery.checkpoints.load_blocks.
    """
    source = _stage_source(file, stage)
    if source is None:
        return None
    kind, prefix, level = source
    if kind == "feedback":
        from ..discovery import checkpoints as discovery_checkpoints, path_blocks
        originals = discovery_checkpoints.load_blocks(file, probs=False)
        blocks, _diagnostics, _memos = path_blocks.load_selection(
            AssemblyCheckpoints(file, prefix), originals)
        return tuple(blocks)
    from ..core.runtime import strip_block_evidence
    from .pipeline import _freeze_inference_snapshots
    checkpoints = AssemblyCheckpoints(file, prefix)
    if kind == "join":
        blocks = level_blocks(checkpoints, checkpoints.load(f"l{level}/join"))
    else:
        from .founder.checkpoints import RefinementCheckpoints
        prepared = checkpoints.load("preprocess").prepared_blocks
        records = RefinementCheckpoints(checkpoints.scoped(f"l{level}/refine"), prepared)
        blocks = records.blocks(checkpoints.load(f"l{level}/refine")["components"])
    # As the assembly keeps every level's output (assembly.pipeline).
    _freeze_inference_snapshots(blocks)
    strip_block_evidence(blocks)
    return tuple(blocks)

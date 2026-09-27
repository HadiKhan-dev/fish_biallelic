"""Checkpointed initial local fit → L1 feedback → L1+L2 feedback workflow.

Feedback rounds generate competing proposals; they do not add observations.
Selection follows each round, and its panels seed the next context assembly.
Final L1–L4 assembly is owned by reconstruction.py and consumes the final
selected local panels.
"""
from ..core.run_record import timed_stage
from dataclasses import asdict, dataclass, field, replace
import gc
import hashlib
import json

from haplotype_reconstruction import PACKAGE_ROOT
from haplotype_reconstruction.assembly import pipeline as assembly
from haplotype_reconstruction.assembly.checkpoints import (
    AssemblyCheckpointStore, AssemblyWorkCheckpoint, ASSEMBLY_RELEASE_CHECKPOINT_SCHEMA)
from haplotype_reconstruction.assembly.founder_refinement import FounderRefinementConfig
from haplotype_reconstruction.core import environment, parallel, runtime
from haplotype_reconstruction.core.config import PathSelectionConfig
from haplotype_reconstruction.discovery import feedback, search, cavity
from haplotype_reconstruction.discovery.candidate_selection import CandidateSelectionConfig


@dataclass(frozen=True)
class BlockFeedbackConfig:
    selection: str = field(default_factory=environment.block_feedback_selection)
    segment_exchange: bool = field(default_factory=environment.block_feedback_segment_exchange)
    path_search: PathSelectionConfig = field(default_factory=PathSelectionConfig)
    posterior_threshold: float = 0.99
    background_mass: float = 0.01
    maximum_context_founders: int = 10

    def __post_init__(self):
        if self.selection not in ("path", "balanced", "strict"):
            raise ValueError("feedback selection must be path, balanced or strict")
        if self.segment_exchange and self.selection != "path":
            raise ValueError("segment exchange requires path feedback selection")
        if (self.path_search.max_updates < 1 or self.path_search.rounds < 0
                or self.path_search.refits_per_kind < 1):
            raise ValueError("invalid local path search budget")
        if not .5 < self.posterior_threshold < 1:
            raise ValueError("feedback posterior_threshold must lie in (0.5, 1)")
        if not 0 < self.background_mass < 1:
            raise ValueError("feedback background_mass must lie in (0, 1)")
        if not 1 <= self.maximum_context_founders <= 10:
            raise ValueError("exact feedback refitting supports at most ten context founders")


def scientific_identity(config):
    files = sorted((PACKAGE_ROOT / "discovery").glob("*.py"))
    files += [PACKAGE_ROOT / name for name in (
        "workflows/block_feedback.py", "core/haplotypes.py", "core/genotypes.py",
        "core/config.py", "core/parallel.py", "core/genetic_map.py",
        "assembly/allele_polynomials.py")]
    return dict(schema="local-block-feedback-v3",
        selection_schedule=("initial_then_after_each_round" if config.selection == "path"
                            else "after_each_round"),
        config=asdict(config),
        local_search=asdict(CandidateSelectionConfig(criterion="bic")),
        code={str(path.relative_to(PACKAGE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files})



def _initial_path_identity(record):
    """Identity of the local fit, excluding later assembly search choices.

    Version this schema when initial-fit orchestration changes mathematically.
    Numerical/local-model dependencies remain content-tracked. Projecting the
    older encompassing identity permits reuse of unchanged initial fits and
    per-block checkpoints without relabelling downstream assembly as current.
    """
    if record.get("schema") == "initial-local-path-fit-v1":
        return record
    if (record.get("selection") != "path" or record.get("feedback_level") != 0
            or record.get("segment_exchange") is not False):
        raise ValueError("Only the initial path fit has assembly-independent identity")
    source = record["source"]
    hierarchy = source["config"]["scientific_hierarchy"]
    return dict(schema="initial-local-path-fit-v1",
        path_search=record["config"]["path_search"],
        local_search=record["local_search"],
        model=dict(generations=hierarchy["n_generations"],
                   recombination_rate_per_bp=hierarchy["recombination_rate"],
                   wildcard_mass=record["config"]["background_mass"],
                   genetic_map=source.get("genetic_map")),
        discovery_identity=source["discovery_identity"],
        ordered_sample_ids=source["ordered_sample_ids"],
        input_block_sha256=source["input_block_sha256"],
        input_array_sha256=source["input_array_sha256"],
        original_latent_sha256=record["original_latent_sha256"],
        code={name: digest for name, digest in record["code"].items()
              if name != "workflows/block_feedback.py"})


class _InitialPathCheckpointStore(AssemblyCheckpointStore):
    """Read unchanged initial fits across a downstream-only linker update."""

    def bind(self, identity):
        super().bind(_initial_path_identity(identity))

    def load(self, phase):
        artifact = self._artifact(phase)
        if not self.checkpoint_store.contig_done(self.work_stage, artifact):
            return None
        saved = self.checkpoint_store.load_contig(self.work_stage, artifact)
        if (not isinstance(saved, AssemblyWorkCheckpoint)
                or saved.schema != ASSEMBLY_RELEASE_CHECKPOINT_SCHEMA
                or saved.phase != phase):
            raise ValueError("Unrecognized initial-fit work checkpoint")
        projected = _initial_path_identity(json.loads(saved.identity_json))
        if projected != json.loads(self._identity_json):
            raise RuntimeError("Initial-fit scientific identity mismatch")
        return saved.payload


def _discovery_config(record):
    values = dict(record)
    values["cavity"] = cavity.HybridCavitySelectionConfig(**values["cavity"])
    if values.get("batched_search_config") is not None:
        values["batched_search_config"] = search.BatchedSearchConfig(**values["batched_search_config"])
    return search.ReversibleCavitySearchConfig(**values)


@timed_stage("block_feedback", "contig")
def run_block_feedback(store, contig, originals, gl, sites, observed, sample_ids,
                       *, discovery_identity, assembly_config, config, chromosome_map=None,
                       chromosome_evidence=None):
    """Return selected local blocks and their mode-specific scientific identity.

    Raw block discovery files are never replaced. Path selection first refits
    the original panels. The cavity-rescue alternatives share their raw L1
    context. Each round's selection, and the second context and raw proposals,
    are mode-specific because selected L1 panels feed that context. Only the
    outer reconstruction runner publishes genome-wide completion.
    """
    cpus = min(assembly_config.num_processes, runtime.available_cpu_count())
    # This option affects only final chromosomes. Do not invalidate or refit
    # identical local contexts merely because final refinement was toggled.
    assembly_config = replace(assembly_config,
        founder_refinement_config=FounderRefinementConfig(enabled=False))
    identity = scientific_identity(config)
    mode = identity["config"].pop("selection")
    original_identity = assembly.assembly_release_identity_record(assembly_config,
        discovery_identity=discovery_identity, sample_ids=sample_ids, input_blocks=originals,
        global_probs=gl, global_sites=sites, global_observed_mask=observed,
        chromosome_map=chromosome_map, chromosome_evidence=chromosome_evidence)
    latent = hashlib.sha256()
    for block in originals:
        fitted = getattr(block, "cavity_selected_mode", None)
        if fitted is not None:
            latent.update(assembly._array_digest(fitted.haplotypes).encode())
    identity.update(source=original_identity, original_latent_sha256=latent.hexdigest())
    if mode == "path":
        # Segment exchange is final-only: toggling it reuses unchanged core rounds.
        identity["config"].pop("segment_exchange")
        return _run_path_feedback(store, contig, originals, gl, sites, observed, sample_ids,
            discovery_identity=discovery_identity, assembly_config=assembly_config,
            config=config, chromosome_map=chromosome_map, chromosome_evidence=chromosome_evidence,
            identity=identity, cpus=cpus)
    selected_identity = dict(identity, selection=mode, feedback_level=2)
    final_io = AssemblyCheckpointStore(store, work_stage=f"feedback_{mode}_l2", contig=contig)
    final_io.bind(selected_identity)
    saved = final_io.load("selected")
    if saved is not None:
        print(f"[Feedback] {contig}: resumed {mode} selected local panels", flush=True)
        return saved["blocks"], selected_identity

    options = dict(threshold=config.posterior_threshold,
                   background_mass=config.background_mass,
                   rate=assembly_config.recombination_rate * assembly_config.n_generations,
                   max_k=config.maximum_context_founders)
    selection_config = CandidateSelectionConfig(
        discovery=_discovery_config(discovery_identity["config"]), criterion="bic")
    selected_rounds = []
    current = originals
    for level in (1, 2):
        selection_stage = f"feedback_{mode}_l{level}"
        selection_identity = dict(identity, selection=mode, feedback_level=level)
        selection_io = AssemblyCheckpointStore(store, work_stage=selection_stage, contig=contig)
        selection_io.bind(selection_identity)
        saved = selection_io.load("selected")
        if saved is not None:
            current = saved["blocks"]
            selected_rounds.append(current)
            print(f"[Feedback] {contig}: resumed L{level} {mode} selected panels", flush=True)
            continue

        stage = "feedback_l1" if level == 1 else selection_stage
        pass_identity = dict(identity, feedback_level=level)
        if level == 2:
            pass_identity["selection"] = mode
        pass_io = AssemblyCheckpointStore(store, work_stage=stage, contig=contig)
        pass_io.bind(pass_identity)
        saved = pass_io.load("blocks")
        if saved is not None:
            proposed = saved["blocks"]
            print(f"[Feedback] {contig}: resumed L{level} proposals", flush=True)
        else:
            print(f"[Feedback] {contig}: assembling context through L{level}", flush=True)
            assembly_io = AssemblyCheckpointStore(store, work_stage=stage + "_assembly", contig=contig)
            with parallel.numba_thread_scope(cpus):
                context = assembly.assemble_chromosome(current, gl, sites, observed, sample_ids,
                    discovery_identity=dict(discovery_identity, feedback_pass=pass_identity),
                    config=replace(assembly_config, max_level=level),
                    release_checkpoints=assembly_io, chromosome_map=chromosome_map,
                    chromosome_evidence=chromosome_evidence)
                proposed, diagnostic = feedback.refine(current, context["components"],
                    gl, sites, observed, cpus, options, chromosome_map=chromosome_map,
                    n_generations=assembly_config.n_generations)
            # Unchanged fallbacks can still carry read arrays; stripping a
            # shallow copy avoids modifying the immutable original inputs.
            import copy
            proposed = feedback.BlockResults([copy.copy(block) for block in proposed])
            runtime.strip_block_evidence(proposed)
            pass_io.save("blocks", dict(blocks=proposed, diagnostic=diagnostic))
            del context
            gc.collect()
            parallel.malloc_trim()
        # Original latent starts stay available at both rounds. Round two
        # competes against selected L1, not the discarded raw L1 proposals.
        with parallel.numba_thread_scope(cpus):
            current, diagnostic = feedback.select_blocks(originals,
                [*selected_rounds, proposed], gl, sites, observed, cpus,
                selection_config, mode, selection_io)
        selection_io.save("selected", dict(blocks=current, diagnostic=diagnostic))
        selected_rounds.append(current)
    return current, selected_identity


def _run_path_feedback(store, contig, originals, gl, sites, observed, sample_ids, *,
                       discovery_identity, assembly_config, config, chromosome_map,
                       chromosome_evidence, identity, cpus):
    """Keep original latent starts, select initially and after each context round."""
    import copy
    from haplotype_reconstruction.discovery import path_blocks

    # Small per-block writes stay single-threaded while numerical workers run.
    local_store = runtime.CheckpointStore(store.root, nthreads=1)
    final_stage = "feedback_path_exchange" if config.segment_exchange else "feedback_path_l2"
    selected_identity = dict(identity, selection="path", feedback_level=2,
                             segment_exchange=config.segment_exchange)
    final_io = AssemblyCheckpointStore(local_store, work_stage=final_stage, contig=contig)
    final_io.bind(selected_identity)
    saved = final_io.load("selected")
    if saved is not None:
        print(f"[Feedback] {contig}: resumed path-selected local panels", flush=True)
        return saved["blocks"], selected_identity

    selection_config = CandidateSelectionConfig(
        discovery=_discovery_config(discovery_identity["config"]), criterion="bic")
    options = dict(threshold=config.posterior_threshold, background_mass=config.background_mass,
                   rate=assembly_config.recombination_rate * assembly_config.n_generations,
                   max_k=config.maximum_context_founders)
    model_options = dict(generations=assembly_config.n_generations,
        recombination_rate_per_bp=assembly_config.recombination_rate,
        wildcard_mass=config.background_mass, chromosome_map=chromosome_map)
    selected_rounds, current = [], originals
    for level in (0, 1, 2):
        stage = "feedback_path_initial" if level == 0 else f"feedback_path_l{level}"
        pass_identity = dict(identity, selection="path", feedback_level=level,
                             segment_exchange=False)
        checkpoint_type = _InitialPathCheckpointStore if level == 0 else AssemblyCheckpointStore
        selection_io = checkpoint_type(local_store, work_stage=stage, contig=contig)
        selection_io.bind(pass_identity)
        saved = selection_io.load("selected")
        if saved is not None:
            current = saved["blocks"]
            selected_rounds.append(current)
            print(f"[Feedback] {contig}: resumed path selection round {level}", flush=True)
            continue
        proposed = None
        if level:
            saved_proposals = selection_io.load("proposals")
            if saved_proposals is None:
                print(f"[Feedback] {contig}: assembling selected context through L{level}", flush=True)
                assembly_io = AssemblyCheckpointStore(store,
                    work_stage=stage + "_assembly", contig=contig)
                with parallel.numba_thread_scope(cpus):
                    context = assembly.assemble_chromosome(current, gl, sites, observed, sample_ids,
                        discovery_identity=dict(discovery_identity, feedback_pass=pass_identity),
                        config=replace(assembly_config, max_level=level),
                        release_checkpoints=assembly_io, chromosome_map=chromosome_map,
                        chromosome_evidence=chromosome_evidence)
                    proposed, diagnostic = feedback.refine(current, context["components"],
                        gl, sites, observed, cpus, options, chromosome_map=chromosome_map,
                        n_generations=assembly_config.n_generations)
                proposed = feedback.BlockResults([copy.copy(block) for block in proposed])
                runtime.strip_block_evidence(proposed)
                saved_proposals = dict(blocks=proposed, diagnostic=diagnostic)
                selection_io.save("proposals", saved_proposals)
                del context
                gc.collect()
                parallel.malloc_trim()
            proposed = saved_proposals["blocks"]
        with parallel.numba_thread_scope(cpus):
            current, diagnostic = path_blocks.select_blocks(originals,
                [*selected_rounds, proposed] if level else [], gl, sites, observed, cpus,
                selection_config, config.path_search, selection_io, **model_options)
        selection_io.save("selected", dict(blocks=current, diagnostic=diagnostic))
        selected_rounds.append(current)
    if config.segment_exchange:
        with parallel.numba_thread_scope(cpus):
            current, diagnostic = path_blocks.select_blocks(current, [], gl, sites, observed,
                cpus, selection_config, config.path_search, final_io,
                segment_exchange=True, **model_options)
        final_io.save("selected", dict(blocks=current, diagnostic=diagnostic))
    return current, selected_identity

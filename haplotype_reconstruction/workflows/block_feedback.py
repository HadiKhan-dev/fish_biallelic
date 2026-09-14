"""Checkpointed L1 → local selection → L1+L2 → local selection workflow.

Feedback rounds generate competing proposals; they do not add observations.
Selection follows each round, and its panels seed the next context assembly.
Final L1–L4 assembly is owned by reconstruction.py and consumes the final
selected local panels.
"""
from dataclasses import asdict, dataclass, field, replace
import gc
import hashlib

from haplotype_reconstruction import PACKAGE_ROOT
from haplotype_reconstruction.assembly import pipeline as assembly
from haplotype_reconstruction.assembly.checkpoints import AssemblyCheckpointStore
from haplotype_reconstruction.assembly.founder_refinement import FounderRefinementConfig
from haplotype_reconstruction.core import environment, parallel, runtime
from haplotype_reconstruction.discovery import feedback, search, cavity
from haplotype_reconstruction.discovery.candidate_selection import CandidateSelectionConfig


@dataclass(frozen=True)
class BlockFeedbackConfig:
    selection: str = field(default_factory=environment.block_feedback_selection)
    posterior_threshold: float = 0.99
    background_mass: float = 0.01
    maximum_context_founders: int = 10

    def __post_init__(self):
        if self.selection not in ("balanced", "strict"):
            raise ValueError("feedback selection must be balanced or strict")
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
    return dict(schema="local-block-feedback-v2", selection_schedule="after_each_round",
        config=asdict(config),
        local_search=asdict(CandidateSelectionConfig(criterion="bic")),
        code={str(path.relative_to(PACKAGE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files})


def _discovery_config(record):
    values = dict(record)
    values["cavity"] = cavity.HybridCavitySelectionConfig(**values["cavity"])
    if values.get("batched_search_config") is not None:
        values["batched_search_config"] = search.BatchedSearchConfig(**values["batched_search_config"])
    return search.ReversibleCavitySearchConfig(**values)


def run_block_feedback(store, contig, originals, gl, sites, observed, sample_ids,
                       *, stage1_identity, assembly_config, config, chromosome_map=None,
                       chromosome_evidence=None):
    """Return selected local blocks and their mode-specific scientific identity.

Raw Stage-1 files are never replaced. The initial L1 context and raw proposals
are shared between modes. Each round's selection, and the second context and
raw proposals, are mode-specific because selected L1 panels feed that context.
Only the outer reconstruction runner publishes genome-wide completion.
"""
    cpus = min(assembly_config.num_processes, runtime.available_cpu_count())
    # This option affects only final chromosomes. Do not invalidate or refit
    # identical local contexts merely because final refinement was toggled.
    assembly_config = replace(assembly_config,
        founder_refinement_config=FounderRefinementConfig(enabled=False))
    identity = scientific_identity(config)
    mode = identity["config"].pop("selection")
    original_identity = assembly.stage2_release_identity_record(assembly_config,
        stage1_identity=stage1_identity, sample_ids=sample_ids, input_blocks=originals,
        global_probs=gl, global_sites=sites, global_observed_mask=observed,
        chromosome_map=chromosome_map, chromosome_evidence=chromosome_evidence)
    latent = hashlib.sha256()
    for block in originals:
        fitted = getattr(block, "cavity_selected_mode", None)
        if fitted is not None:
            latent.update(assembly._array_digest(fitted.haplotypes).encode())
    identity.update(source=original_identity, original_latent_sha256=latent.hexdigest())
    selected_identity = dict(identity, selection=mode, feedback_level=2)
    final_io = AssemblyCheckpointStore(store, work_stage=f"02_feedback_{mode}_l2", contig=contig)
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
        discovery=_discovery_config(stage1_identity["config"]), criterion="bic")
    selected_rounds = []
    current = originals
    for level in (1, 2):
        selection_stage = f"02_feedback_{mode}_l{level}"
        selection_identity = dict(identity, selection=mode, feedback_level=level)
        selection_io = AssemblyCheckpointStore(store, work_stage=selection_stage, contig=contig)
        selection_io.bind(selection_identity)
        saved = selection_io.load("selected")
        if saved is not None:
            current = saved["blocks"]
            selected_rounds.append(current)
            print(f"[Feedback] {contig}: resumed L{level} {mode} selected panels", flush=True)
            continue

        stage = "02_feedback_l1" if level == 1 else selection_stage
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
            assembly_io = AssemblyCheckpointStore(store, work_stage=stage+"_assembly", contig=contig)
            with parallel.numba_thread_scope(cpus):
                context = assembly.assemble_chromosome(current, gl, sites, observed, sample_ids,
                    stage1_identity=dict(stage1_identity, feedback_pass=pass_identity),
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

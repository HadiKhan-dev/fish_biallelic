"""refinement / conditioning for the canonical reconstruction pipeline."""
from __future__ import annotations
from haplotype_reconstruction import PACKAGE_ROOT

import numpy as np
from dataclasses import asdict, dataclass, replace

import hashlib
import json
import os
from pathlib import Path
import time
import pandas as pd
from numba import njit, prange, get_num_threads
import haplotype_reconstruction.painting.components as painting_components
import haplotype_reconstruction.refinement.model as refinement_model

def paint_final_phase(t09, positions, phase_map):
    """Split at actual phase changes, including changes inside an old bin.

Coordinates are half-open, as in the source painting. Original gaps and
component boundaries survive unchanged; no founder namespace is merged and
no inferred allele fill is represented as a new founder label.
"""
    positions=np.asarray(positions,dtype=np.int64)
    changes=[positions[1:][row[1:]!=row[:-1]] for row in phase_map]
    output=[]
    for component in t09.painting_bundle.components:
        samples=[]
        for index,sample in enumerate(component.painting.samples):
            chunks=[];cuts=changes[index]
            for source in sample.chunks:
                lo=np.searchsorted(cuts,source.start,side='right')
                hi=np.searchsorted(cuts,source.end,side='left')
                boundaries=np.r_[source.start,cuts[lo:hi],source.end]
                for left,right in zip(boundaries[:-1],boundaries[1:]):
                    marker=int(np.searchsorted(positions,left))
                    flip=marker<len(positions) and positions[marker]<right and phase_map[index,marker]==1
                    first,second=(source.hap2,source.hap1) if flip else (source.hap1,source.hap2)
                    if chunks and chunks[-1].end==left and (chunks[-1].hap1,chunks[-1].hap2)==(first,second):
                        old=chunks[-1];chunks[-1]=painting_components.PaintedChunk(old.start,int(right),first,second)
                    else:chunks.append(painting_components.PaintedChunk(int(left),int(right),first,second))
            samples.append(painting_components.SamplePainting(index,chunks))
        output.append(painting_components.BlockPainting((component.painting.start_pos,component.painting.end_pos),samples))
    return tuple(output)


SCHEMA = "stage11-pedigree-conditioned-refinement-v1"


@dataclass(frozen=True)
class PhaseScaffold:
    reference_alleles: np.ndarray
    phase_bins: np.ndarray
    component_ids: np.ndarray
    component_bin_edges: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Stage11ChromosomeProduct:
    schema: str
    contig: str
    sample_ids: tuple[str, ...]
    positions: np.ndarray
    component_ids: np.ndarray
    original_component_manifest: dict
    corrected_component_paintings: tuple[painting_components.BlockPainting, ...]
    reference_alleles: np.ndarray
    raw_observed_mask: np.ndarray
    allele_calls: np.ndarray
    allele_call_support: np.ndarray
    call_provenance: np.ndarray
    ordered_genotype_probability: np.ndarray
    phase_zero_probability: np.ndarray
    phase_flip: np.ndarray
    inferred_phase_map: np.ndarray
    phase_map: np.ndarray
    phase_link_support: np.ndarray
    edge_parent: np.ndarray
    edge_child: np.ndarray
    edge_child_slot: np.ndarray
    transmission_zero_probability: np.ndarray
    recombination_probability: np.ndarray
    identity: dict
    summary: dict


@njit(cache=True, parallel=True)
def _map_reference(positions, source_columns, frozen, offsets, chunks):
    samples = len(offsets)-1
    result = np.full((samples,len(positions),2),-1,dtype=np.int8)
    for sample in prange(samples):
        chunk = offsets[sample]
        for index in range(len(positions)):
            pos = positions[index]
            while chunk < offsets[sample+1] and chunks[chunk,1] <= pos:
                chunk += 1
            if chunk >= offsets[sample+1] or chunks[chunk,0] > pos:
                continue
            column = source_columns[index]
            if column < 0:
                continue
            for track in range(2):
                founder = chunks[chunk,track+2]
                if 0 <= founder < frozen.shape[0]:
                    result[sample,index,track] = frozen[founder,column]
    return result


def prepare_phase_scaffold(t09, positions):
    """Decode only frozen hard alleles; unknowns are never argmaxed to REF."""
    t09 = painting_checkpoints.validate_t09_component_checkpoint(t09)
    pos = np.asarray(positions,dtype=np.int64)
    blocks = core_runtime.validate_phase_component_manifest(t09.component_manifest)
    reference = np.full((len(t09.sample_ids),len(pos),2),-1,dtype=np.int8)
    phase_bins = np.full(len(pos),-1,dtype=np.int64)
    component_ids = np.full(len(pos),-1,dtype=np.int32)
    all_edges=[];offset=0
    for index,(block,component) in enumerate(zip(blocks,t09.painting_bundle.components)):
        source_pos=np.asarray(block.positions,dtype=np.int64)
        frozen=np.asarray(block.missing_aware_inference_discrete_haps)
        if frozen.shape != (len(component.founder_keys),len(source_pos)) or np.any(~np.isin(frozen,(-1,0,1))):
            raise ValueError("frozen component allele matrix has incompatible rows/sites")
        lo,hi=np.searchsorted(pos,[source_pos[0],source_pos[-1]],side="left")
        hi=int(np.searchsorted(pos,source_pos[-1],side="right"));lo=int(lo)
        local=pos[lo:hi]
        cols=np.searchsorted(source_pos,local)
        valid=cols<len(source_pos)
        valid[valid]&=source_pos[cols[valid]]==local[valid]
        cols=np.where(valid,cols,-1).astype(np.int64)
        chunks=np.asarray([tuple(chunk) for sample in component.painting.samples for chunk in sample.chunks],dtype=np.int64).reshape((-1,4))
        starts=np.concatenate(([0],np.cumsum([len(sample.chunks) for sample in component.painting.samples]))).astype(np.int64)
        reference[:,lo:hi]=_map_reference(local,cols,np.asarray(frozen,dtype=np.int8),starts,chunks)
        diagnostic=component.ragged_diagnostics
        if diagnostic is not None:
            edges=np.asarray(diagnostic.bin_edges,dtype=np.int64)
        else:
            # Complete/empty reference components still use real SNP coordinates.
            edges=np.unique(np.r_[source_pos[::100],source_pos[-1]+1])
        all_edges.append(edges)
        bins=np.searchsorted(edges,local,side="right")-1
        good=(bins>=0)&(bins<len(edges)-1)
        phase_bins[lo:hi]=np.where(good,offset+bins,-1)
        component_ids[lo:hi]=index
        offset += len(edges)-1
    return PhaseScaffold(reference,phase_bins,component_ids,tuple(all_edges))


def corrected_paintings(t09, positions, phase_map, scaffold):
    """Split original chunks at phase-bin edges; preserve gaps and local IDs."""
    output=[]
    for component,edges in zip(t09.painting_bundle.components,scaffold.component_bin_edges):
        samples=[]
        for sample_index,sample in enumerate(component.painting.samples):
            chunks=[]
            for chunk in sample.chunks:
                boundaries=np.r_[chunk.start,edges[(edges>chunk.start)&(edges<chunk.end)],chunk.end]
                for left,right in zip(boundaries[:-1],boundaries[1:]):
                    marker=int(np.searchsorted(positions,left))
                    swap=(marker<len(positions) and positions[marker]<right
                          and phase_map[sample_index,marker] == 1)
                    first,second=(chunk.hap2,chunk.hap1) if swap else (chunk.hap1,chunk.hap2)
                    if chunks and chunks[-1].end==left and (chunks[-1].hap1,chunks[-1].hap2)==(first,second):
                        previous=chunks[-1];chunks[-1]=painting_components.PaintedChunk(previous.start,int(right),first,second)
                    else:
                        chunks.append(painting_components.PaintedChunk(int(left),int(right),first,second))
            samples.append(painting_components.SamplePainting(sample_index,chunks))
        output.append(painting_components.BlockPainting((component.painting.start_pos,component.painting.end_pos),samples))
    return tuple(output)


def refinement_code_identity():
    return {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__),Path(refinement_model.__file__),
                         (PACKAGE_ROOT / 'refinement/model.py'),
                         (PACKAGE_ROOT / 'refinement/messages.py'),
                         (PACKAGE_ROOT / 'refinement/messages.py'),
                         (PACKAGE_ROOT / 'refinement/messages.py'),
                         (PACKAGE_ROOT / 'refinement/messages.py'),
                         (PACKAGE_ROOT / 'refinement/messages.py'),
                         (PACKAGE_ROOT / 'refinement/messages.py'),
                         (PACKAGE_ROOT / 'refinement/messages.py'),
                         (PACKAGE_ROOT / 'core/genetic_map.py'),
                         (PACKAGE_ROOT / 'refinement/model.py'))}


def relationship_identity(relationships):
    frame=relationships.loc[:,["Sample","ParentState","Parent1","Parent2"]].astype(object)
    records=frame.where(pd.notna(frame),None).to_dict(orient="records")
    return hashlib.sha256(json.dumps(records,sort_keys=True,allow_nan=False).encode()).hexdigest()


def refine_t09_chromosome(t09, gl, positions, observed, relationships, sample_ids, *,
                          contig, config=refinement_model.FamilyRefinementConfig(), work_path=None,
                          identity=None, checkpoint_threads=1, progress_callback=None,
                          checkpoint_min_seconds=120.0, warm_start=None, chromosome_map=None):
    """Single independently checkpointed chromosome, with a fixed pedigree."""
    names=tuple(map(str,sample_ids))
    t09=painting_checkpoints.validate_t09_component_checkpoint(t09,expected_sample_ids=names)
    pedigree_components._validate_release_array_identity(t09,np.asarray(gl),np.asarray(positions),np.asarray(observed))
    scaffold=prepare_phase_scaffold(t09,positions)
    if chromosome_map is not None:
        config = replace(config, recombination_rate=chromosome_map.fallback_rate_per_bp)
    config=config.validated()
    scientific_identity={"schema":SCHEMA,"model":refinement_model.MODEL_VERSION,
        "config":asdict(config),"sample_ids":names,"contig":str(contig),
        "t09_release":t09.release_identity.record(),"t09_painting":t09.painting_product_identity.record(),
        "pedigree_sha256":relationship_identity(relationships),"code":refinement_code_identity(),
        "external":identity}
    if chromosome_map is not None:
        if chromosome_map.contig != str(contig):
            raise ValueError("Stage11 chromosome map name differs from input contig")
        scientific_identity["recombination_map"] = chromosome_map.identity()
    work=None if work_path is None else Path(work_path)
    if not np.isfinite(checkpoint_min_seconds) or checkpoint_min_seconds < 0:
        raise ValueError("checkpoint_min_seconds must be finite and nonnegative")
    def compression_threads():
        return get_num_threads() if checkpoint_threads is None else int(checkpoint_threads)
    resume=None
    if work is not None and work.is_file():
        saved=core_checkpoints.read(str(work),nthreads=compression_threads())
        if saved["identity"]!=scientific_identity:
            raise ValueError("Stage11 iteration checkpoint inputs/configuration differ")
        resume=saved["messages"]
    last_saved=-np.inf
    checkpoints_written=0
    def save(messages):
        nonlocal last_saved, checkpoints_written
        if progress_callback is not None:
            progress_callback(messages)
        final=(messages.deltas[-1] < config.tolerance or messages.iteration >= config.max_iterations)
        now=time.perf_counter()
        if work is not None and (final or now-last_saved >= checkpoint_min_seconds):
            core_checkpoints.write(str(work),{"identity":scientific_identity,"messages":messages},
                                nthreads=compression_threads())
            last_saved=time.perf_counter();checkpoints_written+=1
        print(f"  [T11 {contig}] iteration={messages.iteration} delta={messages.deltas[-1]:.6g}",flush=True)
    result=refinement_model.refine_family(gl,observed,scaffold.reference_alleles,positions,scaffold.phase_bins,
        scaffold.component_ids,relationships,names,config=config,resume=resume,
        warm_start=warm_start if resume is None else None,checkpoint_callback=save,
        chromosome_map=chromosome_map)
    parents,children,slots=refinement_model.pedigree_edges(relationships,names)
    genotype_calls=np.max(np.stack((result.ordered_genotype_probability[:,:,0],
        result.ordered_genotype_probability[:,:,1]+result.ordered_genotype_probability[:,:,2],
        result.ordered_genotype_probability[:,:,3]),axis=2),axis=2)>=config.allele_call_probability
    summary={"contig":str(contig),"samples":len(names),"sites":len(positions),
        "observed_parent_edges":len(parents),"converged":result.converged,
        "iterations":result.messages.iteration,"final_delta":result.messages.deltas[-1],
        "joint_two_generation_clusters":len(result.messages.branch_clusters),
        "elapsed_seconds":result.elapsed_seconds,"imputation_enabled":config.impute_missing,
        "reference_called_alleles":int(np.sum(scaffold.reference_alleles>=0)),
        "output_called_alleles":int(np.sum(result.allele_calls>=0)),
        "family_filled_alleles":int(np.sum(result.provenance==2)),
        "revised_reference_alleles":int(np.sum(result.provenance==3)),
        "iteration_checkpoints_written":checkpoints_written,
        "confident_genotypes":int(np.sum(genotype_calls)),
        "phase_supported_marker_samples":int(np.sum(result.phase_flip>=0)),
        "founder_namespaces_merged":False,"pedigree_updated":False,
        "probability_interpretation":"loopy sum-product conditional on accepted T09 alleles and phase scaffold",
        "phase_policy":"all families" if config.correct_incomplete_parent_phase else "correct M2; preserve M0/M1 scaffold phase",
        "ordered_genotype_probability_frame":"internal family frame; reverse heterozygote states where inferred_phase_map differs from phase_map",
        "phase_link_support_frame":"inferred_phase_map, not retained upstream painting confidence",
        "retained_call_support_interpretation":"conditional on scaffold, not independent measurement confidence",
        "uncertain_adjacent_phase_links":int(np.sum(result.phase_link_support < config.phase_call_probability)),
        "call_provenance_codes":{"0":"unresolved","1":"retained_scaffold","2":"filled_scaffold_gap","3":"revised_scaffold_allele"}}
    return Stage11ChromosomeProduct(SCHEMA,str(contig),names,np.asarray(positions),scaffold.component_ids,
        t09.component_manifest,corrected_paintings(t09,positions,result.phase_map,scaffold),
        scaffold.reference_alleles,np.asarray(observed),result.allele_calls,result.allele_call_support,
        result.provenance,result.ordered_genotype_probability,result.phase_zero_probability,result.phase_flip,
        result.inferred_phase_map,result.phase_map,result.phase_link_support,
        parents,children,slots,result.transmission_zero_probability,result.recombination_probability,
        scientific_identity,summary)


def config_from_environment():
    value=os.environ.get("BHD_FAMILY_IMPUTATION","0").strip().lower()
    if value not in ("0","1","false","true"):
        raise ValueError("BHD_FAMILY_IMPUTATION must be 0/1 or false/true")
    return refinement_model.FamilyRefinementConfig(impute_missing=value in ("1","true"))


def _write_summary_table(output, stage, summaries):
    temporary=output/f".{stage}.csv.tmp"
    pd.DataFrame(summaries).to_csv(temporary,index=False)
    temporary.replace(output/f"{stage}.csv")

import haplotype_reconstruction.core.checkpoints as core_checkpoints
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.painting.checkpoints as painting_checkpoints
import haplotype_reconstruction.pedigree.components as pedigree_components

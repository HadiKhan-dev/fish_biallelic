"""Exact colour execution with live threads inside long independent blocks."""
import numpy as np
from ..execution import claim_threads
from ..execution.budget import current_budget
from . import map_solver as ms


def sweep_colour(fm, state, blocks, cavities=None):
    w, g = fm.workspace, fm.graph
    accepted=np.zeros(len(blocks),np.int64);gains=np.zeros(len(blocks))
    decisions=np.zeros((len(blocks),256,4));counts=np.zeros((len(blocks),1),np.int64)
    saved=[None]*len(blocks)
    def task(index, sample):
        def run(budget):
            claim_threads()
            if fm.roots[sample]:
                starts,trans=((fm.frozen_starts,fm.frozen_trans) if w.frozen[sample] else (fm.free_starts,fm.free_trans))
                return ms._root_block(sample,w.potentials,state.genotype,state.selector,state.frame,w.orientation,
                    starts,trans,w.anchors[sample],g.group_child,fm.child_offsets,fm.child_groups,fm.child_slots,
                    fm.log_keep,fm.log_error,fm.log_match,fm.log_miss,fm.quantum,decisions[index],counts[index],fm.reverse,True)
            group=int(g.node_group[sample])
            if cavities is not None:
                saved[index]=(state.genotype[sample].copy(),state.selector[group].copy())
            return ms._child_block(group,w.potentials,state.genotype,state.selector,state.frame,w.orientation,
                fm.roots,g.group_parent,g.group_child,fm.child_offsets,fm.child_groups,fm.child_slots,
                fm.marker_offsets,fm.marker_sites,fm.marker_theta,fm.log_keep,fm.log_error,fm.log_match,
                fm.log_miss,fm.quantum,decisions[index],counts[index],fm.reverse,cavities,True)
        return run
    for i,value in current_budget().concurrent([task(i,int(sample)) for i,sample in enumerate(blocks)],
                                               fm.sites*144,label='family colour blocks'):
        accepted[i],gains[i]=value
    if cavities is not None:
        for i,sample in enumerate(blocks):
            if saved[i] is not None:
                old_g,old_s=saved[i]
                ms.cavity_counts.update_group(cavities,int(g.node_group[sample]),0,old_g,old_s,state.genotype,state.selector)
    return accepted,gains,decisions,counts

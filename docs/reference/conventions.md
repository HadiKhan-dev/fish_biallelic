# Tie and random conventions

Generated from `core/ties.py`, its policy call sites and `core/random.py` without importing numerical modules. Expressions retain the code's arithmetic. See the [tie guide](../developer/ties.md) for their meaning.

## Shared quantities

| Name | Source expression |
|---|---|
| `GL_NOISE` | `1e-06` |
| `NOISE_MARGIN` | `16.0` |
| `NOISE_SCALE` | `2.0 ** 10` |
| `NOISE_MOTION` | `{'nll': 5.5e-05, 'h_step': 1e-05, 'improvement': 5.5e-05, 'sample': 5e-06, 'cell': 2e-06, 'probability': 1e-06}` |
| `QUANTUM_KINDS` | `{'nll': (noise_quantum(NOISE_MOTION['nll']), 2.0 ** (-30) * NOISE_SCALE), 'h_step': (noise_quantum(NOISE_MOTION['h_step']), 0.0), 'improvement': (noise_quantum(NOISE_MOTION['improvement']), 0.0), 'sample': (noise_quantum(NOISE_MOTION['sample']), 2.0 ** (-36) * NOISE_SCALE), 'cell': (noise_quantum(NOISE_MOTION['cell']), 0.0), 'probability': (noise_quantum(NOISE_MOTION['probability']), 0.0)}` |
| `WIDE_CLASS_QUANTA` | `4.0` |
| `BOUNDARY_GUARD` | `10.0` |
| `ASSEMBLY_QUANTA` | `{'refine_F': (2.0 ** (-9), 2.0 ** (-26)), 'join_ll': (2.0 ** (-19), 2.0 ** (-34)), 'join_ll_wide': (2.0 ** (-19), 2.0 ** (-30)), 'join_proxy': (2.0 ** (-19), 2.0 ** (-24)), 'beam': (2.0 ** (-18), 2.0 ** (-30)), 'count_margin': (2.0 ** (-26), 0.0), 'paint': (2.0 ** (-30), 2.0 ** (-30))}` |
| `ASSEMBLY_MOTION` | `{'refine_F': (9e-05, 5e-10), 'join_ll': (1.3e-07, 2.7e-12), 'join_ll_wide': (1.3e-07, 5.7e-11), 'join_proxy': (1.1e-07, 2.1e-09), 'beam': (6.5e-08, 0.0), 'count_margin': (1.3e-10, 0.0), 'paint': (0.0, 0.0)}` |
| `_POLICY_KINDS` | `{'assembly.refine.accept': 'refine_F', 'assembly.join.panel': 'join_ll', 'assembly.join.wide': 'join_ll_wide', 'assembly.join.proxy': 'join_proxy', 'assembly.beam': 'beam', 'assembly.count.margin': 'count_margin', 'assembly.paint': 'paint'}` |

## Named policy call sites

The name identifies a decision. The kind selects its scale rule; a fixed width is an explicit expression from that decision's parameter record. A dynamic expression is displayed as source rather than guessed. Secondary scientific preferences are documented in the owning method, not encoded by reversing this name.

| Policy name or expression | Constructor and arguments | Source |
|---|---|---|
| `'assembly.alleles.ascent'` | `fixed_policy('assembly.alleles.ascent', self.improvement, reverse=reverse)` | [parameters.py:126](../../haplotype_reconstruction/science/assembly/alleles/parameters.py) |
| `'assembly.completion.abundant'` | `ties.policy('assembly.completion.abundant', kind='probability')` | [occupancy.py:45](../../haplotype_reconstruction/science/assembly/completion/occupancy.py) |
| `'assembly.completion.neighbor'` | `ties.policy('assembly.completion.neighbor', kind='probability')` | [occupancy.py:46](../../haplotype_reconstruction/science/assembly/completion/occupancy.py) |
| `'assembly.join_proposals'` | `ties.policy('assembly.join_proposals', kind='join_proxy', reverse=reverse)` | [proposals.py:206](../../haplotype_reconstruction/science/assembly/join/proposals.py) |
| `name` | `ties.policy(name, kind='join_ll', reverse=settings.reverse)` | [state.py:47](../../haplotype_reconstruction/science/assembly/join/state.py) |
| `name` | `ties.policy(name, kind='join_ll_wide', reverse=settings.reverse)` | [state.py:49](../../haplotype_reconstruction/science/assembly/join/state.py) |
| `name` | `ties.policy(name, kind='join_proxy', reverse=settings.reverse)` | [state.py:51](../../haplotype_reconstruction/science/assembly/join/state.py) |
| `'assembly.refine.accept'` | `policy('assembly.refine.accept', reverse=reverse)` | [acceptance.py:1464](../../haplotype_reconstruction/science/assembly/refinement/acceptance.py) |
| `'assembly.refine_death_class'` | `ties.policy('assembly.refine_death_class', kind='refine_F', reverse=engine.reverse)` | [count.py:578](../../haplotype_reconstruction/science/assembly/refinement/count.py) |
| `'assembly.refine_refit'` | `ties.fixed_policy('assembly.refine_refit', quantum, reverse=reverse)` | [count.py:803](../../haplotype_reconstruction/science/assembly/refinement/count.py) |
| `'assembly.refine_eligible'` | `ties.fixed_policy('assembly.refine_eligible', quantum, reverse=engine.reverse)` | [count.py:855](../../haplotype_reconstruction/science/assembly/refinement/count.py) |
| `'assembly.refine_terminal'` | `ties.fixed_policy('assembly.refine_terminal', quantum, reverse=engine.reverse)` | [count.py:1232](../../haplotype_reconstruction/science/assembly/refinement/count.py) |
| `'assembly.refine_terminal'` | `ties.fixed_policy('assembly.refine_terminal', tie_rules.quantum('refine_F', scale), reverse=engine.reverse)` | [count.py:1251](../../haplotype_reconstruction/science/assembly/refinement/count.py) |
| `'assembly.refine_eligible'` | `ties.fixed_policy('assembly.refine_eligible', tie_rules.quantum('count_margin'), reverse=reverse)` | [count.py:904](../../haplotype_reconstruction/science/assembly/refinement/count.py) |
| `'assembly.refine_terminal'` | `ties.fixed_policy('assembly.refine_terminal', tie_rules.quantum('refine_F', max((abs(v) for v in values))), reverse=engine.reverse)` | [count.py:2360](../../haplotype_reconstruction/science/assembly/refinement/count.py) |
| `'assembly.refine_terminal'` | `ties.fixed_policy('assembly.refine_terminal', tie_rules.quantum('refine_F', max((abs(v) for v in values))), reverse=engine.reverse)` | [count.py:2509](../../haplotype_reconstruction/science/assembly/refinement/count.py) |
| `'assembly.refine_converge'` | `ties.fixed_policy('assembly.refine_converge', quantum, reverse=self.reverse)` | [fit.py:1604](../../haplotype_reconstruction/science/assembly/refinement/fit.py) |
| `'assembly.refine.accept'` | `policy('assembly.refine.accept')` | [parameters.py:18](../../haplotype_reconstruction/science/assembly/refinement/parameters.py) |
| `'assembly.refine.polish'` | `ties.fixed_policy('assembly.refine.polish', tie_rules.quantum('refine_F', max((abs(v) for v in values))), reverse=engine.reverse)` | [run.py:170](../../haplotype_reconstruction/science/assembly/refinement/run.py) |
| `'discovery.fixed_k.a_band'` | `fixed_policy('discovery.fixed_k.a_band', self.quanta.sample, reverse=self.reverse)` | [workspace.py:520](../../haplotype_reconstruction/science/discovery/fixed_k/workspace.py) |
| `'discovery.fixed_k.h_flip'` | `fixed_policy('discovery.fixed_k.h_flip', self.quanta.h_step, reverse=self.reverse)` | [workspace.py:521](../../haplotype_reconstruction/science/discovery/fixed_k/workspace.py) |
| `'discovery.fixed_k.cycle'` | `fixed_policy('discovery.fixed_k.cycle', self.quanta.nll, reverse=self.reverse)` | [workspace.py:522](../../haplotype_reconstruction/science/discovery/fixed_k/workspace.py) |
| `'discovery.k1'` | `fixed_policy('discovery.k1', sample_quantum, reverse=reverse)` | [moves.py:328](../../haplotype_reconstruction/science/discovery/moves.py) |
| `'discovery.data_birth'` | `fixed_policy('discovery.data_birth', quantum, reverse=reverse)` | [moves.py:882](../../haplotype_reconstruction/science/discovery/moves.py) |
| `'discovery.replacement'` | `fixed_policy('discovery.replacement', _nll_quantum(pool, None), reverse=workspace.reverse)` | [moves.py:1235](../../haplotype_reconstruction/science/discovery/moves.py) |
| `'discovery.start_screen'` | `fixed_policy('discovery.start_screen', workspace.quanta.nll, reverse=workspace.reverse)` | [moves.py:155](../../haplotype_reconstruction/science/discovery/moves.py) |
| `'discovery.k1'` | `fixed_policy('discovery.k1', workspace.quanta.nll, reverse=workspace.reverse)` | [moves.py:389](../../haplotype_reconstruction/science/discovery/moves.py) |
| `'discovery.data_birth'` | `fixed_policy('discovery.data_birth', workspace.quanta.nll, reverse=workspace.reverse)` | [moves.py:1114](../../haplotype_reconstruction/science/discovery/moves.py) |
| `'discovery.panel.dedup'` | `fixed_policy('discovery.panel.dedup', q, reverse=reverse)` | [panel.py:474](../../haplotype_reconstruction/science/discovery/panel.py) |
| `'discovery.panel.order'` | `fixed_policy('discovery.panel.order', q, reverse=reverse)` | [panel.py:457](../../haplotype_reconstruction/science/discovery/panel.py) |
| `'discovery.release.cell'` | `fixed_policy('discovery.release.cell', (quanta if quanta is not None else BlockQuanta.floors()).cell, reverse=_CELL_BAND_REVERSE)` | [release.py:60](../../haplotype_reconstruction/science/discovery/release.py) |
| `'discovery.selection'` | `fixed_policy('discovery.selection', quantum, reverse=reverse)` | [search.py:206](../../haplotype_reconstruction/science/discovery/search.py) |
| `'discovery.archive'` | `fixed_policy('discovery.archive', self.quantum, reverse=self.reverse)` | [search.py:297](../../haplotype_reconstruction/science/discovery/search.py) |
| `'discovery.archive'` | `fixed_policy('discovery.archive', self.quantum, reverse=self.reverse)` | [search.py:317](../../haplotype_reconstruction/science/discovery/search.py) |
| `'discovery.archive'` | `fixed_policy('discovery.archive', self.quanta.nll, reverse=self.reverse)` | [search.py:574](../../haplotype_reconstruction/science/discovery/search.py) |
| `'painting.relabel_rank'` | `ties.policy('painting.relabel_rank', kind='nll', reverse=convention_reverse)` | [label_continuity.py:878](../../haplotype_reconstruction/science/painting/label_continuity.py) |
| `'pedigree.junction.trace'` | `policy('pedigree.junction.trace', kind='probability', reverse=reverse)` | [junctions.py:324](../../haplotype_reconstruction/science/pedigree/junctions.py) |
| `'pedigree.screen.rank'` | `tie_rules.policy('pedigree.screen.rank', kind='probability', reverse=reverse)` | [panel.py:176](../../haplotype_reconstruction/science/pedigree/panel.py) |
| `'pedigree.bridge.hall'` | `policy('pedigree.bridge.hall', kind='probability', reverse=reverse)` | [projection.py:470](../../haplotype_reconstruction/science/pedigree/projection.py) |
| `'pedigree.bridge.search'` | `policy('pedigree.bridge.search', kind='probability', reverse=reverse)` | [projection.py:472](../../haplotype_reconstruction/science/pedigree/projection.py) |
| `'pedigree.bridge.pivot'` | `policy('pedigree.bridge.pivot', kind='probability', reverse=reverse)` | [projection.py:473](../../haplotype_reconstruction/science/pedigree/projection.py) |
| `'selection.bank.cavity'` | `fixed_policy('selection.bank.cavity', quanta.nll, reverse=reverse)` | [bank.py:231](../../haplotype_reconstruction/science/selection/bank.py) |
| `kind` | `fixed_policy(kind, quantum, reverse=reverse)` | [bank.py:180](../../haplotype_reconstruction/science/selection/bank.py) |
| `'selection.start.outcome'` | `fixed_policy('selection.start.outcome', tolerance, reverse=reverse)` | [rounds.py:84](../../haplotype_reconstruction/science/selection/rounds.py) |
| `'selection.start.set'` | `fixed_policy('selection.start.set', tolerance, reverse=reverse)` | [rounds.py:223](../../haplotype_reconstruction/science/selection/rounds.py) |
| `'selection.search.rank'` | `fixed_policy('selection.search.rank', tolerance, reverse=reverse)` | [search.py:523](../../haplotype_reconstruction/science/selection/search.py) |
| `'selection.merge.consensus'` | `fixed_policy('selection.merge.consensus', CONSENSUS_TIE, reverse=reverse)` | [search.py:86](../../haplotype_reconstruction/science/selection/search.py) |
| `'selection.search.merge_representative'` | `fixed_policy('selection.search.merge_representative', tolerance, reverse=reverse)` | [search.py:257](../../haplotype_reconstruction/science/selection/search.py) |
| `'selection.search.winner'` | `fixed_policy('selection.search.winner', tolerance, reverse=reverse)` | [search.py:886](../../haplotype_reconstruction/science/selection/search.py) |
| `'selection.search.rank'` | `fixed_policy('selection.search.rank', tolerance, reverse=reverse)` | [search.py:832](../../haplotype_reconstruction/science/selection/search.py) |

## Random streams

The stream name preserves generator construction and draw order, not just a seed. `simulation.structure` uses Python's MT generator through `structure_stream`; the other registered streams use NumPy's default PCG64 generator. See [simulation](../methods/simulation.md) and [pedigree resampling](../methods/pedigree.md#pd16).

| Stream |
|---|
| `assembly.count.bootstrap` |
| `pedigree.bootstrap` |
| `pedigree.relationship.reference` |
| `simulation.structure` |
| `simulation.master` |
| `simulation.child` |
| `simulation.founders` |
| `simulation.reads` |
| `simulation.contig_reads` |
| `simulation.sample_reads` |
| `simulation.sample_region` |
| `simulation.observation` |
| `templates.example` |

Simulation realises an unspecified seed once during initialisation and persists it. Resume reuses that realised seed. Founder streams use configured contig order; read seeds are consumed for every contig, including already completed ones. Since 4.8 each sample's reads come from `simulation.sample_reads` (a seed drawn on its contig's read stream, and the sample's index), so blocks of samples may be simulated anywhere, at any worker count, with the same counts. Bootstrap draws preserve their original vector shapes and replicate order. Renaming a stream or replacing a loop with a bulk draw requires explicit identity proof.

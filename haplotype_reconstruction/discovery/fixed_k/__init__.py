"""Hard fixed-K factorisation of one block: H (K binary rows) x A (each sample's row pair).

The fit alternates an A step (each sample's best unordered pair, a row with an unknown copy, or
two unknown copies, ranked by uncapped cost and scored by a binned switching Viterbi) and an H
step (Gauss-Seidel founder-bit updates on capped costs) until H is a fixed point or the iteration
limit. Modules: workspace (evidence-only tables and the per-block caches), assign (A step),
update (H step), fit (the frontier engine over many starts).
"""

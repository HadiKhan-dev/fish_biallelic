## Local founder discovery and panel selection

### Local founder discovery

Discovery proposes, independently in each block of a contig, a panel of binary founder haplotypes
and its founder count $K$. Contigs are divided into consecutive 200-site blocks, irrespective of
physical spacing. Within a block, discovery uses the $M$ discovery sites (total depth of at least
five reads over all samples) and a founder-construction view of the evidence in which samples
flagged by contamination quality control are unobserved. We write $p_{ij}(g)$ for the genotype
likelihood $P(r_{ij},a_{ij}\mid g)$ normalised over $g$; a cell without reads has $p_{ij}(g)=1/3$
and contributes nothing to any cost below. Samples with at least one observed discovery site are
*active* ($N_a$ of them); a block without any yields no panel. Samples are processed in canonical
identifier order; identifiers affect nothing else. $K$ is bounded only by the identifiability
ceiling $\min(2N_a,2^M)$.

**Fixed-$K$ model.** A panel $H=(h_{kj})$ has $K$ distinct binary rows. Each sample takes one of
$(K+1)(K+2)/2$ diploid states $z$: an unordered row pair $(k,k')$, a row with an *unknown copy*
$(k,U)$, or two unknown copies $(U,U)$. An unknown copy is a chromosome that no row explains; at
each observed site it takes the allele that fits best, at a penalty of $\lambda=0.5$ nats. The cost
of state $z$ at an observed cell is

$$
c_{ij}(z)=\begin{cases}
-\log p_{ij}(h_{kj}+h_{k'j}), & z=(k,k'),\\
\lambda+\min_{g\in\{h_{kj},\,h_{kj}+1\}}\left[-\log p_{ij}(g)\right], & z=(k,U),\\
2\lambda+\min_{g}\left[-\log p_{ij}(g)\right], & z=(U,U),
\end{cases}
$$

and zero at unobserved cells. Sites are grouped into consecutive bins of ten with costs
$C_{ib}(z)=\sum_{j\in b}c_{ij}(z)$, and the panel's negative log-likelihood is a switching Viterbi
cost over the $B$ bins of the block,

$$
\mathrm{NLL}(H)=\sum_{i}\min_{z_1,\ldots,z_B}\Big[\sum_{b=1}^{B}C_{ib}(z_b)+\tau\sum_{b=2}^{B}\mathbb{1}(z_b\neq z_{b-1})\Big],
$$

with $\tau=10$ nats per change of state between adjacent bins. Discovery uses no genetic map.

**Fitting.** A start panel is fitted by alternating two hard steps. The A step gives each sample the
state of least whole-block cost $\sum_b C_{ib}(z)$. States within a tie band of that minimum are
tied; the tied category with fewer unknown copies is preferred, and a sample tied among several
states of that category is summarised by the multiset intersection of their founder pairs (a
differing copy becomes unknown), its evidence being shared equally among the tied states. The data
cannot tell which tied founder supplies a copy, so no tied founder is privileged. The H step then
updates rows one at a time in decreasing order of copy usage, setting each allele $h_{kj}$ to
minimise the summed cost of the samples carrying row $k$, given their partners' current alleles,
with every cell's cost capped at its $(U,U)$ cost; an allele changes only when the two allele costs
differ by at least a tie quantum. When rows tie in usage, both consistent update orders are run and
both endpoints kept. A trajectory ends at a fixed point, after 50 iterations, or on entering a
cycle, when it returns the cycle's lowest-NLL state; a fit whose rows coincide is refitted at the
smaller $K$.

**Founder count.** NLL ranks panels of equal $K$; different $K$ are compared by a regularised
leave-one-sample-out ("cavity") predictive score. Each active sample $i$ is held out in turn. Its
state is predicted from a Dirichlet–multinomial prior over states with concentration $\alpha=0.5$,
base masses $\pi^0_z$ (uniform over ordered founder pairs, with mass $\omega=0.02$ on states
containing an unknown copy, a quarter of it on $(U,U)$) and the other samples' tie-shared state
counts $n^{(-i)}_z$. Founder alleles at each site are predicted from a pairwise Markov random field
built from the other samples' assigned states (unary terms from homozygous and row–unknown carriers,
pairwise terms from heterozygous pairs, uniform prior), approximated by mean field [@jordan1999]. The score is

$$
\mathcal{L}^{\mathrm{cav}}(H)=\sum_{i}\log\sum_{z}\frac{n^{(-i)}_z+\alpha\pi^0_z}{N_a-1+\alpha}\prod_{j}\bar p^{(-i)}_{ij}(z)\;-\;\log\big[K(K+1)\big]\;-\;\log\binom{2^{M}}{K},
$$

where $\bar p^{(-i)}_{ij}(z)$ is the expected genotype likelihood of state $z$ under the mean-field
allele marginals, averaged over the best converged mean-field optima; an unknown copy contributes
both alleles with probability one half. The second term is the proper prior $P(K)=1/[K(K+1)]$, and
the third is the code length of an unordered set of $K$ distinct binary rows of length $M$. The
panel and assignments are fitted from all samples, including the held-out one, so
$\mathcal{L}^{\mathrm{cav}}$ is an uncalibrated pseudo-score.

**Search.** Discovery explores complete panels by a bounded, reversible neighbourhood search. Starts
at $K=1$ are rows pooled from density-based clusters of samples (HDBSCAN* on expected genotype
agreement [@campello2013]). Moves are: deleting a
row; gauge rewires, which flip the alleles of both sides of a maximum cut of the graph linking rows
co-assigned as heterozygous pairs, at sites where each side is internally constant and the sides
differ, a flip that leaves the dosage of every sample whose copies straddle the cut unchanged; data
births, which add a row seeded from the samples with the largest excess NLL over their per-site best
genotype; soft-residual births, which add consensus rows of clustered residual haplotypes left after
subtracting each sample's well-supported founder copies; birth–delete replacements at fixed $K$;
and, when residual births do not improve the selection, births from the best $(K+1)$-founder panels
to $K+2$. Every proposal is refitted. A per-$K$ archive retains leading NLL tie classes, and the
leading class of each $K$ is scored by $\mathcal{L}^{\mathrm{cav}}$. The selected set is the best
tie class of $\mathcal{L}^{\mathrm{cav}}$ across $K$, then the smaller $K$. The search expands the
next unexpanded NLL class of the selected $K$, ordinary moves before residual births, until no
parent remains or a $K$-scaled budget is reached. The result is a local optimum of the generated
neighbourhood: a block is flagged *certified* only when its winners' neighbourhoods were generated
and scored completely without improvement, and *incomplete* when a budget omitted a whole tie class.

**Release.** The selected rows are latent: they, and any equally scored panels of the same $K$,
start local selection. Discovery also releases alleles: the carriers' capped cost differences
between alleles 0 and 1 of row $k$ at site $j$ are summed into log pseudo-odds, and an allele is
called when its logistic pseudo-probability reaches 0.85 with at least two informative carriers;
otherwise it stays unknown rather than adopting the latent value.

### Local panel selection

Selection refits each block's panel under a haplotype-copying model and chooses it by a regularised
objective, in an initial round and two rounds that use assembled founders as context.

**Copying model.** Each of a sample's two homologues copies, at each site, one of $S=K+1$ states: a
founder row, or an unknown state whose alleles are independent Bernoulli(1/2) draws at each site
(not a shared unknown founder). The state prior is $\pi=((1-u)f_1,\ldots,(1-u)f_K,u)$ with founder
frequencies $f$ and fixed unknown mass $u=0.01$. Between discovery sites $j-1$ and $j$, separated by
$x_j$ Morgans on the genetic map, each homologue independently redraws its state from $\pi$ with
probability $\epsilon_j=1-e^{-\gamma x_j}$ ($\gamma=3$ per Morgan; a redraw may return the same
state):

$$
T_j(s,t)=(1-\epsilon_j)\,\mathbb{1}(s=t)+\epsilon_j\,\pi_t .
$$

The first site draws from $\pi$. A state pair $(s,t)$ emits $p_{ij}(h_{sj}+h_{tj})$ for two
founders, $\tfrac12[p_{ij}(h_{sj})+p_{ij}(h_{sj}+1)]$ for a founder and the unknown state,
$\tfrac14p_{ij}(0)+\tfrac12p_{ij}(1)+\tfrac14p_{ij}(2)$ for two unknown states, and 1 at unobserved
cells. This is a diploid copying model [@li2003] with founders as templates; $L(H,f)$ and posterior
expectations come from the forward–backward algorithm [@baum1970; @rabiner1989] over ordered state
pairs, at $O(S^2)$ cost per sample and site since homologues redraw independently.

**Objective.** Panels are compared with

$$
J(H,f)=\log L(H,f)-D(H)-\log\big[K(K+1)\big]+\frac1K\sum_{k=1}^{K}\log(Kf_k),
$$

whose last term shrinks frequencies towards uniform and is zero there. Without context the
dictionary code is $D(H)=\log\binom{2^M}{K}$, as in discovery, so panels of different $K$ are
compared by the full objective, never by likelihood alone. In a context round, a row may instead be
coded by reference to an assembled context founder $\phi$ plus its edits, at cost
$\log V(n_\phi,d)$, where $d$ counts the sites, among $\phi$'s $n_\phi$ called sites, at which the
row differs, and $V(n,d)=\sum_{t\le d}\binom{n}{t}$. The panel's code is its shortest description,

$$
D_{\mathcal C}(H)=\min_{m}\Big[\Gamma_m(H)+\log\binom{2^{M}}{K-m}\Big],
$$

where $\Gamma_m$ is the least total edit code over matchings of $m$ context founders to rows,
obtained for every $m$ by successive shortest augmenting paths [@edmonds1972]. $J$ is a regularised best-panel criterion, not a
posterior over $K$, and its context code covers only called context sites.

**Fixed-$K$ fit.** At fixed $K$, $(H,f)$ is fitted by generalised EM [@dempster1977]. Each update
takes a frequency step, $f_k\propto\bar n_k+1/K$, where $\bar n_k$ is the expected number of initial
and redraw-destination copies of founder $k$, the maximiser of the expected complete-data objective
including the shrinkage term, kept unless $J$ falls by more than $10^{-6}$. It then flips, in the
single row with the largest total, every site whose expected complete-data emission gain is
positive. At a stationary point of these steps, the best exact one-bit flip is taken instead, with
gain $\Delta_{kj}=\log L(H^{(kj)},f)-\log L(H,f)$, where $H^{(kj)}$ is $H$ with $h_{kj}$ flipped.
The fit stops when no flip improves, when the best moves tie between different panels (keeping the
current panel), or when a flip would make two rows equal, since changing $K$ belongs to the outer
search. In context rounds the EM climbs the context-free $J$; the state along the trajectory that is
best under the context-coded objective is returned if it beats the endpoint by more than the
decision tolerance $\eta=2^{-10}$ nats, and one-bit moves towards matched context founders are then
taken while the context-coded objective rises.

**Search.** From a fitted start, each round proposes deleting any row, merging any two rows into an
occupancy-weighted consensus row, and adding any row of a candidate bank. Deletions and additions
are screened by their exact objective at the proposed frequencies, merges by an EM lower bound from
the corresponding deletion's posterior, with up to $2K$ merges scored exactly. For each kind, whole
tie classes of screened values are refitted until at least eight are taken. The best refit (best tie
class, then fewer founders) replaces the current panel only if it improves the objective by more
than $\eta$; otherwise the climb stops.

**Rounds and context.** The initial round starts from discovery's latent panel and its tied
alternatives, whose rows form the bank. Before each of the two later rounds, the previous round's
panels are assembled to the first, then the second, assembly level. Each component that exactly
covers a set of blocks is refitted to their evidence by a block-level copying model whose weights
for each block exclude that block's own emission, and a founder allele is released at site posterior
0.99 with support of at least two carriers. The refitted founders become the context and a proposal.
The bank of a context round is built by refitting, with discovery's fixed-$K$ model, the discovery
panel and its tied alternatives, earlier rounds' released panels and the context proposal (missing
alleles completed from the nearest latent rows). Each source's best panels under the cavity score
become starts, and a neighbourhood search over bank rows, scored by
$2\,\mathrm{NLL}+K\kappa (M/200)N_a$ with $\kappa=0.5$ in the form of the Bayesian information
criterion [@schwarz1978], adds its selected rows. The distinct starts in the best class of fitted
objective, with fewest founders, are each searched. The best outcome replaces the previous round's
panel only if it improves the context-coded objective by more than $\eta$.

**Release.** An allele is called at the panel value when its one-bit conditional probability
$1/(1+e^{\Delta_{kj}})$ reaches 0.85 and favourable carriers support it. A favourable carrier is a
sample whose own evidence favours the allele by more than $2^{-14}$ nats; support is their number
with carrier probability above one half or their summed carrier probability, whichever is larger,
and must reach two. A cell that passes the probability gate but lacks support is still released when
the context founder matched to its row carries the same allele and the context refit released it.
Distinct panels tied in the outcome are released conservatively: only rows present in all of them,
called only where all agree. Each sample's largest number of unknown copies at any site is passed to
assembly.

Both stages are bounded local searches with whole-tie-class budgets; neither establishes global
optimality, and every budget omission is recorded.

## Implementation and computational design

### Task-graph execution

The pipeline is a task graph; each task declares its key, records, importable action, resource model and optional contig affinity. Only model prerequisites are barriers: relatedness waits for every contig's evidence; a contig's reconstruction (discovery through painting and founder refit) needs only its own evidence and runs alongside relatedness; pedigree needs relatedness and all paintings; family phase needs the pedigree, painting and refit; recombination follows phase. Nothing feeds back upstream. Results are placed by task key, not completion order.

One runtime owns processes, threads, memory and compilation: forkserver workers start after the numerical-library policy is fixed, one CPU budget bounds processes × threads, contig lanes lend idle cores only at task boundaries, and admission follows measured memory pressure. Reachable kernels are compiled as tasks of the same graph, so a stage waits only for the kernels it needs. Contig reports and export also use the graph's parallel resources. A worker failure stops the run, keeping completed publications. Compiled kernels use Numba [@lam2015] on NumPy arrays [@harris2020], with fast-math disabled. Workflow steps (per-contig scatters and genome-wide gathers) are cuts of the same graph, usable from workflow managers such as Snakemake [@molder2021].

### Deterministic numerics

Scheduling may change when work runs, never what it computes: scientific functions receive explicit parameters, arrays, streams and storage capabilities, and read no environment variables, files, clocks or memory measurements. Sums whose order would follow worker counts, batch sizes or memory layout use a declared order, so results do not depend on thread count. Near-tie decisions sort values exactly and group them into tie classes by consecutive-gap single linkage with quantum

$$q=2^{\lceil\log_2\max(q_0,\;\alpha\,|s|)\rceil},$$

where $s$ is the decision's scale and $\alpha$ and $q_0$ a relative factor and floor for that kind of quantity (for the shared kinds, $q_0$ is sixteen times its measured noise motion, rounded up to a power of two); some decisions use a fixed width. Within a class, order follows a secondary scientific key, then exact discrete content, never a floating value; classes wider than $4q$ keep value order. An improvement must exceed $q$, and threshold rules use the band $[t-q,\,t+q]$ around threshold $t$. An explicit reversal setting tests whether an arbitrary content convention moves a scientific claim. Supplementary Table S2 records the declared arithmetic forms and reduction orders.

### Checkpoints and resume

Each scientific product is published atomically to an append-only container under a derivation identity binding consumed source content (not paths or timestamps), upstream identities, parameters, tie settings, code closure, codecs and numerical policy, but not resources or output paths. Resume reuses a product only under a matching identity and refuses a mismatch rather than overwriting it; completion markers never authorise reuse.

### Complexity in the number of founders

The design targets one pipeline usable up to $K=30$ founder haplotypes (documented envelope: $N\le2{,}500$ samples on bounded crops), with polynomial work and cubic or lower dependence on $K$ where possible. Named budgets bound potentially exponential candidate families. Three cost exemptions retain rare context-coded local-selection screening, $O(K^5)$ per round (0); bounded discovery and selection searches, including discovery fitting $O(NLK^4)$ and the selection backstop $O(NLK^3(K+L))$ (0c); and optional pairwise IBD, $O(N^2L)$, off by default (0d). Founder-count searches fit $O(K)$ candidates per round, with $O(NLK^3)$ count-refit work under the quadratic fixed-$K$ model and a quartic worst-case pass. This is distinct from the remaining search costs and does not establish a cubic bound for a whole stage.

The cost register also records structural screening in joins, refinement window fixing, painting label-continuity checks and founder-refit tie resolution under exception 0e. The adaptive join-screening route has a cubic analytic bound; the direct route has a quartic query-count trend on the documented finite grid. Window fixing is bounded by its work account. Painting continuity and founder-refit ties retain above-cubic worst cases that small observed tie classes do not certify away (Supplementary Information, section S6).

### Kernel scaling ladders

$K$-dependence was also measured with kernel ladders on direct synthetic inputs, single-threaded and timed after a discarded first call: $K$ from 5 to 30 at $N=200$, and $N$ from 250 to 2,500 at $K=15$. The fitted log–log exponents $\beta_K$ and $\beta_N$ are reported with standard errors, and $\beta_K\le3.3$ is accepted as at most cubic. A slope covers only the probed kernel, not an adaptive search's worst case, which for exempt stages remains analytic. Infrastructure without an intrinsic $K$ was probed on size proxies.


![Measured kernel scaling. Points are fitted log–log slopes against founder count (K = 5–30 at
N = 200) and sample count (N = 250–2,500 at K = 15), with one-standard-error bars. The dashed
3.3 threshold applies to kernels gated as at most cubic, not adaptive searches. Exempt and
kernel-only rows retain the whole-search cost qualifications in the text. Operational size
proxies are omitted. Source: the repository cost register.](figures/fig2_scaling.svg){#fig:scaling width=100%}

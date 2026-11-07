# Configuration

See [running a cross](user/running.md), the generated [settings reference](reference/settings.md) and [migration](user/migration.md).

Resampling task grain is operational, outside the scientific configuration.
`stages.pedigree.execution.RESAMPLING_BATCH_WORK` is 250,000 alternative-row scans;
`stages.pedigree.execution.RESAMPLING_BATCHES_PER_WORKER` is 2. A batch stops at the
smaller of the work ceiling and the share that leaves two waves per worker.
A single indivisible fit can exceed the ceiling. Neither constant limits the
number of bootstrap or LOCO fits, candidates, iterations or support counts.

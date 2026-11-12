"""Operational family-phase publication parameters, independent of CPU authority."""

# Units: samples per consecutive canonical-order storage batch, for both solves together.
# Sixteen amortises durable publications while limiting repeated work within each
# incomplete storage batch. All pending samples share the full task CPU authority.
# The bound changes write granularity and partial identities; each sample's
# computation and available task CPUs are independent of it.
PHASE_RELEASE_BATCH = 16

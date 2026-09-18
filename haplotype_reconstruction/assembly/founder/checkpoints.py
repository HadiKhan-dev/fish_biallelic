"""Path-based intermediate checkpoints for final founder refinement.

The enclosing assembly store binds these paths to the exact prepared inputs and
code/configuration identity. Final release products retain their ordinary block
format. Only derived chromosome arrays are omitted from intermediate snapshots.
"""
from..import hierarchy, paths


_DERIVED_FIELDS = {
    "positions", "haplotypes", "discrete_haps",
    "missing_aware_inference_discrete_haps", "founder_alt_pseudo_probability",
    "n_directional_site_supporters", "missing_aware_atomic_source_row_paths",
    "missing_aware_atomic_position_offsets", "missing_aware_atomic_position_counts",
    "missing_aware_atomic_source_row_counts",
}


class FounderCheckpointStore:
    """Wrap an already-bound store; preserve every phase and proposal token."""

    def __init__(self, store, prepared):
        self.store = store
        self.prepared = list(prepared)
        self.starts = {int(b.positions[0]): i for i, b in enumerate(self.prepared)}
        self.ends = {int(b.positions[-1]): i + 1 for i, b in enumerate(self.prepared)}

    def _encode(self, block):
        from..founder_refinement import _local_selection
        start = self.starts[int(block.positions[0])]
        stop = self.ends[int(block.positions[-1])]
        return {
            "founder_paths": True, "start": start, "stop": stop,
            "rows": _local_selection(block, self.prepared[start:stop]),
            "keys": list(block.haplotypes),
            "haplotype_dtypes": [str(row.dtype) for row in block.haplotypes.values()],
            # Preserve phase boundaries, keep masks and any non-derived fields.
            "extras": {k: v for k, v in vars(block).items() if k not in _DERIVED_FIELDS},
        }

    def _decode(self, value):
        from..founder_refinement import _LeafKeyMap
        batch = self.prepared[value["start"]:value["stop"]]
        reconstruction = paths.reconstruct_haplotypes_from_beam(
            [(list(row), 0.) for row in value["rows"]], _LeafKeyMap(batch), batch)
        block = hierarchy.convert_reconstruction_to_superblock(reconstruction, batch)
        block.haplotypes = {key: row.astype(dtype, copy=False) for key, row, dtype in zip(
            value["keys"], block.haplotypes.values(), value["haplotype_dtypes"])}
        vars(block).update(value["extras"])
        return block

    def save(self, phase, payload):
        if isinstance(payload, dict) and "block" in payload:
            payload = dict(payload, block=self._encode(payload["block"]))
        self.store.save(phase, payload)

    def load(self, phase):
        payload = self.store.load(phase)
        if (isinstance(payload, dict) and isinstance(payload.get("block"), dict)
                and payload["block"].get("founder_paths")):
            payload = dict(payload, block=self._decode(payload["block"]))
        return payload

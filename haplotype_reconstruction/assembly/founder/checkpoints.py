"""Records of one founder refinement level (key prefix ``<assembly>/l<k>/refine``).

``<n:05d>`` holds component n's refined block as rows of the prepared blocks
(its paths), its non-derived fields and attribute names, and its diagnostic;
decoding rebuilds the derived chromosome arrays from the prepared blocks
exactly. ``<n:05d>/sweep/<fit>`` holds a long component's converged-sweep
states (converged_sweep.SweepEngine). The enclosing assembly section binds
them to the exact prepared inputs and code/configuration identity.
"""
from .. import hierarchy, paths
from ...core.haplotypes import BlockResults


_DERIVED_FIELDS = {
    "positions", "haplotypes", "discrete_haps",
    "missing_aware_inference_discrete_haps", "founder_alt_pseudo_probability",
    "n_directional_site_supporters", "missing_aware_atomic_source_row_paths",
    "missing_aware_atomic_position_offsets", "missing_aware_atomic_position_counts",
    "missing_aware_atomic_source_row_counts",
}


class RefinementCheckpoints:
    """Component records of one refinement level on its assembly's prepared blocks."""

    def __init__(self, checkpoints, prepared):
        self.checkpoints = checkpoints
        self.prepared = list(prepared)
        self.starts = {int(b.positions[0]): i for i, b in enumerate(self.prepared)}
        self.ends = {int(b.positions[-1]): i + 1 for i, b in enumerate(self.prepared)}

    def _encode(self, block):
        from ..founder_refinement import _local_selection
        start = self.starts[int(block.positions[0])]
        stop = self.ends[int(block.positions[-1])]
        return {
            "start": start, "stop": stop,
            "rows": _local_selection(block, self.prepared[start:stop]),
            "keys": list(block.haplotypes),
            "haplotype_dtypes": [str(row.dtype) for row in block.haplotypes.values()],
            # Phase boundaries, keep masks and every other non-derived field.
            "extras": {k: v for k, v in vars(block).items() if k not in _DERIVED_FIELDS},
            "names": list(vars(block)),
        }

    def _decode(self, value):
        from ..founder_refinement import _LeafKeyMap
        batch = self.prepared[value["start"]:value["stop"]]
        reconstruction = paths.reconstruct_haplotypes_from_beam(
            [(list(row), 0.) for row in value["rows"]], _LeafKeyMap(batch), batch)
        block = hierarchy.convert_reconstruction_to_superblock(reconstruction, batch)
        block.haplotypes = {key: row.astype(dtype, copy=False) for key, row, dtype in zip(
            value["keys"], block.haplotypes.values(), value["haplotype_dtypes"])}
        vars(block).update(value["extras"])
        # Fields the rebuild copied from prepared blocks the original never had.
        for name in set(vars(block)).difference(value["names"]):
            delattr(block, name)
        return block

    def completed(self):
        """Numbers of the components already recorded (one key listing)."""
        prefix = self.checkpoints.key("")
        return {int(key[len(prefix):]) for key in self.checkpoints.file.keys(prefix)
                if key[len(prefix):].isdigit()}

    def load(self, number):
        """(block, diagnostic) of component ``number``, or None when not recorded."""
        payload = self.checkpoints.load(f"{number:05d}")
        if payload is None:
            return None
        return self._decode(payload["block"]), payload["diagnostic"]

    def save(self, number, block, diagnostic):
        self.checkpoints.save(f"{number:05d}", {"block": self._encode(block),
                                                "diagnostic": diagnostic})

    def blocks(self, count):
        """The refined blocks of a completed level with ``count`` components."""
        loaded = [self.load(number) for number in range(count)]
        if any(item is None for item in loaded):
            raise RuntimeError(f"{self.checkpoints!r}: a refined component record is missing")
        return BlockResults([block for block, _diagnostic in loaded])

    def sweeps(self, number):
        """Component ``number``'s converged-sweep records (picklable; one writer thread)."""
        return self.checkpoints.scoped(f"{number:05d}/sweep", nthreads=1)

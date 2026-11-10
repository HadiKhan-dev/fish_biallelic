"""Own the founder-release codec and its independently bound input identities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ...contracts.assembly import FounderRelease
from ...contracts.payload import from_payload
from ...contracts.roles import require_schema
from ...core.errors import InputError, InvariantError
from ...store.protocol import NumericalPolicy, bound_payload_identity, read_items
from ..assembly.records import operation_derivation

RELEASE = "release"


def derivation(
    upstream: Mapping[str, str], *, reverse: bool = False, numerical_policy: NumericalPolicy | None = None
) -> str:
    """Bind all four upstream derivations and the common evidence payload identity."""
    required = {"assembly", "painting", "refit", "evidence", "payload"}
    if set(upstream) != required:
        raise InvariantError(f"release identity needs {sorted(required)}")
    return operation_derivation(
        {},
        upstream,
        reverse=reverse,
        roots=("haplotype_reconstruction.stages.release.tasks", "haplotype_reconstruction.stages.release.compute"),
        numerical_policy=numerical_policy,
    ).identity


@dataclass(frozen=True)
class Reader:
    """A founder release under the stage's expected derivation."""

    identity: str
    key: str = RELEASE
    schema: int = 1

    def decode(self, payload):
        require_schema(payload, self.schema)
        return from_payload(FounderRelease, payload)


def expected(store, manifest, *, product, contig):
    """Bind the checked assembly, painting, refit and evidence in one place."""
    from ..assembly import records as assembly_records
    from ..evidence import records as evidence_records
    from ..painting import records as painting_records
    from ..refit import records as refit_records

    if product != "release" or contig not in manifest["contigs"]:
        raise InputError("release reader requires a bound contig")
    file = store.contig(contig)
    requests = dict(
        assembly=assembly_records.result_reader(store, manifest, product="assembly", contig=contig),
        painting=painting_records.result_reader(store, manifest, product="painting", contig=contig, panels=False, emissions=False),
        refit=refit_records.result_reader(store, manifest, product="refit", contig=contig, panels=False),
        evidence=evidence_records.result_reader(store, manifest, product="evidence", contig=contig),
    )
    upstream = {}
    for name, request in requests.items():
        if request is None or read_items(file, request.key, request.identity, ("schema",)) is None:
            raise InputError("release upstream publication is incomplete")
        upstream[name] = request.identity
    upstream["payload"] = bound_payload_identity(file)
    return Reader(derivation(upstream, reverse=bool(manifest["ties"]["reverse"]),
                             numerical_policy=NumericalPolicy(**manifest["numerical_policy"])))


def result_reader(store, manifest, *, product, contig=None):
    """Read the same expected request when the release commit exists."""
    if contig is None or not store.contig(contig).has(RELEASE):
        return None
    return expected(store, manifest, product=product, contig=contig)

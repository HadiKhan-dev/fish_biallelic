"""Derivations and readers of the run-level painting calibration."""

from dataclasses import dataclass

from ...contracts.contigs import AnalysisContigs
from ...core.concurrency import threaded_each
from ...contracts.payload import from_payload
from ...core.errors import InvariantError
from ...contracts.roles import require_schema
from ...science.painting.parameters import PaintingParameters
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, read_items

KEYS = dict(
    robustness="painting/calibration/robustness",
    refine="painting/calibration/refine",
    outlier="painting/calibration/outlier",
    share="painting/calibration/share",
    divisor_fit="painting/calibration/divisor_fit",
    divisor="painting/calibration/divisor",
    calibration="painting/calibration",
)


@dataclass(frozen=True)
class Reader:
    identity: str
    key: str
    schema: int = 1
    items: tuple | None = None

    def decode(self, payload):
        require_schema(payload, self.schema)
        return dict(payload)


def fit_round(fold, divisor=None):
    return str(fold) if divisor is None else f"{fold}-{float(divisor)}"


def fit_key(fold, divisor):
    return f"{KEYS['divisor_fit']}/{fit_round(fold, divisor)}"


def expected(store, manifest, *, product="calibration", contig=None, fold=None, divisor=None):
    # Pooled over the analysed contigs (the reconstruction axis), like the pedigree.
    contigs = tuple(AnalysisContigs.from_manifest(manifest).included)
    from .records import bound_upstream

    inputs = {}
    # each contig's chain (header reads and decodes back to discovery) is independent of the others, and is bound
    # once per process (bound_upstream): every contig's painting reader derives this identity
    for c, bound in zip(contigs, threaded_each(lambda c: bound_upstream(store, manifest, c), contigs), strict=True):
        if bound is None:
            raise InvariantError("painting calibration requires every final assembly and evidence product")
        inputs[c + "/assembly"], inputs[c + "/evidence"], inputs[c + "/map"] = bound[1], bound[3], bound[4]
    params = from_payload(PaintingParameters, manifest["parameters"]["painting"])
    auto = manifest["parameters"].get("estimate_foreign_read_share", False)
    # The outlier profile's clusters are the pedigree's material units (its MaterialParameters, read as the run's
    # values, never as pedigree code): bound whenever several candidate rates are fitted. A single-value grid fixes
    # the rate, fits nothing and reads no material.
    material = manifest["parameters"]["pedigree"]["material"] if len(set(params.robustness_grid)) > 1 else None
    policy = NumericalPolicy(**manifest["numerical_policy"])
    code = code_dependency_identity(
        ("science.painting.calibration", "stages.painting.calibration_tasks", "stages.painting.calibration_records")
    )

    def identity(which, scope=None, upstream=None):
        return Derivation(
            code,
            dict(product=which, contig=scope, contigs=contigs, parameters=params, material=material,
                 estimate_foreign_read_share=auto if which == "calibration" else None),
            policy,
            dict(inputs, **(upstream or {})),
            tie_reverse=manifest["ties"]["reverse"],
            tie_keep=tuple(manifest["ties"]["keep"]),
        ).identity

    shards = {c: identity("robustness", c) for c in contigs}
    # the two midpoint rates around the pooled grid maximum (ADVICE2 14.1 item 6): every contig's grid fits decide them
    refine = {c: identity("refine", c, shards) for c in contigs}
    outlier = identity("outlier", upstream={**shards, **{c + "/refine": r for c, r in refine.items()}})
    if product == "robustness":
        result = shards[contig]
    elif product == "refine":
        result = refine[contig]
    elif product == "outlier":
        result = outlier
    else:
        upstream = dict(outlier=outlier)
        if auto and params.read_classes > 1:
            share_shards = {c + "/share": identity("share", c, upstream) for c in contigs}
            share = identity("share", upstream=dict(share_shards, **upstream))
            if product == "share":
                return Reader(share if contig is None else share_shards[contig + "/share"], KEYS["share"])
            upstream["share"] = share
        elif product == "share":
            raise InvariantError("share estimation requires auto foreign_read_share and multiple read classes")
        if product == "divisor_fit":
            key = fit_key(fold, divisor)
            return Reader(identity(key, contig, upstream), key)
        if product == "divisor":
            result = identity("divisor", contig, upstream)
        else:
            # The declared or inferred share reaches the paintings through this receipt.
            divisors = {c: identity("divisor", c, upstream) for c in contigs}
            result = identity("calibration", upstream=dict(divisors, **upstream))
    return Reader(result, KEYS[product])


def read_setup(store, manifest, contig):
    """Read the fitted library laws from the contig's checked robustness record."""
    request = expected(store, manifest, product="robustness", contig=contig)
    record = read_items(store.contig(contig), request.key, request.identity, ("read_setup",))
    return None if record is None else record.payload.get("read_setup")


def result_reader(store, manifest, *, product="calibration", contig=None):
    file = store.run if contig is None else store.contig(contig)
    if not file.has(KEYS[product]):
        return None
    request = expected(store, manifest, product=product, contig=contig)
    return request if read_items(file, request.key, request.identity, ("schema",)) is not None else None

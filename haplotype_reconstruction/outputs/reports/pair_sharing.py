"""OUTPUT/painting/pair_sharing.csv and pair_sharing_chromosomes.csv: founder sharing of every sample pair (PR20).

Written only when ``[painting] pair_sharing`` is selected. Per sample pair, the shares of the jointly named painted
length (Morgans of the working map) on which the two samples' released tracks share 0, 1 or 2 founder rows
(identity by descent relative to the panel's founder rows), and the shares of the painted length not counted, by
reason. pair_sharing.csv sums the included contigs; pair_sharing_chromosomes.csv has one row per pair and contig.
Rows follow the run's input sample order: the first sample precedes the second in it. A view of the released
paintings: nothing here is filtered, reweighted or fed back.

Per-contig collectors write private parts (the contig's rows and its float64 lengths); the run-level writer sums the
lengths and publishes both tables atomically, then removes the parts.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...contracts.pair_sharing import LENGTHS, PairSharing, contig_pair_lengths
from ...core.errors import InvariantError
from ...core.files import atomic_path
from ...results.pair_sharing import painting_calls, working_maps
from .tables import write_frame

if TYPE_CHECKING:
    from ...contracts.genetic_map import GeneticMapSet
    from ...results.reader import Results

__all__ = ["pair_frame", "contig_pair_sharing", "write_pair_sharing", "PAIR_SHARING_COLUMNS", "CHROMOSOME_COLUMNS"]
FAMILY = "painting"
GENOME_TABLE = "pair_sharing.csv"
CHROMOSOME_TABLE = "pair_sharing_chromosomes.csv"


def pair_frame(sharing: PairSharing, input_ids: Sequence[str] | None = None):
    """The rows of ``sharing`` (columns PAIR_SHARING_COLUMNS) in input order; default the records' order."""
    import pandas as pd

    names = np.asarray(sharing.sample_ids, dtype=object)
    position = np.arange(len(names))
    if input_ids is not None:
        rank = {str(name): index for index, name in enumerate(input_ids)}
        position = np.asarray([rank.get(str(name), len(rank) + index) for index, name in enumerate(names)])
    pairs = sharing.pairs()
    swap = position[pairs[:, 0]] > position[pairs[:, 1]]
    first = np.where(swap, pairs[:, 1], pairs[:, 0])
    second = np.where(swap, pairs[:, 0], pairs[:, 1])
    order = np.lexsort((position[second], position[first]))
    shares, coverage = sharing.shares()[order], sharing.coverage()[order]
    return pd.DataFrame(
        dict(
            SampleA=names[first[order]],
            SampleB=names[second[order]],
            IBD0=shares[:, 0],
            IBD1=shares[:, 1],
            IBD2=shares[:, 2],
            NamedMorgans=sharing.named_morgans()[order],
            Coverage=coverage[:, 0],
            Candidates=coverage[:, 1],
            Unidentified=coverage[:, 2],
            Abstained=coverage[:, 3],
        ),
        columns=list(PAIR_SHARING_COLUMNS),
    )


def contig_pair_sharing(
    output: str | Path, results: Results, contig: str, *, genetic_maps: GeneticMapSet
) -> tuple[str, dict[str, Path]]:
    """Publish one contig's private parts: its rows and its per-pair lengths on the run's working map (from the
    configured ``genetic_maps`` and, with ``infer_rate``, the relatedness decision the contig's painting waited for)."""
    painting = painting_calls(results, contig)
    if painting is None or painting.sample_ids != tuple(results.manifest["sample_ids"]):
        raise InvariantError(f"{contig}: pair sharing needs the contig's published painting on the run's sample axis")
    lengths = contig_pair_lengths(painting, working_maps(results, genetic_maps).for_contig(contig))
    frame = pair_frame(PairSharing(painting.sample_ids, (contig,), lengths), results.input_sample_ids)
    frame.insert(0, "contig", contig)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=".pair_sharing-", dir=output))
    try:
        np.save(directory / "lengths.npy", lengths)
        rows = write_frame(directory / CHROMOSOME_TABLE, frame)
    except BaseException:
        shutil.rmtree(directory)
        raise
    return contig, {"lengths": directory / "lengths.npy", "rows": rows}


def write_pair_sharing(
    output: str | Path, results: Results, parts: Sequence[tuple[str, dict[str, Path]]]
) -> list[Path]:
    """Sum the parts' lengths in the supplied contig order and publish both tables; the parts are removed."""
    directory = Path(output) / FAMILY
    directory.mkdir(parents=True, exist_ok=True)
    try:
        sample_ids = tuple(results.manifest["sample_ids"])
        total = np.zeros((len(sample_ids) * (len(sample_ids) - 1) // 2, len(LENGTHS)), dtype=np.float64)
        for _, paths in parts:
            total += np.load(paths["lengths"])
        if not parts:
            raise InvariantError("pair sharing has no painted contig")
        genome = pair_frame(PairSharing(sample_ids, tuple(c for c, _ in parts), total), results.input_sample_ids)
        written = [write_frame(directory / GENOME_TABLE, genome)]
        # The parts share one header (this module wrote them): keep the first, append the others' rows unchanged.
        path = directory / CHROMOSOME_TABLE
        with atomic_path(path) as temporary, temporary.open("wb") as target:
            for index, (_, paths) in enumerate(parts):
                with Path(paths["rows"]).open("rb") as source:
                    header = source.readline()
                    if index == 0:
                        target.write(header)
                    shutil.copyfileobj(source, target)
        written.append(path)
    finally:
        for parent in {Path(path).parent for _, paths in parts for path in paths.values()}:
            shutil.rmtree(parent, ignore_errors=True)
    return written


PAIR_SHARING_COLUMNS = (
    "SampleA",
    "SampleB",
    "IBD0",
    "IBD1",
    "IBD2",
    "NamedMorgans",
    "Coverage",
    "Candidates",
    "Unidentified",
    "Abstained",
)
"""Columns of pair_sharing.csv in published order (one row per sample pair, over the included contigs).

    SampleA: first sample identifier, earlier in the input sample order (text).
    SampleB: second sample identifier (text).
    IBD0: share of the jointly named length on which the two samples' released tracks share no founder row; empty when that length is zero (1).
    IBD1: share of the jointly named length on which they share one founder row (1).
    IBD2: share of the jointly named length on which they share two founder rows (1).
    NamedMorgans: jointly named length, painted bins where all four tracks name one founder row, on the working map (Morgans).
    Coverage: jointly named share of the painted length (1).
    Candidates: share of the painted length where a track names candidate rows (pooled or local allele equivalence, or a switch interval) and none is unidentified or abstains (1).
    Unidentified: share of the painted length where a track is background or an unanchored class and none abstains (1).
    Abstained: share of the painted length where a track abstains (no evidence or low posterior); Coverage, Candidates, Unidentified and Abstained sum to 1 (1).
"""

CHROMOSOME_COLUMNS = (
    "contig",
    "SampleA",
    "SampleB",
    "IBD0",
    "IBD1",
    "IBD2",
    "NamedMorgans",
    "Coverage",
    "Candidates",
    "Unidentified",
    "Abstained",
)
"""Columns of pair_sharing_chromosomes.csv in published order (one row per contig and sample pair).

    contig: contig identifier (text).
    SampleA: first sample identifier, earlier in the input sample order (text).
    SampleB: second sample identifier (text).
    IBD0: share of the contig's jointly named length on which the pair shares no founder row; empty when that length is zero (1).
    IBD1: share on which the pair shares one founder row (1).
    IBD2: share on which the pair shares two founder rows (1).
    NamedMorgans: the contig's jointly named length on the working map (Morgans).
    Coverage: jointly named share of the contig's painted length (1).
    Candidates: share of the painted length where a track names candidate rows and none is unidentified or abstains (1).
    Unidentified: share of the painted length where a track is background or an unanchored class and none abstains (1).
    Abstained: share of the painted length where a track abstains; the four shares sum to 1 (1).
"""

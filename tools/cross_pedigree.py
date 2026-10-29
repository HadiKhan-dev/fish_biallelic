"""Write an explicit pedigree table for a simulated two-generation cross ([simulation] pedigree).

Design (every count is an option): G0 founders are `dams` females and `sires` males, all sequenced. F1: each dam
mated with each sire, `f1_per_dam` children per dam (sires used in turn), sexes drawn at random; `f1_observed` of
them are sequenced (half female, half male where possible, the rest from the other sex). F2: `f2` children of F1 x F1 matings (one female and one
male F1), drawn from `pairs` mating pairs whose use follows a Zipf weight (pair rank ** -pair_exponent), so families
range from one child to many and some F1 parents mate with several partners; F2 sexes are drawn at random and every
F2 is sequenced. Most F2 parents are then unsequenced F1, and the sequenced F1 are full or half sibs of them.

Usage: python tools/cross_pedigree.py OUT.tsv [--dams 2] [--sires 1] [--f1-per-dam 16,18] [--f1-observed 16]
           [--f2 96] [--pairs 30] [--pair-exponent 1.2] [--seed 217]
Founder rows come first, in the order dams then sires; the simulation templates must hold that many founders.
"""
import argparse
import random
from pathlib import Path


def build(dams, sires, f1_per_dam, f1_observed, f2, pairs, pair_exponent, seed):
    rng = random.Random(seed)
    rows = []
    dam_ids = [f"G0_D{i + 1}" for i in range(dams)]
    sire_ids = [f"G0_S{i + 1}" for i in range(sires)]
    for name in dam_ids:
        rows.append(dict(sample=name, parent1="", parent2="", sex="F", generation="G0", observed=1))
    for name in sire_ids:
        rows.append(dict(sample=name, parent1="", parent2="", sex="M", generation="G0", observed=1))
    f1 = []
    for d, (dam, count) in enumerate(zip(dam_ids, f1_per_dam)):
        for i in range(count):
            sire = sire_ids[i % len(sire_ids)]
            f1.append(dict(sample=f"F1_D{d + 1}_{i + 1}", parent1=dam, parent2=sire, sex=rng.choice("FM"),
                           generation="F1", observed=0))
    females = [row for row in f1 if row["sex"] == "F"]
    males = [row for row in f1 if row["sex"] == "M"]
    if not females or not males:
        raise ValueError("the F1 need both sexes")
    if f1_observed > len(f1):
        raise ValueError(f"{f1_observed} sequenced F1 requested but only {len(f1)} F1")
    if f2 and not pairs:
        raise ValueError("F2 children need at least one mating pair")
    if pairs > len(females) * len(males):
        raise ValueError(f"{pairs} distinct mating pairs requested but the F1 allow only {len(females) * len(males)}")
    half = f1_observed // 2
    chosen = rng.sample(females, min(half, len(females))) + rng.sample(males, min(f1_observed - half, len(males)))
    if len(chosen) < f1_observed:
        # One sex has too few F1 for an even split: fill the deficit from the other sex's remaining F1.
        remaining = [row for row in f1 if row not in chosen]
        chosen += rng.sample(remaining, f1_observed - len(chosen))
    for row in chosen:
        row["observed"] = 1
    rows.extend(f1)
    matings = []
    while len(matings) < pairs:
        pair = (rng.choice(females)["sample"], rng.choice(males)["sample"])
        if pair not in matings:
            matings.append(pair)
    weights = [(rank + 1) ** -pair_exponent for rank in range(len(matings))]
    for i in range(f2):
        dam, sire = rng.choices(matings, weights=weights)[0]
        rows.append(dict(sample=f"F2_{i + 1}", parent1=dam, parent2=sire, sex=rng.choice("FM"), generation="F2",
                         observed=1))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--dams", type=int, default=2)
    parser.add_argument("--sires", type=int, default=1)
    parser.add_argument("--f1-per-dam", default="16,18")
    parser.add_argument("--f1-observed", type=int, default=16)
    parser.add_argument("--f2", type=int, default=96)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--pair-exponent", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=217)
    args = parser.parse_args()
    per_dam = [int(x) for x in args.f1_per_dam.split(",")]
    if len(per_dam) != args.dams:
        raise SystemExit("--f1-per-dam needs one count per dam")
    rows = build(args.dams, args.sires, per_dam, args.f1_observed, args.f2, args.pairs, args.pair_exponent, args.seed)
    columns = ("sample", "parent1", "parent2", "sex", "generation", "observed")
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text("\t".join(columns) + "\n"
                         + "".join("\t".join(str(row[c]) for c in columns) + "\n" for row in rows))
    temporary.replace(args.out)
    observed = sum(row["observed"] for row in rows)
    print(f"{args.out}: {len(rows)} individuals, {observed} sequenced")


if __name__ == "__main__":
    main()

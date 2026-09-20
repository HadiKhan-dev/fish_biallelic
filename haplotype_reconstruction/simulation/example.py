"""Generate a portable, explicitly synthetic end-to-end example."""
from pathlib import Path
import json
import numpy as np


def create_example(destination, *, seed=400, contigs=3, sites=1200, length_bp=12_000_000):
    """Independent Bernoulli founder SNPs: a wiring example, not a biological benchmark."""
    if contigs < 1 or sites < 400 or length_bp < sites:
        raise ValueError("example needs positive contigs, at least 400 sites and length >= sites")
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("example destination must be new or empty")
    templates = destination/"templates"
    templates.mkdir(parents=True, exist_ok=True)
    names = [f"chr{i+1}" for i in range(contigs)]
    sequences = np.random.SeedSequence(seed).spawn(contigs)
    for name, child in zip(names, sequences):
        rng = np.random.default_rng(child)
        alleles = rng.integers(0, 2, (6, sites), dtype=np.int8)
        positions = np.linspace(1, length_bp, sites, dtype=np.int64)
        np.savez_compressed(templates/(name+".npz"), positions=positions,
                            allele_probabilities=np.eye(2, dtype=float)[alleles])
    config = (
        '[run]\ncores = "auto"\noutput = '+json.dumps(str(destination/"run"))+'\n\n'
        '[inputs]\ntemplates = '+json.dumps(str(templates))+'\ncontigs = '+json.dumps(names)+'\n\n'
        '[simulation]\nseed = '+str(seed)+'\ndepth = 5.0\ngenerations = [20, 30, 30]\n\n'
        '[recombination]\nrate_cm_per_mb = 5.0\nshared_family_evidence = true\n')
    (destination/"simulation.toml").write_text(config)
    (destination/"README.txt").write_text(
        "Synthetic six-founder Bernoulli marker templates. Not a reference genome, "
        "empirical diversity model, or calibrated pedigree accuracy benchmark.\n"
        "The default three short chromosomes are an end-to-end software example.\n"
        "Run: python run.py simulate --config "+str(destination/"simulation.toml")+"\n"
        "Then: python run.py evaluate --output "+str(destination/"run")+"\n")
    return destination/"simulation.toml"

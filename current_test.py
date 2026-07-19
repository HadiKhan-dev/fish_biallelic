#!/usr/bin/env python3
"""
screen_g0_reads.py

Read-level screen of G0_1's deep CRAM over the problem window vs a control
region, to tell a REAL divergent haplotype from a mapping artifact -- using the
one carrier with high coverage (~50x), so the shallow-F1 noise is irrelevant.

What to look for over the window, relative to the control region:
  * per-read mismatch count (NM) goes BIMODAL -- a clean ~0-1 mode (the common
    haplotype) plus a high mode (~4-10, the divergent haplotype) -> X is a real
    divergent haplotype, and you can read it straight off G0_1.
  * NM uniformly low like the control -> no divergent reads; the het calls are
    not coming from a divergent haplotype (points to a genotyping quirk).
  * MAPQ collapses toward 0 -> reads multi-map (repeat/paralog), though normal
    depth already argues against extra copies.

Needs samtools in PATH and the reference FASTA the CRAM was aligned to.
Edit CONFIG, then:  python screen_g0_reads.py
(Untested here -- no CRAM/samtools in the sandbox; the parsing is straightforward.)
"""
import subprocess, re, sys
import numpy as np

# =============================== CONFIG ===============================
SAMTOOLS  = "samtools"
CRAM      = "/rds/project/rds-8b3VcZwY7rY/projects/cichlid/alignments/DURB_000005-LabCichlids/X_TmAc/Astcal_F0_2/fAstCal1.2/2021cicX11218978.mem.crumble.cram"  # G0_1
REFERENCE = "/path/to/fAstCal1.2.fa"          # <-- SET THIS (the .fa the CRAM was aligned to)
WINDOW    = "chr12:31275584-31279061"          # block 1013
CONTROL   = ["chr12:5000000-5100000", "chr12:20000000-20100000"]  # normal regions, same sample
DIV_NM    = 4                                  # reads with NM >= this = "divergent"
OUT_PNG   = "g0_1_block1013_readscreen.png"
OUT_TXT   = "g0_1_block1013_readscreen.txt"
# =====================================================================

class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for f in self.s: f.write(x)
    def flush(self):
        for f in self.s: f.flush()
_logf = open(OUT_TXT, "w", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _logf)

_nm = re.compile(r"NM:i:(\d+)")

def collect(region):
    """primary alignments only (-F 0x904). returns mapq[], nm[], rate[]."""
    cmd = [SAMTOOLS, "view", "-F", "0x904", "-T", REFERENCE, CRAM, region]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    mapq, nm, rate = [], [], []
    for line in proc.stdout:
        f = line.split("\t")
        if len(f) < 11:
            continue
        try:
            q = int(f[4])
        except ValueError:
            continue
        mapq.append(q)
        m = _nm.search(line)
        if m:
            v = int(m.group(1)); L = len(f[9]) if f[9] != "*" else 0
            nm.append(v)
            if L > 0:
                rate.append(v / L)
    _, err = proc.communicate()
    if proc.returncode != 0:
        print(f"  ** samtools failed for {region}:\n{err.strip()[:400]}")
    return np.array(mapq), np.array(nm), np.array(rate)

def summarize(tag, mapq, nm, rate):
    if mapq.size == 0:
        print(f"{tag}: no reads"); return
    print(f"{tag}: {mapq.size} reads")
    print(f"   MAPQ   median {np.median(mapq):.0f} | %MAPQ=0 {100*np.mean(mapq==0):.1f}% | %MAPQ<20 {100*np.mean(mapq<20):.1f}%")
    if nm.size:
        print(f"   NM     median {np.median(nm):.0f} | mean {nm.mean():.2f} | %reads NM>={DIV_NM} {100*np.mean(nm>=DIV_NM):.1f}%")
    if rate.size:
        print(f"   mism%  mean {100*rate.mean():.2f}% per bp")

def nm_hist(nm, width=50, top=20):
    if nm.size == 0:
        return
    cap = min(top, int(nm.max()))
    counts = np.bincount(np.clip(nm, 0, cap), minlength=cap + 1)
    mx = counts.max() or 1
    print("   NM histogram (mismatches per read; look for two humps):")
    for k in range(cap + 1):
        bar = "#" * int(round(width * counts[k] / mx))
        lab = f"{k}" if k < cap else f"{k}+"
        print(f"     {lab:>3} | {bar} {counts[k]}")

print(f"=== read-level screen of G0_1 over {WINDOW} ===")
print(f"CRAM: {CRAM}")
wq, wn, wr = collect(WINDOW)
print("\n-- WINDOW --")
summarize("window", wq, wn, wr)
nm_hist(wn)

# baseline = pooled control regions
cq, cn, cr = [np.array([], int)] * 3
cq, cn, cr = (np.concatenate(x) if x else np.array([]) for x in zip(*[collect(r) for r in CONTROL]) ) if CONTROL else (cq, cn, cr)
print("\n-- CONTROL (pooled) --")
summarize("control", cq, cn, cr)

print("\nread:")
if wn.size and cn.size:
    fdiv_w = 100 * np.mean(wn >= DIV_NM); fdiv_c = 100 * np.mean(cn >= DIV_NM)
    fclean_w = 100 * np.mean(wn <= 1)
    print(f"  divergent reads (NM>={DIV_NM}): window {fdiv_w:.1f}% vs control {fdiv_c:.1f}%")
    print(f"  clean reads (NM<=1): window {fclean_w:.1f}%")
    if fdiv_w > fdiv_c + 15 and fclean_w >= 25 and fdiv_w >= 25:
        print("  -> window has BOTH a clean mode and a divergent mode = REAL divergent 2nd haplotype (het).")
    elif fdiv_w > fdiv_c + 15:
        print("  -> window enriched for divergent reads (check the histogram: uniform shift vs two humps).")
    else:
        print("  -> window looks like control: no divergent-haplotype read signal.")
    if wq.size and 100 * np.mean(wq == 0) > 20:
        print("  -> NOTE: many MAPQ=0 reads -> multi-mapping/repeat contribution.")

# optional plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if wn.size or cn.size:
        cap = int(max(wn.max() if wn.size else 0, cn.max() if cn.size else 0, 1))
        cap = min(cap, 20); bins = np.arange(cap + 2) - 0.5
        plt.figure(figsize=(7, 4))
        if cn.size: plt.hist(np.clip(cn, 0, cap), bins=bins, density=True, alpha=0.5, label="control")
        if wn.size: plt.hist(np.clip(wn, 0, cap), bins=bins, density=True, alpha=0.5, label="window")
        plt.xlabel("per-read mismatches (NM)"); plt.ylabel("density")
        plt.title("G0_1 reads: window vs control"); plt.legend(); plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=110); print(f"\nsaved -> {OUT_PNG}")
except Exception as e:
    print(f"(plot skipped: {e})")

_logf.flush(); sys.stdout = sys.__stdout__; _logf.close()
print(f"full text results saved -> {OUT_TXT}  (upload this)")
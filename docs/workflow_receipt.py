#!/usr/bin/env python3
"""Project and validate the evidence that the documented Snakemake example runs.

The integration test `WORKFLOW_TEST` runs every rule of `docs/examples/Snakefile`
as its exact shell command on the synthetic example and compares the published
products with a single `run`. It writes a raw receipt into its temporary
directory. This tool turns that raw receipt, plus proof that the test passed (a
JUnit XML report, or an `outcome` field the test writes after its final
assertion), into the portable `docs/examples/workflow_receipt.json`: temporary
paths become the placeholders `CONFIG` and `CHECKPOINTS`.

`validate` is the single rule that the documentation check applies. A receipt
is current only for the Snakefile and package sources whose hashes it records.

Usage: python docs/workflow_receipt.py RAW_RECEIPT [--junit REPORT.xml] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from haplotype_reconstruction.core.files import atomic_path  # noqa: E402
SNAKEFILE = ROOT / "docs/examples/Snakefile"
RECEIPT = ROOT / "docs/examples/workflow_receipt.json"
WORKFLOW_TEST = "tests/cli/test_workflow_mode.py::test_shell_workflow_equals_run_record_for_record"
MODE = "exact-shell-equivalent"
RULES = ("init", "evidence", "relatedness", "reconstruct", "pedigree", "phase", "report")
CONTIG_RULES = frozenset({"evidence", "reconstruct", "phase"})
PLACEHOLDERS = {"{CONFIG:q}": "CONFIG", "{CHECKPOINTS:q}": "CHECKPOINTS"}


def rule_commands(text: str) -> dict[str, str]:
    """Each rule's shell template, parsed exactly as the integration test parses it."""
    commands = {}
    for name, body in re.findall(r"^rule (\w+):\n(.*?)(?=^rule |\Z)", text, re.M | re.S):
        shell = re.search(r'shell: "([^"\n]+)"', body)
        if shell is not None:
            commands[name] = shell.group(1)
    return commands


def candidate_sha256(root: Path = ROOT) -> str:
    """SHA-256 of the package's sorted `.py` sources, concatenated (the test's candidate identity)."""
    paths = sorted((root / "haplotype_reconstruction").rglob("*.py"))
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


def snakefile_sha256(path: Path = SNAKEFILE) -> str:
    return hashlib.sha256(path.read_text().encode()).hexdigest()


def template_words(template: str) -> list[str]:
    words = template.split()
    return [PLACEHOLDERS.get(word, word) for word in words]


def matches(words: list[str], template: str) -> str | None:
    """The contig a receipt command was run for ('' for genome-wide rules), or None if it is not the template."""
    expected = template_words(template)
    if len(words) != len(expected):
        return None
    contig = ""
    for word, pattern in zip(words, expected, strict=True):
        if pattern == "{threads}":
            if not word.isdigit() or int(word) < 1:
                return None
        elif pattern == "{wildcards.c:q}":
            contig = word
        elif word != pattern:
            return None
    return contig


def validate(receipt: dict, snakefile: Path = SNAKEFILE, root: Path = ROOT) -> list[str]:
    """Every reason this receipt does not prove that the current Snakefile runs on the current sources."""
    problems = []
    if receipt.get("mode") != MODE:
        problems.append(f"mode is {receipt.get('mode')!r}, not {MODE!r}")
    if receipt.get("test") != WORKFLOW_TEST:
        problems.append(f"test is {receipt.get('test')!r}, not {WORKFLOW_TEST}")
    if receipt.get("outcome") != "passed":
        problems.append(f"test outcome is {receipt.get('outcome')!r}, not 'passed'")
    if receipt.get("snakefile_sha256") != snakefile_sha256(snakefile):
        problems.append("stale: the Snakefile changed after the receipt was recorded")
    if receipt.get("candidate_sha256") != candidate_sha256(root):
        problems.append("stale: the package sources differ from the candidate the receipt names")
    templates = rule_commands(snakefile.read_text())
    if tuple(templates) != RULES:
        problems.append(f"the Snakefile's rules are {tuple(templates)}, not {RULES}")
        return problems
    commands = receipt.get("commands")
    if not isinstance(commands, list) or not commands:
        return [*problems, "no commands recorded"]
    seen: dict[str, list[str]] = {rule: [] for rule in RULES}
    position = 0
    for words in commands:
        if not isinstance(words, list) or len(words) < 2 or words[0] != "haplotypes" or words[1] not in templates:
            problems.append(f"not a documented workflow command: {words!r}")
            continue
        rule = words[1]
        if RULES.index(rule) < position:
            problems.append(f"{rule} ran after a later workflow step")
        position = RULES.index(rule)
        contig = matches(words, templates[rule])
        if contig is None:
            problems.append(f"{' '.join(words)} is not the Snakefile's {rule} command")
            continue
        seen[rule].append(contig)
    for rule, scopes in seen.items():
        if not scopes:
            problems.append(f"rule {rule} was not run")
        elif rule not in CONTIG_RULES and len(scopes) != 1:
            problems.append(f"genome-wide rule {rule} ran {len(scopes)} times")
    contig_sets = {rule: sorted(scopes) for rule, scopes in seen.items() if rule in CONTIG_RULES}
    if len({tuple(scopes) for scopes in contig_sets.values()}) > 1:
        problems.append(f"contig rules ran for different contigs: {contig_sets}")
    for scopes in contig_sets.values():
        if len(set(scopes)) != len(scopes):
            problems.append("a contig rule ran twice for one contig")
    text = json.dumps(receipt)
    if re.search(r"/(?:rds|home|tmp|scratch)/|\b(?:CSD3|Cambridge)\b", text):
        problems.append("receipt contains a site-specific path")
    return problems


def project(raw: dict, passed: bool) -> dict:
    """The portable receipt: paths replaced by placeholders, outcome from independent pass evidence."""
    commands = []
    for words in raw.get("commands", []):
        words = list(words)
        if len(words) > 2:
            words[2] = "CONFIG"
        for index, word in enumerate(words[:-1]):
            if word == "--checkpoints":
                words[index + 1] = "CHECKPOINTS"
        commands.append(words)
    return dict(
        mode=raw.get("mode"),
        test=WORKFLOW_TEST,
        outcome="passed" if passed else raw.get("outcome", "unknown"),
        snakefile_sha256=raw.get("snakefile_sha256"),
        candidate_sha256=raw.get("candidate_sha256"),
        commands=commands,
    )


def junit_passed(path: Path) -> bool:
    """Whether the JUnit report holds the workflow test with no failure, error or skip."""
    module, _, name = WORKFLOW_TEST.partition("::")
    classname = module.removesuffix(".py").replace("/", ".")
    for case in ElementTree.parse(path).iter("testcase"):
        if case.get("name") == name and case.get("classname", "").endswith(classname.split(".", 1)[-1]):
            return not any(child.tag in ("failure", "error", "skipped") for child in case)
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("raw", type=Path, nargs="?", help="Raw workflow-source-receipt.json written by the test.")
    parser.add_argument("--junit", type=Path, help="JUnit XML report of the same pytest session.")
    parser.add_argument("--check", action="store_true", help="Validate the committed receipt only.")
    args = parser.parse_args(argv)
    if args.check:
        if not RECEIPT.exists():
            sys.stdout.write(f"{RECEIPT.relative_to(ROOT)}: missing\n")
            return 1
        problems = validate(json.loads(RECEIPT.read_text()))
    else:
        if args.raw is None:
            parser.error("RAW receipt required unless --check")
        raw = json.loads(args.raw.read_text())
        passed = raw.get("outcome") == "passed" or (args.junit is not None and junit_passed(args.junit))
        receipt = project(raw, passed)
        problems = validate(receipt)
        if not problems:
            with atomic_path(RECEIPT) as temporary:
                temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    for problem in problems:
        sys.stdout.write(f"workflow receipt: {problem}\n")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())

"""Negative controls of the documentation check: each known failure kind must fail, and its fix must pass."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check_docs = _load("check_docs", ROOT / "tools/check_docs.py")
workflow_receipt = _load("workflow_receipt", ROOT / "docs/workflow_receipt.py")


@pytest.fixture(scope="module")
def checker():
    return check_docs.Checker()


def page(lines: list[str], name: str = "docs/user/control.md") -> check_docs.Page:
    return check_docs.Page(ROOT / name, name, lines)


def fresh(checker):
    checker.errors.clear()
    checker.warnings.clear()
    checker.pending.clear()
    return checker


def numeric_parameter(checker) -> tuple[str, float]:
    for row in checker.parameter_rows:
        if row["owner"] in checker.roots:
            try:
                value = ast.literal_eval(row["expression"])
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return f"{row['owner']}.{row['name']}", value
    raise AssertionError("no numeric parameter field to cite")


def test_stale_parameter_constant_fails(checker):
    dotted, value = numeric_parameter(checker)
    fresh(checker).check_page(page([f"`{dotted} = {value * 2 + 1}`"]))
    assert any("code's default" in error for error in checker.errors), checker.errors
    fresh(checker).check_page(page([f"`{dotted} = {value}`"]))
    assert not checker.errors
    fresh(checker).check_page(page([f"`{dotted} = {value}`"], "docs/methods/control.md"))
    assert not checker.errors and checker.warnings  # methods prose names parameters, never values


def test_unknown_constant_fails(checker):
    owner = next(iter(checker.roots))
    fresh(checker).check_page(page([f"`{owner}.no_such_field = 1`"]))
    assert any("no such constant" in error for error in checker.errors)


def test_broken_anchor_fails(checker):
    fresh(checker).check_page(page(["[outputs](outputs.md#no-such-anchor)"]))
    assert any("anchor #no-such-anchor not found" in error for error in checker.errors)
    fresh(checker).check_page(page(["[outputs](outputs.md#d1-founder-vcf)", "[missing](no_such_page.md)"]))
    assert checker.errors == ["docs/user/control.md:2: link target does not exist: no_such_page.md"]


# Built at run time so that the shipped file itself holds no site-specific string (packaging portability scan).
SITE_TEXTS = ("see /" + "rds/project/data", "under /" + "home/someone/run", "on CS" + "D3", "in Cam" + "bridge")


@pytest.mark.parametrize("text", SITE_TEXTS)
def test_site_specific_path_fails(checker, text):
    fresh(checker).check_page(page([text]))
    assert any("site-specific" in error for error in checker.errors)
    fresh(checker).check_page(page(["```text", text, "```"]))
    assert any("site-specific text in code block" in error for error in checker.errors)


def test_history_phrasing_warns_outside_history_pages(checker):
    fresh(checker).check_page(page(["The model previously used another order."], "docs/methods/control.md"))
    assert checker.warnings
    fresh(checker).check_page(page(["In release 3 the fit differed."], "docs/methods/control.md"))
    assert checker.warnings
    fresh(checker).check_page(page(["The model previously used another order."], "docs/user/migration.md"))
    assert not checker.warnings


def test_unknown_command_option_and_setting_fail(checker):
    fresh(checker).check_page(page(["`haplotypes run CONFIG --no-such-option`", "`[relatedness] no_such_key = 1`"]))
    assert any("has no option --no-such-option" in error for error in checker.errors)
    assert any("unknown setting [relatedness] no_such_key" in error for error in checker.errors)
    fresh(checker).check_page(page(["`haplotypes run CONFIG --cores 4`", "`[relatedness] pairwise_ibd = true`"]))
    assert not checker.errors


def test_stale_generated_reference_fails(checker):
    fresh(checker).check_generated({"docs/reference/settings.md": "not what the generator writes\n"})
    assert checker.errors == ["docs/reference/settings.md: stale generated reference; run python docs/generate_docs.py"]


def test_coverage_omission_duplicate_and_removed_unit_fail(checker):
    coverage = json.loads((ROOT / "docs/methods/coverage.json").read_text())
    manifest = json.loads((ROOT / "docs/methods/units.json").read_text())["units"]
    fresh(checker).check_methods(coverage, manifest)
    assert not checker.errors, checker.errors[:5]
    fresh(checker).check_methods(coverage, manifest[1:])
    assert any("has 0 sections" in error for error in checker.errors)
    fresh(checker).check_methods(coverage, [*manifest, manifest[0]])
    assert any("has 2 sections" in error for error in checker.errors)
    removed = dict(manifest[0], **coverage["removed"][0])
    fresh(checker).check_methods(coverage, [*manifest, removed])
    assert any("is a removed unit" in error for error in checker.errors)
    renamed = [dict(manifest[0], parameters=["no_such_parameter_name"]), *manifest[1:]]
    fresh(checker).check_methods(coverage, renamed)
    assert any("no_such_parameter_name" in error for error in checker.errors)


def test_environment_read_outside_runtime_fails(tmp_path):
    (tmp_path / "science").mkdir()
    (tmp_path / "engine").mkdir()
    (tmp_path / "science/model.py").write_text('import os\nflag = os.environ.get("TIE_REVERSE")\n')
    (tmp_path / "science/other.py").write_text("from os import getenv\n")
    (tmp_path / "engine/runtime.py").write_text('import os\ncache = os.environ.get("NUMBA_CACHE_DIR")\n')
    found = check_docs.environment_access(tmp_path)
    assert found == ["science/model.py:2: os.environ", "science/other.py:1: os.getenv"]


def test_published_file_without_dictionary_entry_fails(tmp_path):
    reports = tmp_path / "outputs/reports"
    reports.mkdir(parents=True)
    (reports / "tables.py").write_text('names = [f"{name}.csv" for name in ()]\n')
    (reports / "extra.py").write_text('def write(contig, kind):\n    return f"{kind}.csv", f"{contig}.map.csv"\n')
    (tmp_path / "outputs/export").mkdir()
    (tmp_path / "outputs/export/run.py").write_text('stem = ".founders" ".samples" ".csi" ".tbi"\n')
    (tmp_path / "outputs/export/vcf.py").write_text("")
    names, problems = check_docs.published_names(tmp_path)
    assert "founder_support.csv" in names and "CONTIG.map.csv" in names
    assert any("'*.csv' has no DYNAMIC_OUTPUTS entry" in problem for problem in problems)
    assert any("no longer matches the writer" in problem for problem in problems)
    assert check_docs.documented_output("CONTIG.map.csv", "`recombination/CONTIG.map.csv`")
    assert not check_docs.documented_output("CONTIG.map.csv", "maps are written")


def test_undefined_evaluation_field_fails():
    dictionary = (ROOT / "docs/user/outputs.md").read_text()
    assert check_docs.undefined_evaluation_keys(dictionary) == []
    removed = dictionary.replace("`called_fraction`", "called fraction")
    assert [key for key, _ in check_docs.undefined_evaluation_keys(removed)] == ["called_fraction"]
    assert check_docs.undefined_evaluation_keys("no D7 section") != []


def test_edited_header_capture_fails(checker, tmp_path):
    source = ROOT / "docs/examples/vcf_headers.json"
    receipt = json.loads(source.read_text())
    for entry in receipt["files"]:
        (tmp_path / entry["file"]).write_bytes((source.parent / entry["file"]).read_bytes())
    (tmp_path / "vcf_headers.json").write_text(json.dumps(receipt))
    fresh(checker).check_examples(tmp_path / "vcf_headers.json")
    assert not checker.errors
    edited = tmp_path / receipt["files"][0]["file"]
    edited.write_text(edited.read_text().replace("VCFv4.2", "VCFv4.4"))
    fresh(checker).check_examples(tmp_path / "vcf_headers.json")
    assert any("recorded capture hash" in error for error in checker.errors)


def current_receipt() -> dict:
    templates = workflow_receipt.rule_commands(workflow_receipt.SNAKEFILE.read_text())
    commands = []
    for rule in workflow_receipt.RULES:
        for contig in ("chr1", "chr2") if rule in workflow_receipt.CONTIG_RULES else ("",):
            words = workflow_receipt.template_words(templates[rule])
            words = [contig if word == "{wildcards.c:q}" else "1" if word == "{threads}" else word for word in words]
            commands.append(words)
    return dict(
        mode=workflow_receipt.MODE,
        test=workflow_receipt.WORKFLOW_TEST,
        outcome="passed",
        snakefile_sha256=workflow_receipt.snakefile_sha256(),
        candidate_sha256=workflow_receipt.candidate_sha256(),
        commands=commands,
    )


def test_workflow_receipt_controls(checker, tmp_path):
    receipt = current_receipt()
    assert workflow_receipt.validate(receipt) == []
    stale = dict(receipt, candidate_sha256="0" * 64)
    assert any("package sources differ" in problem for problem in workflow_receipt.validate(stale))
    incomplete = dict(receipt, commands=[words for words in receipt["commands"] if words[1] != "pedigree"])
    assert "rule pedigree was not run" in workflow_receipt.validate(incomplete)
    unpassed = {key: value for key, value in receipt.items() if key != "outcome"}
    assert any("outcome" in problem for problem in workflow_receipt.validate(unpassed))
    leaked = dict(
        receipt, commands=[["haplotypes", "init", "/" + "home/someone/cross.toml", *receipt["commands"][0][3:]]]
    )
    assert any("site-specific" in problem for problem in workflow_receipt.validate(leaked))
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(stale))
    fresh(checker).check_workflow(path)
    assert any("package sources differ" in error for error in checker.errors)
    checker.draft = True
    try:
        fresh(checker).check_workflow(path)
        assert not checker.errors and checker.pending
    finally:
        checker.draft = False
    path.write_text(json.dumps(receipt))
    fresh(checker).check_workflow(path)
    assert not checker.errors


def test_projection_replaces_paths_and_requires_pass_evidence():
    raw = current_receipt()
    raw["commands"] = [
        [
            *words[:2],
            "/tmp/x/cross.toml",
            *("/tmp/x/run/checkpoints" if word == "CHECKPOINTS" else word for word in words[3:]),
        ]
        for words in raw["commands"]
    ]
    del raw["outcome"]
    projected = workflow_receipt.project(raw, passed=True)
    assert workflow_receipt.validate(projected) == []
    assert any("outcome" in problem for problem in workflow_receipt.validate(workflow_receipt.project(raw, False)))

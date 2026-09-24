"""CPU-only contracts for the published one-design postprocess adapter."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.headtail_geometry import audit
from analysis.linear_geometry import audit as audit_linear
from analysis.run_postprocess import run
from analysis.structure_io import _render, normalize_closed_complex, require_complex
from analysis.summarize_postprocess import summarize


def atom(serial, name, residue, chain, resid, x, y, z, element):
    return ("ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  1.00          %2s"
            % (serial, name, residue, chain, resid, x, y, z, element))


def small_complex(peptide_chain="X", receptor_chain="Y", receptor_clash=False):
    records = []
    serial = 1
    for chain, residue_count, base in ((peptide_chain, 3, 0.0),
                                       (receptor_chain, 1, 0.0 if receptor_clash else 30.0)):
        for resid in range(1, residue_count + 1):
            for name, dx, element in (("N", 0.0, "N"), ("CA", 1.4, "C"),
                                      ("C", 2.8, "C"), ("O", 3.4, "O")):
                records.append(atom(serial, name, "ALA", chain, resid,
                                    base + resid * 3.8 + dx, 0.0, 0.0, element))
                serial += 1
        records.append("TER")
    records.extend(["CONECT    3   13", "END", ""])
    return "\n".join(records)


class StructureTests(unittest.TestCase):
    def test_closed_normalization_preserves_conect_and_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "in.pdb", Path(tmp) / "out.pdb"
            source.write_text(small_complex())
            before = source.read_bytes()
            normalize_closed_complex(source, "X", "Y", output)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(require_complex(output), (3, 1))
            self.assertIn("CONECT    3   13", output.read_text())
            with self.assertRaises(ValueError):
                normalize_closed_complex(source, "X", "Y", output)

    def test_render_renumbers_each_chain(self):
        source = small_complex().splitlines()
        peptide = [line.replace("X   1", "X  41") for line in source
                   if line.startswith("ATOM") and line[21] == "X"]
        receptor = [line for line in source if line.startswith("ATOM") and line[21] == "Y"]
        rendered = _render(peptide, receptor)
        self.assertIn(" A   1 ", rendered)
        self.assertIn(" C   1 ", rendered)

    def test_geometry_detects_broken_closure_and_clash(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "in.pdb", Path(tmp) / "out.pdb"
            source.write_text(small_complex(receptor_clash=True))
            normalize_closed_complex(source, "X", "Y", output)
            result = audit(output)
            self.assertIn(3, result["bad_cn_links"])
            self.assertGreater(result["interchain_heavy_pairs_below_2A"], 0)
            self.assertFalse(result["geometry_ok"])

    def test_geometry_detects_broken_internal_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "in.pdb", Path(tmp) / "out.pdb"
            lines = small_complex().splitlines()
            # Move peptide residue 2 out of peptide-bond range, leaving all atoms.
            moved = []
            for line in lines:
                if line.startswith("ATOM") and line[21] == "X" and line[22:26].strip() == "2":
                    line = line[:30] + "%8.3f" % (float(line[30:38]) + 10.0) + line[38:]
                moved.append(line)
            source.write_text("\n".join(moved) + "\n")
            normalize_closed_complex(source, "X", "Y", output)
            self.assertIn(1, audit(output)["bad_cn_links"])
            self.assertIn(1, audit_linear(output)["bad_cn_links"])


class WorkflowTests(unittest.TestCase):
    def test_rejects_missing_explicit_original_and_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.pdb"
            source.write_text(small_complex())
            options = argparse.Namespace(topology="linear", input_state="backbone",
                                         input_pdb=source, original_pdb=None,
                                         binder_chain="X", receptor_chain="Y",
                                         output_dir=Path(tmp) / "result", seed=1,
                                         score_only=False)
            self.assertEqual(run(options), 1)
            self.assertIn("--original-pdb", (options.output_dir / "status.json").read_text())
            options.output_dir = Path(tmp) / "invalid_score"
            options.input_state = "closed-complex"
            options.score_only = True
            self.assertEqual(run(options), 1)
            options.score_only = False
            with self.assertRaises(FileExistsError):
                run(options)

    def test_summarizer_keeps_failures_and_missing_statuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rep001").mkdir()
            (root / "rep002").mkdir()
            (root / "rep002" / "status.json").write_text(json.dumps(
                {"status": "failed", "stage": "amber", "reason": "test failure"}))
            rows = summarize(root, root / "out.tsv")
            self.assertEqual([row["status"] for row in rows], ["missing", "failed"])
            self.assertEqual(len((root / "out.tsv").read_text().splitlines()), 3)

    def test_stage_failure_returns_nonzero_and_records_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.pdb"
            source.write_text(small_complex())
            output = Path(tmp) / "result"
            options = argparse.Namespace(topology="headtail", input_state="closed-complex",
                                         input_pdb=source, original_pdb=None,
                                         binder_chain="X", receptor_chain="Y",
                                         output_dir=output, seed=1, score_only=False)
            with patch("analysis.headtail_geometry.audit", return_value={"geometry_ok": True}), \
                 patch("analysis.run_postprocess.stage", side_effect=RuntimeError("amber failed")):
                self.assertEqual(run(options), 1)
            record = json.loads((output / "status.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["stage"], "headtail_amber")
            self.assertIn("amber failed", record["reason"])


if __name__ == "__main__":
    unittest.main()

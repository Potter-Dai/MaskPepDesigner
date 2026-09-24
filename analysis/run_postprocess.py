"""One-design linear or head-to-tail UFcpdiff postprocessing entry point.

Run from the repository root with ``python -m analysis.run_postprocess``.
The older top-level ``run_postprocess.py`` is intentionally unchanged.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from analysis.structure_io import (normalize_closed_complex, peptide_sequence,
                                   reconstruct_backbone, require_complex)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(module, input_pdb, directory, seed):
    directory.mkdir()
    command = [sys.executable, "-m", "analysis." + module,
               "--input-pdb", str(input_pdb), "--output-dir", str(directory)]
    if module != "headtail_amber":
        command += ["--seed", str(seed)]
    result = subprocess.run(command, text=True, capture_output=True)
    status_name = {"headtail_close": "status.json", "headtail_amber": "amber_status.json",
                   "headtail_cart": "cart_status.json", "headtail_score": "status.json"}[module]
    status = directory / status_name
    if not status.is_file():
        raise RuntimeError("%s produced no status: %s" % (module, result.stderr[-1000:]))
    data = json.loads(status.read_text())
    if result.returncode or data.get("status") != "ok":
        raise RuntimeError("%s failed: %s" % (module, data.get("reason", result.stderr[-1000:])))
    return data


def run(args):
    source = args.input_pdb.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    report = {"status": "failed", "stage": "input", "topology": args.topology,
              "input_state": args.input_state, "input_pdb": str(source),
              "seed": args.seed,
              "python_version": sys.version.split()[0], "stages": {}}
    status_file = output / "status.json"
    try:
        if not source.is_file():
            raise FileNotFoundError(source)
        report["input_sha256"] = sha256(source)
        if args.input_state == "backbone" and not args.original_pdb:
            raise ValueError("--original-pdb is required for backbone input")
        if args.score_only and not (args.topology == "headtail" and
                                    args.input_state == "closed-complex"):
            raise ValueError("--score-only requires headtail closed-complex input")
        if args.topology == "linear" and args.input_state == "closed-complex":
            raise ValueError("closed-complex input requires headtail topology")
        prepared = output / "prepared_complex.pdb"
        report["stage"] = "prepare"
        if args.input_state == "backbone":
            report["original_pdb"] = str(args.original_pdb.resolve())
            report["original_sha256"] = sha256(args.original_pdb)
            reconstruct_backbone(source, args.original_pdb, args.binder_chain,
                                 args.receptor_chain, prepared)
        else:
            normalize_closed_complex(source, args.binder_chain,
                                     args.receptor_chain, prepared)
        n, receptor_length = require_complex(prepared)
        report["peptide_length"] = n
        report["receptor_length"] = receptor_length
        report["prepared_pdb"] = str(prepared)
        report["sequence_before"] = peptide_sequence(prepared)
        if args.topology == "linear":
            report["stage"] = "linear_protocol"
            from analysis.linear_protocol import run as run_linear
            result = run_linear(prepared, output, args.seed)
            report["stages"]["linear_protocol"] = result
            report["stages"].update(result["stages"])
            report.update({key: result[key] for key in
                           ("final_pdb", "dG_separated", "dG_cross", "dSASA_int",
                            "sequence_before", "sequence_after", "pyrosetta_version")})
            report["protocol"] = result["protocol"]
            report["geometry"] = result["geometry"]
        else:
            from analysis.headtail_geometry import audit
            if args.input_state == "closed-complex":
                report["stage"] = "validate_closed_input"
                geometry = audit(prepared, n)
                report["closed_input_geometry"] = geometry
                if not geometry["geometry_ok"]:
                    raise ValueError("closed input failed full cyclic geometry audit")
                current = prepared
            else:
                report["stage"] = "headtail_close"
                item = stage("headtail_close", prepared, output / "close", args.seed)
                report["stages"]["close"] = item
                current = Path(item["repacked_pdb"])
            if not args.score_only:
                report["stage"] = "headtail_amber"
                item = stage("headtail_amber", current, output / "amber", args.seed)
                report["stages"]["amber"] = item
                current = Path(item["output_pdb"])
                report["stage"] = "headtail_cart"
                item = stage("headtail_cart", current, output / "cart", args.seed + 1)
                report["stages"]["cart"] = item
                current = Path(item["output_pdb"])
            report["stage"] = "headtail_score"
            item = stage("headtail_score", current, output / "score", args.seed + 2)
            report["stages"]["score"] = item
            report["final_pdb"] = str(current)
            report["protocol"] = "headtail_ref2015_interface_analyzer"
            report["pyrosetta_version"] = item.get("pyrosetta_version")
            report["sequence_after"] = item["binder_sequence"]
            if report["sequence_before"] != report["sequence_after"]:
                raise ValueError("headtail protocol changed peptide sequence")
            for key in ("dG_separated", "dG_cross", "dSASA_int"):
                report[key] = item[key]
            report["geometry"] = audit(current, n)
        if not Path(report["final_pdb"]).is_file():
            raise RuntimeError("final PDB is missing")
        report["final_sha256"] = sha256(report["final_pdb"])
        report["units"] = {"dG_separated": "R.E.U.", "dG_cross": "R.E.U.",
                           "dSASA_int": "A^2"}
        report["status"] = "ok"
        report["stage"] = "complete"
    except Exception as exc:
        report["reason"] = "%s: %s" % (type(exc).__name__, exc)
        linear_status = output / "linear_status.json"
        if linear_status.is_file():
            detail = json.loads(linear_status.read_text())
            report["stages"].update(detail.get("stages", {}))
            if detail.get("status") == "failed":
                report["stage"] = "linear_" + detail.get("stage", "unknown")
    status_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=("linear", "headtail"), required=True)
    parser.add_argument("--input-state", choices=("backbone", "closed-complex"), required=True)
    parser.add_argument("--input-pdb", type=Path, required=True)
    parser.add_argument("--original-pdb", type=Path)
    parser.add_argument("--binder-chain", required=True)
    parser.add_argument("--receptor-chain", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

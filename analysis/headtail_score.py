#!/usr/bin/env python3
"""Score an already-cyclized binder/receptor complex without rebuilding it."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from analysis.headtail_close import rmsd, terminal_geometry
from analysis.headtail_geometry import audit
from analysis.structure_io import require_complex


REQUIRED = ("dG_separated", "dG_cross", "dSASA_int")
OPTIONAL = ("dSASA_hphobic", "dSASA_polar", "dG_separated/dSASAx100",
            "dG_cross/dSASAx100", "packstat", "sc_value", "nres_int",
            "delta_unsatHbonds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pdb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path,
                        default=Path(__file__).with_name("headtail_interface.xml"))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    report = {"input_pdb": str(args.input_pdb.resolve()),
              "xml": str(args.xml.resolve()), "seed": args.seed,
              "status": "failed", "stage": "input"}
    try:
        if not args.input_pdb.is_file() or not args.xml.is_file():
            raise FileNotFoundError("input PDB or XML missing")
        binder_length, receptor_length = require_complex(args.input_pdb)
        report["peptide_length"] = binder_length
        report["receptor_length"] = receptor_length
        report["geometry_before"] = audit(args.input_pdb, binder_length)
        if not report["geometry_before"]["geometry_ok"]:
            raise ValueError("input cyclic geometry failed")
        report["input_sha256"] = hashlib.sha256(args.input_pdb.read_bytes()).hexdigest()
        report["xml_sha256"] = hashlib.sha256(args.xml.read_bytes()).hexdigest()
        report["terminal_geometry_before"] = terminal_geometry(args.input_pdb, binder_length)
        if not 1.20 <= report["terminal_geometry_before"]["cn_angstrom"] <= 1.45:
            raise ValueError("input C(last)-N(first) geometry is not peptide-like")

        import pyrosetta
        from pyrosetta.rosetta.protocols.rosetta_scripts import RosettaScriptsParser
        report["stage"] = "pose_validation"
        pyrosetta.init("-mute all -ignore_zero_occupancy false -constant_seed -jran %d"
                       % args.seed)
        pose = pyrosetta.pose_from_pdb(str(args.input_pdb))
        report["pyrosetta_version"] = pyrosetta.version()
        if pose.total_residue() != binder_length + receptor_length or pose.num_chains() != 2:
            raise ValueError("unexpected residue or chain count")
        if any(pose.pdb_info().chain(i) != "A" for i in range(1, binder_length + 1)):
            raise ValueError("binder must be the first chain and have PDB chain ID A")
        if any(pose.pdb_info().chain(i) != "C" for i in range(binder_length + 1,
                                                                 pose.total_residue() + 1)):
            raise ValueError("receptor must be the second chain and have PDB chain ID C")
        if pose.num_jump() != 1:
            raise ValueError("expected exactly one interchain jump")
        report["bonded_before"] = bool(pose.residue(1).is_bonded(pose.residue(binder_length)))
        if not report["bonded_before"]:
            raise ValueError("Rosetta did not recognize the terminal covalent bond")
        before = pose.clone()
        report["binder_sequence"] = pose.sequence()[:binder_length]

        report["stage"] = "interface_analysis"
        mover = RosettaScriptsParser().generate_mover(str(args.xml))
        mover.apply(pose)
        report["bonded_after"] = bool(pose.residue(1).is_bonded(pose.residue(binder_length)))
        report["binder_ca_rmsd_angstrom"] = rmsd(before, pose, range(1, binder_length + 1))
        report["receptor_ca_rmsd_angstrom"] = rmsd(before, pose,
                                                     range(binder_length + 1,
                                                           pose.total_residue() + 1))
        report["terminal_cn_after_angstrom"] = float(
            pose.residue(binder_length).xyz("C").distance(pose.residue(1).xyz("N")))
        scores = dict(pose.scores)
        report["score_keys"] = sorted(scores)
        missing = [key for key in REQUIRED if key not in scores]
        if missing:
            raise ValueError("missing InterfaceAnalyzer metrics: %s" % ", ".join(missing))
        for key in REQUIRED + OPTIONAL:
            if key in scores:
                value = float(scores[key])
                if not math.isfinite(value):
                    raise ValueError("non-finite %s" % key)
                report[key] = value
        report["units"] = {"dG_separated": "R.E.U.", "dG_cross": "R.E.U.",
                           "dSASA_int": "A^2"}
        report["scoring_options"] = {"scorefxn": "ref2015", "ligandchain": "A",
                                     "pack_input": False, "pack_separated": True}
        if (not report["bonded_after"] or
                report["binder_ca_rmsd_angstrom"] > 0.05 or
                report["receptor_ca_rmsd_angstrom"] > 0.05 or
                not 1.20 <= report["terminal_cn_after_angstrom"] <= 1.45 or
                report["dSASA_int"] <= 0):
            raise ValueError("scoring changed coordinates/topology or has invalid interface area")
        report["status"] = "ok"
        report["stage"] = "complete"
    except Exception as exc:
        report["reason"] = "%s: %s" % (type(exc).__name__, exc)
    status_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

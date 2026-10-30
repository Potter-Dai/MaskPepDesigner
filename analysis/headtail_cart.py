#!/usr/bin/env python3
"""Cartesian Rosetta relax of a pre-cyclized Amber PDB."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from analysis.headtail_geometry import audit
from analysis.headtail_close import add_conect, rmsd
from analysis.structure_io import require_complex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pdb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--xml", type=Path,
                        default=Path(__file__).with_name("headtail_cart.xml"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = output / "cart_status.json"
    if status.exists():
        raise FileExistsError(status)
    xml = args.xml
    report = {"input_pdb": str(args.input_pdb.resolve()),
              "xml": str(xml.resolve()), "seed": args.seed, "status": "failed",
              "stage": "input"}
    try:
        if not args.input_pdb.is_file() or not xml.is_file():
            raise FileNotFoundError("input PDB or XML missing")
        report["input_sha256"] = hashlib.sha256(args.input_pdb.read_bytes()).hexdigest()
        report["xml_sha256"] = hashlib.sha256(xml.read_bytes()).hexdigest()
        n, receptor_length = require_complex(args.input_pdb)
        report["peptide_length"] = n
        report["receptor_length"] = receptor_length
        before_geometry = audit(args.input_pdb, n)
        report["before"] = before_geometry
        if (before_geometry["bad_cn_links"] or before_geometry["bad_o_n_links"] or
                before_geometry["bad_angle_links"] or
                before_geometry["intrapeptide_nonadjacent_heavy_pairs_below_2A"]):
            raise ValueError("input binder geometry failed Amber-stage acceptance")
        import pyrosetta
        from pyrosetta.rosetta.protocols.rosetta_scripts import RosettaScriptsParser
        report["stage"] = "cartesian_relax"
        pyrosetta.init("-mute all -ignore_zero_occupancy false -constant_seed -jran %d"
                       % args.seed)
        pose = pyrosetta.pose_from_pdb(str(args.input_pdb))
        if pose.total_residue() != n + receptor_length or pose.num_chains() != 2:
            raise ValueError("unexpected complex topology")
        if not pose.residue(1).is_bonded(pose.residue(n)):
            raise ValueError("Rosetta did not recognize head-to-tail bond")
        before = pose.clone()
        report["sequence_before"] = pose.sequence()
        scorefxn = pyrosetta.get_fa_scorefxn()
        report["input_ref2015_energy_reu"] = float(scorefxn(pose))
        from pyrosetta.rosetta.core.id import AtomID
        from pyrosetta.rosetta.core.scoring.constraints import (
            AtomPairConstraint, AngleConstraint, DihedralConstraint)
        from pyrosetta.rosetta.core.scoring.func import (
            HarmonicFunc, CircularHarmonicFunc)
        def atom_id(residue, atom_name):
            return AtomID(pose.residue(residue).atom_index(atom_name), residue)
        cn = (atom_id(n, "C"), atom_id(1, "N"))
        pose.add_constraint(AtomPairConstraint(
            cn[0], cn[1], HarmonicFunc(1.335, 0.025)))
        pose.add_constraint(AtomPairConstraint(
            atom_id(n, "O"), cn[1], HarmonicFunc(2.25, 0.06)))
        pose.add_constraint(AngleConstraint(
            atom_id(n, "CA"), cn[0], cn[1],
            HarmonicFunc(math.radians(116), math.radians(5))))
        pose.add_constraint(AngleConstraint(
            cn[0], cn[1], atom_id(1, "CA"),
            HarmonicFunc(math.radians(122), math.radians(5))))
        pose.add_constraint(DihedralConstraint(
            atom_id(n, "CA"), cn[0], cn[1], atom_id(1, "CA"),
            CircularHarmonicFunc(math.pi, math.radians(10))))
        report["closure_constraints"] = {"cn_angstrom": 1.335, "cn_sd_angstrom": 0.025,
                                         "o_n_angstrom": 2.25, "o_n_sd_angstrom": 0.06,
                                         "ca_c_n_deg": 116, "c_n_ca_deg": 122,
                                         "angle_sd_deg": 5, "omega_deg": 180,
                                         "omega_sd_deg": 10}
        RosettaScriptsParser().generate_mover(str(xml)).apply(pose)
        report["bonded_after"] = bool(pose.residue(1).is_bonded(pose.residue(n)))
        report["sequence_after"] = pose.sequence()
        report["binder_ca_rmsd_from_amber_angstrom"] = rmsd(before, pose, range(1, n + 1))
        report["receptor_ca_rmsd_angstrom"] = rmsd(
            before, pose, range(n + 1, pose.total_residue() + 1))
        report["output_ref2015_energy_reu"] = float(scorefxn(pose))
        final = output / "cart_relaxed.pdb"
        pose.dump_pdb(str(final))
        add_conect(final, n)
        report["output_pdb"] = str(final)
        report["output_sha256"] = hashlib.sha256(final.read_bytes()).hexdigest()
        report["after"] = audit(final, n)
        report["status"] = ("ok" if report["bonded_after"] and
                            report["sequence_before"] == report["sequence_after"] and
                            report["receptor_ca_rmsd_angstrom"] <= 0.05 and
                            report["after"]["geometry_ok"] else "geometry_failed")
        report["stage"] = "complete"
    except Exception as exc:
        report["reason"] = "%s: %s" % (type(exc).__name__, exc)
    status.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report.get(key) for key in
                      ("status", "stage", "reason", "bonded_after",
                       "binder_ca_rmsd_from_amber_angstrom", "receptor_ca_rmsd_angstrom",
                       "input_ref2015_energy_reu", "output_ref2015_energy_reu")},
                     sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

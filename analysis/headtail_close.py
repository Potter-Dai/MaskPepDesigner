#!/usr/bin/env python3
"""Declare a head-to-tail bond and repack an explicitly prepared complex."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from analysis.structure_io import require_complex


def coords(pose, resid, atom):
    return pose.residue(resid).xyz(atom)


def rmsd(before, after, residue_ids):
    values = [before.residue(i).xyz("CA").distance(after.residue(i).xyz("CA")) ** 2
              for i in residue_ids]
    return math.sqrt(sum(values) / len(values))


def pdb_atom(path, chain, resid, atom):
    for line in Path(path).read_text().splitlines():
        if line.startswith("ATOM  ") and line[21] == chain and int(line[22:26]) == resid \
                and line[12:16].strip() == atom:
            return int(line[6:11]), tuple(float(line[i:i + 8]) for i in (30, 38, 46))
    raise ValueError("missing %s%d %s in %s" % (chain, resid, atom, path))


def length(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def angle(a, b, c):
    u = [x - y for x, y in zip(a, b)]
    v = [x - y for x, y in zip(c, b)]
    cosine = sum(x * y for x, y in zip(u, v)) / (length(a, b) * length(c, b))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dihedral(a, b, c, d):
    p = [x - y for x, y in zip(b, a)]
    q = [x - y for x, y in zip(c, b)]
    r = [x - y for x, y in zip(d, c)]
    n1, n2 = cross(p, q), cross(q, r)
    qnorm = math.sqrt(sum(x * x for x in q))
    y = sum(x * y for x, y in zip(cross(n1, n2), q)) / qnorm
    x = sum(x * y for x, y in zip(n1, n2))
    return math.degrees(math.atan2(y, x))


def terminal_geometry(path, nres):
    n = pdb_atom(path, "A", 1, "N")[1]
    ca1 = pdb_atom(path, "A", 1, "CA")[1]
    c = pdb_atom(path, "A", nres, "C")[1]
    calast = pdb_atom(path, "A", nres, "CA")[1]
    return {"cn_angstrom": length(c, n),
            "ca_c_n_deg": angle(calast, c, n),
            "c_n_ca_deg": angle(c, n, ca1),
            "omega_deg": dihedral(calast, c, n, ca1)}


def add_conect(path, nres):
    carbon = pdb_atom(path, "A", nres, "C")[0]
    nitrogen = pdb_atom(path, "A", 1, "N")[0]
    lines = path.read_text().splitlines()
    lines = [line for line in lines if not line.startswith("END") and
             not (line.startswith("CONECT") and
                  ("%5d" % carbon) in line and ("%5d" % nitrogen) in line)]
    lines.extend(["CONECT%5d%5d" % (carbon, nitrogen), "END", ""])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pdb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    report = {"input_pdb": str(args.input_pdb), "seed": args.seed,
              "status": "failed", "stage": "start"}
    try:
        expected_length, receptor_length = require_complex(args.input_pdb)
        report["peptide_length"] = expected_length
        report["receptor_length"] = receptor_length
        report["before"] = terminal_geometry(args.input_pdb, expected_length)

        import pyrosetta
        from pyrosetta.rosetta.protocols.rosetta_scripts import RosettaScriptsParser
        report["stage"] = "cyclize_and_repack"
        pyrosetta.init("-mute all -ignore_zero_occupancy false -constant_seed -jran %d"
                       % args.seed)
        before = pyrosetta.pose_from_pdb(str(args.input_pdb))
        report["pyrosetta_version"] = pyrosetta.version()
        if before.total_residue() != expected_length + receptor_length:
            raise ValueError("unexpected total residue count")
        if any(before.pdb_info().chain(i) != "A" for i in range(1, expected_length + 1)):
            raise ValueError("reconstructed binder is not chain A")
        pose = before.clone()
        xml = Path(__file__).with_name("headtail_close.xml")
        report["xml"] = str(xml)
        report["xml_sha256"] = hashlib.sha256(xml.read_bytes()).hexdigest()
        protocol = RosettaScriptsParser().generate_mover(str(xml))
        protocol.apply(pose)
        report["bonded_in_rosetta_pose"] = pose.residue(1).is_bonded(pose.residue(expected_length))
        report["sequence_before"] = before.sequence()[:expected_length]
        report["sequence_after"] = pose.sequence()[:expected_length]
        report["binder_ca_rmsd_angstrom"] = rmsd(before, pose, range(1, expected_length + 1))
        report["receptor_ca_rmsd_angstrom"] = rmsd(before, pose,
                                                   range(expected_length + 1, pose.total_residue() + 1))
        scorefxn = pyrosetta.get_fa_scorefxn()
        report["total_energy_reu"] = float(scorefxn(pose))
        packed = output / "headtail_repacked.pdb"
        pose.dump_pdb(str(packed))
        add_conect(packed, expected_length)
        report["repacked_pdb"] = str(packed)
        report["after"] = terminal_geometry(packed, expected_length)
        g = report["after"]
        valid = (report["bonded_in_rosetta_pose"] and
                 report["sequence_before"] == report["sequence_after"] and
                 report["receptor_ca_rmsd_angstrom"] <= 0.05 and
                 1.20 <= g["cn_angstrom"] <= 1.45 and
                 100 <= g["ca_c_n_deg"] <= 135 and
                 105 <= g["c_n_ca_deg"] <= 140 and
                 abs(g["omega_deg"]) >= 150)
        if not valid:
            raise ValueError("closure, receptor, or sequence acceptance check failed")
        report["status"] = "ok"
        report["stage"] = "complete"
    except Exception as exc:
        report["reason"] = "%s: %s" % (type(exc).__name__, exc)
    status_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit every peptide bond and obvious nonbonded clashes in a cyclic complex."""
import argparse
import json
from pathlib import Path

from analysis.headtail_close import angle, dihedral, length


def atoms_by_chain(path):
    chains = {"A": {}, "C": {}}
    for line in Path(path).read_text().splitlines():
        if not line.startswith("ATOM  ") or line[21] not in chains:
            continue
        element = line[76:78].strip().upper()
        atom = line[12:16].strip()
        if element in ("H", "D") or atom.startswith("H"):
            continue
        residue = int(line[22:26])
        xyz = tuple(float(line[i:i + 8]) for i in (30, 38, 46))
        chains[line[21]].setdefault(residue, {})[atom] = xyz
    return chains


def audit(path, peptide_length=None):
    chains = atoms_by_chain(path)
    peptide, receptor = chains["A"], chains["C"]
    n = peptide_length or len(peptide)
    if set(peptide) != set(range(1, n + 1)) or not receptor:
        raise ValueError("unexpected peptide numbering or missing receptor")
    links = []
    for i in range(1, n + 1):
        j = i + 1 if i < n else 1
        left, right = peptide[i], peptide[j]
        c, n_atom = left["C"], right["N"]
        links.append({"from_residue": i, "to_residue": j,
                      "cn_angstrom": length(c, n_atom),
                      "o_n_angstrom": length(left["O"], n_atom),
                      "ca_c_n_deg": angle(left["CA"], c, n_atom),
                      "c_n_ca_deg": angle(c, n_atom, right["CA"]),
                      "omega_deg": dihedral(left["CA"], c, n_atom, right["CA"])})
    intrachain_min = float("inf")
    intrachain_pairs_below_2 = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if min(j - i, n - (j - i)) <= 1:
                continue
            for a in peptide[i].values():
                for b in peptide[j].values():
                    distance = length(a, b)
                    intrachain_min = min(intrachain_min, distance)
                    intrachain_pairs_below_2 += distance < 2.0
    interchain_min = float("inf")
    interchain_pairs_below_2 = 0
    for residue in peptide.values():
        for a in residue.values():
            for other in receptor.values():
                for b in other.values():
                    distance = length(a, b)
                    interchain_min = min(interchain_min, distance)
                    interchain_pairs_below_2 += distance < 2.0
    bad_cn = [x["from_residue"] for x in links if not 1.20 <= x["cn_angstrom"] <= 1.45]
    bad_on = [x["from_residue"] for x in links if x["o_n_angstrom"] < 2.0]
    bad_angles = [x["from_residue"] for x in links if not
                  (95 <= x["ca_c_n_deg"] <= 140 and 100 <= x["c_n_ca_deg"] <= 145)]
    return {"pdb": str(path), "peptide_length": n,
            "links": links, "bad_cn_links": bad_cn, "bad_o_n_links": bad_on,
            "bad_angle_links": bad_angles,
            "min_nonadjacent_intrapeptide_heavy_atom_angstrom": intrachain_min,
            "intrapeptide_nonadjacent_heavy_pairs_below_2A": intrachain_pairs_below_2,
            "min_interchain_heavy_atom_angstrom": interchain_min,
            "interchain_heavy_pairs_below_2A": interchain_pairs_below_2,
            "geometry_ok": not (bad_cn or bad_on or bad_angles or
                                intrachain_pairs_below_2 or interchain_pairs_below_2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peptide-length", type=int)
    parser.add_argument("pdb", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.pdb, args.peptide_length), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Check all internal linear-peptide bonds and heavy-atom clashes."""
from pathlib import Path

from analysis.headtail_close import length
from analysis.headtail_geometry import atoms_by_chain


def audit(path):
    chains = atoms_by_chain(Path(path))
    peptide, receptor = chains["A"], chains["C"]
    n = len(peptide)
    if set(peptide) != set(range(1, n + 1)) or not receptor:
        raise ValueError("unexpected linear-peptide numbering or missing receptor")
    links = [{"from_residue": i, "to_residue": i + 1,
              "cn_angstrom": length(peptide[i]["C"], peptide[i + 1]["N"])}
             for i in range(1, n)]
    bad_cn = [item["from_residue"] for item in links
              if not 1.20 <= item["cn_angstrom"] <= 1.45]
    intrachain_clashes = 0
    for i in range(1, n + 1):
        for j in range(i + 2, n + 1):
            for a in peptide[i].values():
                for b in peptide[j].values():
                    intrachain_clashes += length(a, b) < 2.0
    interface_clashes = 0
    for residue in peptide.values():
        for a in residue.values():
            for other in receptor.values():
                for b in other.values():
                    interface_clashes += length(a, b) < 2.0
    return {"links": links, "bad_cn_links": bad_cn,
            "intrapeptide_heavy_pairs_below_2A": intrachain_clashes,
            "interchain_heavy_pairs_below_2A": interface_clashes,
            "geometry_ok": not (bad_cn or intrachain_clashes or interface_clashes)}

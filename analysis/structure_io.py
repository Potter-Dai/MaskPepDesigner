"""Explicit, single-receptor-chain PDB preparation for peptide postprocessing.

The public adapter never infers an original PDB from a generated filename.
Output uses peptide chain A and receptor chain C. Other molecules are rejected.
"""
from pathlib import Path


AA3 = set("ALA ARG ASN ASP CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
AA1 = dict(zip("ALA ARG ASN ASP CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
               "ARNDCEQGHILKMFPSTWYV"))
BACKBONE = {"N", "CA", "C", "O"}


def chains(path):
    found = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith("ATOM  "):
            continue
        chain = line[21]
        key = (line[22:26], line[26])
        name = line[17:20].strip()
        if name not in AA3:
            raise ValueError("nonstandard residue: %s" % name)
        atom = line[12:16].strip()
        residues = found.setdefault(chain, {})
        record = residues.setdefault(key, {"name": name, "atoms": set()})
        if record["name"] != name:
            raise ValueError("inconsistent residue identity")
        record["atoms"].add(atom)
    return found


def lengths(path):
    return {chain: len(residues) for chain, residues in chains(path).items()}


def peptide_sequence(path):
    return "".join(AA1[item["name"]] for item in chains(path)["A"].values())


def require_complex(path, require_full_backbone=True):
    found = chains(path)
    if set(found) != {"A", "C"} or not found["A"] or not found["C"]:
        raise ValueError("expected peptide A and one receptor C")
    if require_full_backbone:
        for chain, residues in found.items():
            for key, residue in residues.items():
                if not BACKBONE <= residue["atoms"]:
                    raise ValueError("missing backbone atoms in %s%s" % (chain, key[0].strip()))
    return len(found["A"]), len(found["C"])


def _selected_atoms(path, chain):
    lines = [line for line in Path(path).read_text().splitlines()
             if line.startswith("ATOM  ") and line[21] == chain]
    if not lines:
        raise ValueError("chain %s is absent: %s" % (chain, path))
    return lines


def _render(peptide, receptor):
    rendered = []
    serial = 1
    for chain, records in (("A", peptide), ("C", receptor)):
        residue_ids = {}
        for line in records:
            key = line[22:27]
            residue_ids.setdefault(key, len(residue_ids) + 1)
            rendered.append(line[:6] + "%5d" % serial + line[11:21] + chain +
                            "%4d " % residue_ids[key] + line[27:])
            serial += 1
        rendered.append("TER")
    return "\n".join(rendered + ["END", ""])


def reconstruct_backbone(input_pdb, original_pdb, peptide_chain, receptor_chain, output):
    """Combine generated peptide with the exact original receptor, then PDBFixer."""
    input_pdb, original_pdb, output = map(Path, (input_pdb, original_pdb, output))
    if not input_pdb.is_file() or not original_pdb.is_file() or output.exists():
        raise ValueError("inputs must exist and output must not exist")
    if peptide_chain == receptor_chain:
        raise ValueError("peptide and receptor chain IDs must differ")
    prepared = output.with_name(output.stem + "_assembled.pdb")
    if prepared.exists():
        raise FileExistsError(prepared)
    prepared.write_text(_render(_selected_atoms(input_pdb, peptide_chain),
                                _selected_atoms(original_pdb, receptor_chain)))
    # The PDBFixer step is for open/backbone inputs only. It must never process
    # an already closed complex because terminal chemistry could be changed.
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    fixer = PDBFixer(filename=str(prepared))
    fixer.findMissingResidues()
    model_chains = list(fixer.topology.chains())
    for key in list(fixer.missingResidues):
        chain = model_chains[key[0]]
        if key[1] in (0, len(list(chain.residues()))):
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        raise ValueError("nonstandard residues are unsupported")
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7)
    with output.open("w") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    require_complex(output)
    return output


def normalize_closed_complex(input_pdb, peptide_chain, receptor_chain, output):
    """Copy a complete complex, changing chain IDs but preserving CONECT serials."""
    input_pdb, output = Path(input_pdb), Path(output)
    if not input_pdb.is_file() or output.exists():
        raise ValueError("input must exist and output must not exist")
    if peptide_chain == receptor_chain:
        raise ValueError("peptide and receptor chain IDs must differ")
    found = chains(input_pdb)
    if set(found) != {peptide_chain, receptor_chain}:
        raise ValueError("input contains unexpected protein chains")
    if any(line.startswith("HETATM") for line in input_pdb.read_text().splitlines()):
        raise ValueError("heterogens are unsupported")
    lines = []
    residue_ids = {peptide_chain: {}, receptor_chain: {}}
    for line in input_pdb.read_text().splitlines():
        if line.startswith("ATOM  ") and len(line) > 26:
            key = line[22:27]
            ids = residue_ids[line[21]]
            ids.setdefault(key, len(ids) + 1)
            replacement = {peptide_chain: "A", receptor_chain: "C"}.get(line[21])
            if replacement is not None:
                line = line[:21] + replacement + "%4d " % ids[key] + line[27:]
        elif line.startswith("TER"):
            # TER residue columns are optional; Rosetta needs only the separator.
            line = "TER"
        lines.append(line)
    output.write_text("\n".join(lines) + "\n")
    require_complex(output)
    return output

#!/usr/bin/env python3
"""Cyclic Amber minimization; preserves and checks the head-to-tail bond.

Adapted in scope/parameters from DiffPepBuilder analysis/amber_minimize.py,
without editing or importing its linear-peptide postprocess wrapper.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from analysis.headtail_geometry import audit, atoms_by_chain
from analysis.headtail_close import add_conect
from analysis.structure_io import require_complex


def ca_rmsd(before, after, chain, residue_ids):
    a, b = atoms_by_chain(before), atoms_by_chain(after)
    return math.sqrt(sum(sum((x - y) ** 2 for x, y in zip(a[chain][i]["CA"],
                                                            b[chain][i]["CA"]))
                         for i in residue_ids) / len(residue_ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pdb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = output / "amber_status.json"
    if status.exists():
        raise FileExistsError(status)
    report = {"input_pdb": str(args.input_pdb.resolve()),
              "status": "failed", "stage": "input"}
    try:
        if args.iterations < 1 or not args.input_pdb.is_file():
            raise ValueError("positive iterations and existing input PDB required")
        report["input_sha256"] = hashlib.sha256(args.input_pdb.read_bytes()).hexdigest()
        n, receptor_length = require_complex(args.input_pdb)
        report["peptide_length"] = n
        report["receptor_length"] = receptor_length
        report["before"] = audit(args.input_pdb, n)

        import openmm
        from openmm import app, unit
        report["openmm_version"] = openmm.version.version
        report["stage"] = "amber_minimize"
        pdb = app.PDBFile(str(args.input_pdb))
        residues = list(pdb.topology.residues())
        if len([r for r in residues if r.chain.id == "A"]) != n or \
                len([r for r in residues if r.chain.id == "C"]) != receptor_length:
            raise ValueError("unexpected chain IDs or residue counts")
        terminal_bonds = [(a.name, b.name) for a, b in pdb.topology.bonds()
                          if {a.residue.index, b.residue.index} == {0, n - 1}]
        report["terminal_bonds_in_openmm"] = terminal_bonds
        if ("C", "N") not in terminal_bonds and ("N", "C") not in terminal_bonds:
            raise ValueError("OpenMM did not recognize the cyclic bond")
        # Minimize the cyclic binder in isolation. Freezing the entire receptor
        # with zero masses left its large internal strain in the minimizer and
        # empirically stalled the 2-complex smoke test (2026-09-23). The original
        # receptor coordinates are copied back exactly after binder minimization.
        model = app.Modeller(pdb.topology, pdb.positions)
        model.delete([chain for chain in model.topology.chains() if chain.id == "C"])
        report["amber_scope"] = "cyclic binder only; receptor coordinates unchanged"
        forcefield = app.ForceField("amber99sb.xml")
        system = forcefield.createSystem(model.topology, constraints=app.HBonds)
        atoms = list(model.topology.atoms())

        # DiffPepBuilder uses 10 kcal/mol/A^2 heavy-atom restraints. Begin with
        # that setting, then relax only binder C-alpha atoms at 2 kcal/mol/A^2.
        restraint = openmm.CustomExternalForce(
            "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        for name in ("k", "x0", "y0", "z0"):
            restraint.addPerParticleParameter(name)
        kcal_a2_to_kj_nm2 = 4.184 / 0.01
        restraints = []
        positions_nm = model.positions.value_in_unit(unit.nanometer)
        for atom in atoms:
            if atom.element.symbol == "H":
                continue
            xyz = positions_nm[atom.index]
            index = restraint.addParticle(atom.index,
                                          [10.0 * kcal_a2_to_kj_nm2, xyz.x, xyz.y, xyz.z])
            restraints.append((index, atom, (xyz.x, xyz.y, xyz.z)))
        system.addForce(restraint)
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName("CPU")
        simulation = app.Simulation(model.topology, system, integrator, platform,
                                    {"Threads": "1"})
        simulation.context.setPositions(model.positions)
        report["initial_energy_kcal_per_mol"] = simulation.context.getState(
            getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        simulation.minimizeEnergy(tolerance=2.39 * unit.kilocalories_per_mole,
                                  maxIterations=args.iterations)
        report["stage1_energy_kcal_per_mol"] = simulation.context.getState(
            getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        for index, atom, xyz in restraints:
            k = 2.0 if atom.name == "CA" else 0.0
            restraint.setParticleParameters(index, atom.index,
                                            [k * kcal_a2_to_kj_nm2] + list(xyz))
        restraint.updateParametersInContext(simulation.context)
        simulation.minimizeEnergy(tolerance=2.39 * unit.kilocalories_per_mole,
                                  maxIterations=args.iterations)
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        report["stage2_energy_kcal_per_mol"] = state.getPotentialEnergy().value_in_unit(
            unit.kilocalories_per_mole)
        relaxed = output / "amber_cyclic_relaxed.pdb"
        final_positions = state.getPositions().value_in_unit(unit.angstrom)
        # OpenMM normalizes some hydrogen names (e.g. 1HB -> HB2). PDBFile
        # retains atom order, so map by validated order rather than name.
        binder_lines = [line for line in args.input_pdb.read_text().splitlines()
                        if line.startswith("ATOM  ") and line[21] == "A"]
        if len(binder_lines) != len(atoms):
            raise ValueError("binder atom count changed during PDB import")
        for line, atom in zip(binder_lines, atoms):
            if (line[22:26].strip() != atom.residue.id or
                    (line[76:78].strip().upper() != atom.element.symbol.upper()) or
                    (atom.element.symbol != "H" and line[12:16].strip() != atom.name)):
                raise ValueError("binder atom order changed at %s %s" %
                                 (atom.residue.id, atom.name))
        lines = []
        binder_index = 0
        for line in args.input_pdb.read_text().splitlines():
            if line.startswith("ATOM  ") and line[21] == "A":
                xyz = final_positions[binder_index]
                binder_index += 1
                line = line[:30] + "%8.3f%8.3f%8.3f" % (xyz.x, xyz.y, xyz.z) + line[54:]
            lines.append(line)
        relaxed.write_text("\n".join(lines) + "\n")
        add_conect(relaxed, n)
        report["output_pdb"] = str(relaxed)
        report["output_sha256"] = hashlib.sha256(relaxed.read_bytes()).hexdigest()
        report["after"] = audit(relaxed, n)
        report["binder_ca_rmsd_angstrom"] = ca_rmsd(
            args.input_pdb, relaxed, "A", range(1, n + 1))
        receptor_ids = sorted(atoms_by_chain(args.input_pdb)["C"])
        report["receptor_ca_rmsd_angstrom"] = ca_rmsd(
            args.input_pdb, relaxed, "C", receptor_ids)
        geometry = report["after"]
        report["binder_geometry_ok"] = not (
            geometry["bad_cn_links"] or geometry["bad_o_n_links"] or
            geometry["bad_angle_links"] or
            geometry["intrapeptide_nonadjacent_heavy_pairs_below_2A"])
        # Interface clashes are recorded here, but handled by the following
        # receptor-side-chain/binder-backbone Cartesian Rosetta stage.
        report["status"] = ("ok" if report["binder_geometry_ok"] and
                            report["receptor_ca_rmsd_angstrom"] <= 0.01
                            else "geometry_failed")
        report["stage"] = "complete"
    except Exception as exc:
        report["reason"] = "%s: %s" % (type(exc).__name__, exc)
    status.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report.get(key) for key in
                      ("status", "stage", "reason", "binder_ca_rmsd_angstrom",
                       "initial_energy_kcal_per_mol", "stage2_energy_kcal_per_mol")},
                     sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

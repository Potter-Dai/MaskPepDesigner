"""Dependency-light linear Amber stage using DiffPepBuilder relax parameters.

The legacy ``amber_minimize.py`` imports Uni-Fold at module load time. This
adapter operates on the already PDBFixer-completed complex and uses its same
amber99sb force field, heavy-atom restraints, tolerance, and three passes.
"""
import io


def relax(pdb_text, passes=3):
    import openmm
    from openmm import app, unit

    energy = unit.kilocalories_per_mole
    length = unit.angstrom
    current = pdb_text
    energies = []
    for _ in range(passes):
        pdb = app.PDBFile(io.StringIO(current))
        forcefield = app.ForceField("amber99sb.xml")
        system = forcefield.createSystem(pdb.topology, constraints=app.HBonds)
        restraint = openmm.CustomExternalForce(
            "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addGlobalParameter("k", 10.0 * energy / length ** 2)
        for name in ("x0", "y0", "z0"):
            restraint.addPerParticleParameter(name)
        for atom in pdb.topology.atoms():
            if atom.element.symbol != "H":
                restraint.addParticle(atom.index, pdb.positions[atom.index])
        system.addForce(restraint)
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        simulation = app.Simulation(pdb.topology, system, integrator,
                                    openmm.Platform.getPlatformByName("CPU"),
                                    {"Threads": "1"})
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy(tolerance=2.39 * energy, maxIterations=0)
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energies.append(state.getPotentialEnergy().value_in_unit(energy))
        output = io.StringIO()
        app.PDBFile.writeFile(pdb.topology, state.getPositions(), output, keepIds=True)
        current = output.getvalue()
    return current, {"openmm_version": openmm.version.version,
                     "passes": passes, "energy_kcal_per_mol": energies}

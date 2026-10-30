# Linear and head-to-tail peptide postprocessing

The historical repository-root `run_postprocess.py` and DiffPepBuilder-derived
`analysis/postprocess.py` remain available. The new one-design entry point is:

```bash
python -m analysis.run_postprocess \
  --topology linear --input-state backbone \
  --input-pdb generated_linear.pdb --original-pdb original_complex.pdb \
  --binder-chain B --receptor-chain A --output-dir results/linear_001

python -m analysis.run_postprocess \
  --topology headtail --input-state backbone \
  --input-pdb generated_cyclic.pdb --original-pdb original_complex.pdb \
  --binder-chain B --receptor-chain A --output-dir results/headtail_001

python -m analysis.run_postprocess \
  --topology headtail --input-state closed-complex --score-only \
  --input-pdb already_closed.pdb \
  --binder-chain A --receptor-chain C --output-dir results/headtail_score_001

python -m analysis.summarize_postprocess \
  --input-dir results --output results_summary.tsv
```

`backbone` takes peptide coordinates from `--input-pdb` and the complete
receptor from the **exact** `--original-pdb`; it reconstructs missing atoms
before topology-specific treatment. `closed-complex` copies the full input
complex, preserving atom serials and `CONECT` records. The input is never
modified. Both modes accept exactly one standard-amino-acid peptide chain and
one standard-amino-acid receptor chain. Output is normalized to peptide A and
receptor C. The current implementation does not support nonstandard residues,
other ligands, multi-chain receptors, or disulfide cyclization.

The head-to-tail path runs closure/repack (only for backbone input), cyclic
Amber, Cartesian Rosetta relax, full-link/clash validation, and direct interface
scoring. `--score-only` requires an already closed complex and refuses one
whose covalent topology is not recognized by Rosetta. Each output directory
must be new and contains `status.json`; any failed stage returns nonzero.
`dG_separated` and `dG_cross` are Rosetta energy units (R.E.U.), **not**
experimental binding free energies or kcal/mol. Linear and head-to-tail
protocols are recorded separately and their scores should not be treated as
automatically comparable.

Postprocessing needs BioPython, PDBFixer, OpenMM and a compatible licensed
PyRosetta installation. PyRosetta is not distributed with this repository;
see https://www.pyrosetta.org/downloads and its license terms. The linear
Amber adapter uses DiffPepBuilder's `amber99sb`, 10 kcal/mol/Å² heavy-atom
restraints, 2.39 kcal/mol tolerance and three-pass settings on the already
PDBFixer-completed complex, without the legacy Uni-Fold import. The original
linear Rosetta XML is reused unchanged. The analysis code and XML derive from
DiffPepBuilder. Keep its original attribution
and the MIT notice in `analysis/DIFFPEPBUILDER_LICENSE.txt` when redistributing.

# MaskPepDesigner

MaskPepDesigner is a deep learning framework dedicated to the **de novo design of peptide binders**. Developed from Diffold, which builds on DP Technology Uni-Fold and AlphaFold, it shifts the focus from general protein structure prediction to the specific challenge of designing novel, high-affinity peptide binders for given target protein structures.

## Overview

Unlike traditional structure prediction models, MaskPepDesigner incorporates advanced multi-task training mechanics, including structural masking and sequence-structure co-design. It aims to eliminate distribution shifts during inference and enables accurate prediction and generation of peptide sequences that bind to target pockets or interfaces.

Key features include:
*   **De Novo Peptide Design:** Specifically tuned for designing peptide binders from scratch based on structural targets.
*   **Multi-Task Training Framework:** Enhanced robustness and single-side prediction accuracy via mixed "Fold" (sequence to structure) and "Invfold" (structure to sequence) tasks.
*   **Integrated Feature Encoders:** Support for leveraging pre-trained language model representations (e.g., ESM-2) for stronger sequence priors.

## Repository Structure

- `MPDesigner/`: MaskPepDesigner model, training task, data processing and inference utilities.
- `diffold/`: Compatibility imports for existing code and checkpoints.
- `unifold/`: Retained upstream Uni-Fold components and attribution.
- `SSbuilder/`: Sequence and Structure builder utilities for dataset preparation and feature extraction.
- `analysis/`: Scripts for evaluating predictions and calculating metrics like ipSAE, pLDDT, and Rosetta energy scores.
- `run_MPDesigner.py` / `run_MPDesigner_ss.py`: Inference entry points.
- `train_MPDesigner.sh` / `finetune_MPDesigner*.sh`: Training entry points.

## Quick Start

### Inference

To run the design process using pre-trained model weights:
```bash
python run_MPDesigner.py \
    --tasks /path/to/tasks.json \
    --input_pdbs /path/to/reference_pdbs \
    --checkpoint /path/to/checkpoint.pt \
    --outputs /path/to/output \
    --model MPDesigner_v65
```

`MPDesigner_v65` is an example configuration name, not a declaration of which
checkpoint you have. Use the exact version and modifiers used to train your
checkpoint. Existing configuration strings (including `denovo` variants) remain
valid. The name change does not convert one architecture/version into another.
Peptide inference currently expects receptor first, peptide second, with explicit
hotspots and the peptide `@` marker; see [input syntax](MPDESIGNER_README.md).

### Training

The general training script takes five positional arguments:

```bash
bash train_MPDesigner.sh \
    /path/to/unifold_db /path/to/distillation_db /path/to/esm_db \
    /path/to/output MPDesigner_v65
```

General finetuning takes the same first four arguments, then checkpoint and model
configuration. The peptide-specific launcher uses four arguments:

```bash
bash finetune_MPDesigner_peptide.sh \
    /path/to/peptide_db /path/to/output /path/to/checkpoint.pt denovo
```

Run launchers from the repository root in an environment with the project's
PyTorch, Uni-Core and Uni-Fold dependencies. `pip install -e .` installs this
repository as `MaskPepDesigner`; it does not provision the complete GPU environment.

### Naming and compatibility

The Python API is `from MPDesigner.nn.mpdesigner import MPDesigner`. Uni-Core uses
`--user-dir MPDesigner --task MPDesigner --loss MPDesigner --arch MPDesigner_v6`.
Always supply `--model-name` explicitly. Old `run_diffold*.py`, training shell
scripts, `diffold.*` imports and `diffold`/`diffold_v6` registrations remain as
compatibility entry points. Module parameter names and layer dimensions are
unchanged by this migration; real checkpoint/GPU validation is still required.
Do not rename `unifold` imports or remove upstream copyright/license notices.

See [migration and review](docs/code_review_2026-09-07.md) for fixed defects,
remaining scientific/reproducibility risks and verification limits.

## Acknowledgments

MaskPepDesigner was developed from Diffold, which in turn is based on DP Technology Uni-Fold. This repository retains components from [Uni-Fold](https://github.com/dptech-corp/Uni-Fold) and [AlphaFold](https://github.com/deepmind/alphafold/). We thank the authors for making their codebase open-source.

## License

Please refer to the `LICENSE` file for details.

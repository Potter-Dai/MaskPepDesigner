# MaskPepDesigner

MaskPepDesigner is a deep learning framework dedicated to the **de novo design of peptide binders**. Built upon the foundations of Uni-Fold/AlphaFold, it shifts the focus from general protein structure prediction to the specific challenge of designing novel, high-affinity peptide binders for given target protein structures.

## Overview

Unlike traditional structure prediction models, MaskPepDesigner incorporates advanced multi-task training mechanics, including structural masking and sequence-structure co-design. It aims to eliminate distribution shifts during inference and enables accurate prediction and generation of peptide sequences that bind to target pockets or interfaces.

Key features include:
*   **De Novo Peptide Design:** Specifically tuned for designing peptide binders from scratch based on structural targets.
*   **Multi-Task Training Framework:** Enhanced robustness and single-side prediction accuracy via mixed "Fold" (sequence to structure) and "Invfold" (structure to sequence) tasks.
*   **Integrated Feature Encoders:** Support for leveraging pre-trained language model representations (e.g., ESM-2) for stronger sequence priors.

## Repository Structure

- `diffold/` & `unifold/`: Core model implementations.
- `SSbuilder/`: Sequence and Structure builder utilities for dataset preparation and feature extraction.
- `analysis/`: Scripts for evaluating predictions and calculating metrics like ipSAE, pLDDT, and complex free energy.
- `run_diffold.py` / `run_diffold_ss.py`: Entry points for training and running inference.

## Quick Start

### Inference

To run the design process using pre-trained model weights:
```bash
python run_diffold.py \
    --config_path /path/to/your/config.yml \
    --checkpoint_path /path/to/checkpoint.pt \
    --output_dir /path/to/output
```

### Training

To finetune or train the model on your own peptide-protein complex data:
```bash
bash train_diffold.sh \
    /path/to/training/data/ \
    /path/to/output/directory/ \
    model_name
```
(See `train_diffold.sh` or `finetune_diffold.sh` for exact arguments and distributed training setup).

## Acknowledgments

This repository builds upon the architecture and framework established by [Uni-Fold](https://github.com/dptech-corp/Uni-Fold) and [AlphaFold](https://github.com/deepmind/alphafold/). We thank the authors for making their codebase open-source.

## License

Please refer to the `LICENSE` file for details.

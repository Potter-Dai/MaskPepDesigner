# MPDesigner input and inference reference

Reference documentation for preparing task configurations and `contig` strings for `MaskPepDesigner`.

## 1. Historical task types:

1. protein folding / conformation sampling.

2. inverse folding (aka sequence design).

3. joint design of protein sequence & structure, including

    1. motif-scaffolding

    2. structure inpainting

    3. de novo protein generation

4. protein sequence and / or structure representation

4. a hybrid task of all.

## 2. Run MPDesigner:

1. install the environment (the same as Uni-Fold)

2. clone the MaskPepDesigner repo, `cd` into it, and run `pip install -e .`.

3. run the following command. for instructions on optional parameters, run `python run_MPDesigner.py -h`.

    ```
    python run_MPDesigner.py \
        -t <path to a json file of task configuration> \
        -i <input directories of pdbs> \
        -o <output directory> \
        -c <model parameter (*.pt)> \
        -m <exact model configuration used for this checkpoint>
    ```

## 3. How to prepare your inputs:

### 3.1. task configuration.

The json file of task configuration is a dictionary of dictionaries.

The outer dictionary is of items indexed by (unique) task ids.

The inner dictionary contains keyword arguments to build a `Job` object.

Available keyword arguments are below. The current peptide pipeline needs a
reference PDB and an explicit contig with receptor hotspots.

```python
{
    "contig": "A1-153",
    # prompt of how to compose a sample. see how to build contigs below.
    "pdb": "101m",
    # pdb id of the reference of design.
    # a pdb file must exist at path `<input_pdbs>/<pdb>.pdb`.
    "initial_t": 1.,
    # float in (0, 1], specifies the start of the diffusion process.
    # with t=1 the pure prior is used;
    # with t<1, noise is injected (the lower t, the lower noise ratio).
    "num_steps": 10,
    # a total of reverse-diffusion steps.
    # notably, [`initial_t`, 0.] is uniformly splitted into `num_steps`+1 grids.
    # by default, setting 10 can be good enough.
    "num_replica": 10,
    # repeating times of the job (different seeds).
    "seed": 0,
    # meta random seed of the job.
    "save_trajectory": True,
    # whether to save the trajectory. Setting `False` is very slightly faster.
}
```

### 3.2. setting the `contig`.

A `contig` is a string that specified what the protein is composed of.

A `contig` is built of `segment`s. Each `segement` specified a continuous segment in the protein. Use "/" to concatenate different segments into a chain. Use ":" to concatenate different chains.

Each `segment` should be specified by corresponding grammar:

    ```
        <s><chain_id><start_id>-<end_id>
    ```

Here, `symbol` is a symbol specifying how to use the segment; `chain_id` is the corresponding chain id in the given pdb reference; `start_id` and `end_id` are residue ids in the corresponding chain (both ends included). When `end_id` is not set, it is set to `start_id` by default (i.e. grabbing only the residue with the `start_id`).

The valid `symbol`s contain "\$", "\*" and "\%".

- if the `symbol` is not set, then both structure and the sequence of the residue is used as fixed motif (i.e. they are not changed in both the noise-adding and the output).

- if "\$" is provided, only the *structure* of the residue is used as motif. (remark: "\$"=S with a strikethrough, indicating *no sequence*.)

- if "\*" is provided, only the *sequence* of the residue is used as motif. (remark: "\*"=x with a strikethrough, indicating *no structure*.)

- if "\%" is provided then *both sequence and structure* are injected with noise. This option is generally useful when `initial_t`<1 is specified.

If a segment is to be designed (i.e. with on referenced structures or sequences at all), then it should contain no `chain_id` nor `symbol`. A length from `start_id` and `end_id` is sampled for this segment.

See the example usage of `task`s and `contig`s at [example_tasks.json](examples/example_tasks.json).

### 3.3. Peptide and hotspot markers

Use `@` before the peptide segment and `{...}` after a receptor segment to mark
hotspot residue numbers. For example, `A1-153{25,30-32}:@12-18` keeps receptor
residues A1–153 as input and generates a peptide of length 12–18. All referenced
residues and hotspots must exist in the reference PDB. Put the receptor first and
the peptide second: current crop/mask code assumes internal chain IDs 1 and 2.
A generated sample's receptor may be cropped; outputs are backbone structures.
See the review for current output-mask and coordinate-frame limitations.

## Notes

1. Running time is approximately $180*x^2+70$ milliseconds per step, where $100*x$ is the number of residues.

import os
import numpy as np
from typing import *

from .utils import (
    LMDBDataset,
    NumpyDict,
    SymmetryOp,
    sequence_to_aatype,
    apply_symmetry_op,
    residue_to_all_atom_info,
    recur_print
)

# from .db import ESMAtlasDatabase

from Bio.PDB import PDBParser
import numpy as np
from unifold.data import residue_constants as rc

LOAD_KEY_SCHEMA = {
    "aatype": np.int64,
    "residue_index": np.int64,
    "chain_id": np.int64,
    "all_atom_positions": np.float32,
    "all_atom_mask": np.float32,
    "bfactor": np.float32,
    "resolution": np.float32,
    "ligmask": np.int64,
    "hpmask": np.int64,
}

import pickle, gzip

def load_plddt_spline():
    """Load precalcuated esm spline.
    """
    plddt_spline_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "plddt_spline.pkl.gz"
    )
    with gzip.open(plddt_spline_path, "rb") as f:
        spline = pickle.load(f)
    return spline

spline = load_plddt_spline()

def esm_plddt_kernel(x):
    return (
        (x < .4) * spline(.4) * x / .4 +
        (x >= .4) * spline(x)
    ) / spline(1.)

def preprocess(
    feature: NumpyDict,
) -> NumpyDict:
    # make bfactors
    if "confidence" in feature:
        bfactor = feature.pop("confidence")
    elif "local_plddt" in feature:
        bfactor = feature.pop("local_plddt") / 100.
    elif "bfactor" not in feature:
        bfactor = np.min(feature["all_atom_mask"][:, :3], axis=-1)
    else:   # for esm fold results, we square the plddt, as the model tends to be
            # too optimistic compared with alphafold.
        bfactor = esm_plddt_kernel(feature["bfactor"])

    assert np.all(bfactor >= 0.) and np.all(bfactor <= 1.), \
        f"wrong bfactor found. min={bfactor.min()}, max={bfactor.max()}."
    feature["bfactor"] = bfactor

    # clear aatype
    assert len(feature["aatype"].shape) <= 2
    if len(feature["aatype"].shape) == 2:
        feature["aatype"] = np.argmax(feature["aatype"], axis=-1)

    # clear resolution
    if "resolution" in feature:
        feature["resolution"] = feature["resolution"].reshape(1)
    else:
        feature["resolution"] = np.array([0.], dtype=float)

    # recalculate residx (starting from 1)
    feature["residue_index"] = np.arange(1, feature["aatype"].shape[0]+1, dtype=int)

    # placeholder for chain ids
    feature["chain_id"] = np.ones_like(feature["aatype"], dtype=int)

    feature = {
        k: v.astype(LOAD_KEY_SCHEMA[k])
        for k, v in feature.items()
        if k in LOAD_KEY_SCHEMA
    }

    return feature


def load_unifold_reference(
    sequence_ids: List[str],
    feature_lmdb: LMDBDataset,
    label_ids: Optional[List[str]]=None,
    label_lmdb: Optional[LMDBDataset]=None,
    symmetry_ops: Optional[List[SymmetryOp]]=None,
) -> NumpyDict:
    features = [feature_lmdb.get_by_key(i) for i in sequence_ids]
    if label_ids is not None:   # load labels
        assert label_lmdb is not None
        labels = [label_lmdb.get_by_key(i) for i in label_ids]
        if symmetry_ops is not None:
            labels = [apply_symmetry_op(i, op) for i, op in zip(labels, symmetry_ops)]
        [f.update(l) for f, l in zip(features, labels)]
    features = [preprocess(f) for f in features]
    features = merge_monomers(features)
    return features

def merge_monomers(
    features: List[NumpyDict],
) -> NumpyDict:
    for i, f in enumerate(features, start=1):
        f["chain_id"] = f["chain_id"] * i

    merged_feat = {
        k: np.concatenate([f[k] for f in features], axis=0)
        for k in features[0].keys()
    }

    merged_feat["resolution"] = merged_feat["resolution"][0:1]

    return merged_feat

pdb_parser = PDBParser(QUIET=True)

# not used until inference.
def load_pdb_reference(
    pdb_path: str,
    pdb_id: str = '',
    load_bfactor: bool = True,
    reindex: bool = False,
):
    # TODO: this script does not handle missing residues.
    prot = pdb_parser.get_structure(pdb_id, pdb_path)
    prot = next(prot.get_models())  # get the first model and ignore the rest.

    res = [r for r in prot.get_residues()]

    sequence = "".join(rc.restype_3to1.get(r.get_resname(), "X") for r in res)
    aatype = sequence_to_aatype(sequence)

    if reindex:
        residue_index = np.arange(1, len(sequence)+1, dtype=int)
    else:
        residue_index = np.array([r.get_id()[1] for r in res], dtype=int)

    all_atom_infos = [residue_to_all_atom_info(r) for r in res]
    all_atom_mask = np.stack([r[0] for r in all_atom_infos], axis=0)
    all_atom_positions = np.stack([r[1] for r in all_atom_infos], axis=0)

    if load_bfactor:
        # specifically for distillation samples, where bfactors \in [0, 1] are plddts.
        bfactor = np.array([r[2] for r in all_atom_infos], dtype=float)
    else:
        # for general PDB files, ignore these bfactors.
        bfactor = np.ones_like(aatype, dtype=float)

    features = {
        "aatype": aatype,
        "residue_index": residue_index,
        "all_atom_positions": all_atom_positions,
        "all_atom_mask": all_atom_mask,
        "chain_id": np.ones_like(aatype, dtype=int),
        "bfactor": bfactor,
    }

    return features

from unifold.dataset import *

from typing import Mapping
import numpy as np
import pickle, gzip

def augment_positions(all_chain_labels, alg='identity'):
    if alg == "identity":
        ret = [
            {
                "input_atom_positions": l["all_atom_positions"],
                "input_atom_mask": l["all_atom_mask"],
            } for l in all_chain_labels
        ]
    else:
        raise NotImplementedError(f"{alg} not implemented.")
    
    return ret

def load_decoy_positions(sequence_ids, label_ids, decoy_dir):
    # demo load.
    paths = [os.path.join(decoy_dir, f"{s}.decoy.pkl.gz") for s in sequence_ids]
    return [pickle.load(gzip.open(p)) for p in paths]


def load(
    sequence_ids: List[str],
    feature_dir,
    msa_feature_dir: str,
    template_feature_dir: str,
    uniprot_msa_feature_dir: Optional[str] = None,
    label_ids: Optional[List[str]] = None,
    label_dir: Optional[str] = None,
    symmetry_operations: Optional[List[Operation]] = None,
    is_monomer: bool = False,
    train_max_date: Optional[str] = None,
    initial_structure_mode: str = "augment",
    augmentation_alg: str = "identity",
    decoy_dir: Optional[str] = None,
) -> NumpyExample:

    all_chain_features = [
        load_single_feature(
            s,
            feature_dir,
            msa_feature_dir,
            template_feature_dir,
            uniprot_msa_feature_dir,
            is_monomer,
            train_max_date,
        )
        for s in sequence_ids
    ]

    if label_ids is not None:
        # load labels
        assert len(label_ids) == len(sequence_ids)
        assert label_dir is not None
        if symmetry_operations is None:
            symmetry_operations = ["I" for _ in label_ids]
        all_chain_labels = [
            load_single_label(l, label_dir, o)
            for l, o in zip(label_ids, symmetry_operations)
        ]
        # update labels into features to calculate spatial cropping etc.
        [f.update(l) for f, l in zip(all_chain_features, all_chain_labels)]
    
    if initial_structure_mode == "augment":
        assert isinstance(augmentation_alg, str)
        [f.update(l) for f, l in zip(all_chain_features, augment_positions(all_chain_labels, augmentation_alg))]
    else:
        assert decoy_dir is not None
        [f.update(l) for f, l in zip(all_chain_features, load_decoy_positions(sequence_ids, label_ids, decoy_dir))]

    all_chain_features = add_assembly_features(all_chain_features)

    # get labels back from features, as add_assembly_features may alter the order of inputs.
    if label_ids is not None:
        all_chain_labels = [
            {
                k: f[k]
                for k in ["aatype", "all_atom_positions", "all_atom_mask", "resolution"]
            }
            for f in all_chain_features
        ]
    else:
        all_chain_labels = None

    asym_len = np.array([c["seq_length"] for c in all_chain_features], dtype=np.int64)
    if is_monomer:
        all_chain_features = all_chain_features[0]
    else:
        all_chain_features = pair_and_merge(all_chain_features)
        all_chain_features = post_process(all_chain_features)
    all_chain_features["asym_len"] = asym_len

    return all_chain_features, all_chain_labels



def load_and_process(
    config: mlc.ConfigDict,
    mode: str,
    seed: int = 0,
    batch_idx: Optional[int] = None,
    data_idx: Optional[int] = None,
    is_distillation: bool = False,
    **load_kwargs,
):
    try:
        is_monomer = (
            is_distillation
            if "is_monomer" not in load_kwargs
            else load_kwargs.pop("is_monomer")
        )
        features, labels = load(
            **load_kwargs,
            is_monomer=is_monomer,
            train_max_date=config.common.train_max_date,
        )

        features, labels = process(
            config, mode, features, labels, seed, batch_idx, data_idx, is_distillation
        )
        return features, labels
    except Exception as e:
        print("Error loading data", load_kwargs, e)
        raise e

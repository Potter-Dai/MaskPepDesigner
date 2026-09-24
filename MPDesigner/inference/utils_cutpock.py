import os

import pickle, gzip

import datetime

import numpy as np

import torch
import torch.nn.functional as F

from absl import logging

from typing import *

from unicore.utils import tensor_tree_map
from unifold.data import residue_constants as rc
from unifold.data.protein import Protein, to_pdb, PDB_CHAIN_IDS
from unifold.modules.frame import Frame, Rotation

from MPDesigner.data.utils import to_numpy
from MPDesigner.data.data_ops import compute_relative_positions

from copy import deepcopy

from unifold.data import residue_constants as rc
from Bio.PDB import PDBParser
parser = PDBParser(QUIET=True)

n_ca_c_trans = torch.tensor(
    [[-0.5250,  1.3630,  0.0000],
     [ 0.0000,  0.0000,  0.0000],
     [ 1.5260, -0.0000, -0.0000]],
    dtype=torch.float,
)


def compute_atomic_positions(
    frames: torch.Tensor,
    seq_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
):
    frames = Frame.from_tensor_4x4(frames)
    n_ca_c = frames[..., None].apply(n_ca_c_trans.to(frames.device))
    relpos = compute_relative_positions(residue_index, chain_id, cutoff=2)
    is_next_res = (relpos == -1).float()    # [*, L, L]
    next_n_ca_c = torch.einsum(
        "...jad,...ij->...iad", n_ca_c, is_next_res
    )   # [*, L, na=3, d=3]
    next_frame_exist = torch.einsum(
        "...j,...ij->...i", seq_mask, is_next_res
    )   # [*, L]
    oxygen_frames = Frame.from_3_points(
        n_ca_c[..., 1, :],
        n_ca_c[..., 2, :],
        next_n_ca_c[..., 0, :],
    )
    oxygen_coord = oxygen_frames.apply(
        torch.tensor(
            [0.627, -1.062, 0.000],
            dtype=n_ca_c.dtype, device=n_ca_c.device
        )
    )[..., None, :]
    # pseudo_oxygen_coord = frames.apply(
    #     torch.tensor(
    #         [2.153, 1.062, 0.000],
    #         dtype=n_ca_c.dtype, device=n_ca_c.device
    #     )
    # )
    # oxygen_coord = torch.where(
    #     next_frame_exist[..., None] > 0.,
    #     oxygen_coord, pseudo_oxygen_coord,
    # )[..., None, :]
    n_ca_c_o = torch.cat((
        n_ca_c,
        oxygen_coord.new_zeros(oxygen_coord.shape),
        oxygen_coord
    ), dim=-2)
    atom_pos = F.pad(n_ca_c_o, (0, 0, 0, 32))
    atom_mask = torch.stack((
        seq_mask, seq_mask, seq_mask,
        seq_mask.new_zeros(seq_mask.shape),
        seq_mask
    ), dim=-1)
    atom_mask = F.pad(atom_mask, (0, 32))
    return atom_pos, atom_mask


def to_pdb_string(
    aalogits: torch.Tensor,
    frames: torch.Tensor,
    seq_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
    b_factor: torch.Tensor = None,
    model_id: int = 1,
):
    # n_ca_c = n_ca_c_trans.to(frames.device)
    aatype = aalogits[..., :20].argmax(dim=-1)  # L
    atom_pos, atom_mask = compute_atomic_positions(frames, seq_mask, residue_index, chain_id)
    if b_factor is None:
        b_factor = atom_mask
    else:
        assert b_factor.shape == atom_mask.shape

    prot_dict = {
        "atom_positions": atom_pos,
        "aatype": aatype,
        "atom_mask": atom_mask,
        "residue_index": residue_index,
        "chain_index": chain_id - 1,
        "b_factors": b_factor
    }
    has_batch_dim = (len(aalogits.shape) == 3)
    prot_dict = tensor_tree_map(
        lambda x: to_numpy(x, reduce_batch_dim=has_batch_dim),
        prot_dict
    )

    prot = Protein(**prot_dict)
    ret = to_pdb(prot, model_id)
    return ret

def save_pdb(
    path: str,
    aalogits: torch.Tensor,
    frames: torch.Tensor,
    seq_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
    b_factor: torch.Tensor = None,
    model_id: int = 1,
):
    pdb_string = to_pdb_string(
        aalogits,
        frames,
        seq_mask,
        residue_index,
        chain_id,
        b_factor,
        model_id,
    )

    with open(path, 'w') as f:
        f.write(pdb_string)


def save_embedding(path, tensor):
    data = to_numpy(tensor)
    np.save(path, data)

restype_3to1 = rc.restype_3to1
restype_3to1.update({
    "HID": "H",
    "HIE": "H",
})

def load_pdb_ref(name: str, path: str, silent: bool = True):

    prot = parser.get_structure(name, path)  # TODO: separate chains.
    pdb_ref = {}

    for c in prot.get_chains():
        for r in c.get_residues():
            if r.resname not in rc.restype_3to1:
                if not silent:
                    logging.warn(f"{r.resname} is not standard, ignored.")
                continue # ignore non-standard residues
            residue_idx = r.get_id()[1]
            aatype = rc.restype_order[restype_3to1[r.resname]]
            atom_positions = np.zeros([rc.atom_type_num, 3], dtype=float)
            atom_mask = np.zeros([rc.atom_type_num], dtype=bool)
            b_factor = np.zeros([rc.atom_type_num], dtype=float)
            for a in r.get_atoms():
                atom_idx = rc.atom_order.get(a.get_name(), None)
                if atom_idx is not None:
                    atom_positions[atom_idx, :] = a.get_coord()
                    atom_mask[atom_idx] = True
                    b_factor[atom_idx] = a.get_bfactor()
                elif not silent:
                    logging.warn(f"Unknown atom name {a.get_name()}, ignored.")
            pdb_ref[(c.get_id(), residue_idx)] = {
                "aatype": np.array(aatype, dtype=int),
                "all_atom_positions": atom_positions,
                "all_atom_mask": atom_mask,
                "b_factor": b_factor,
            }
    return pdb_ref

PSEUDO_RESIDUE = {
    "aatype": np.array(rc.unk_restype_index, dtype=int),
    "all_atom_positions": np.zeros([rc.atom_type_num, 3], dtype=float),
    "all_atom_mask": np.zeros([rc.atom_type_num], dtype=float),
    "b_factor": np.zeros([rc.atom_type_num], dtype=float),
}


def collate_numpy_dict(list_of_dicts, axis=0):
    if not len(list_of_dicts):
        raise ValueError("cannot collate empty inputs.")
    ret = {}
    for key in list_of_dicts[0].keys():
        ret[key] = np.stack([v[key] for v in list_of_dicts], axis=axis)
    return ret


def build_sample_from_residx(
    residx: list, pdb_ref: dict = None, ignore_missing_residues: bool = False,
):
    def _build_residue(residx, new_resid, new_chainid):
        # 更新为5个元素
        symbol, old_chainid, old_resid, is_ligand, is_hotspot = residx
        if old_chainid is None:
            res = deepcopy(PSEUDO_RESIDUE)
            assert symbol is None, "must not provide symbols for residues to generate."
            res["seq_gen_mask"] = np.array(1.)
            res["frame_gen_mask"] = np.array(1.)
        else:
            assert pdb_ref is not None, "must provide pdb reference if any chain id is specified."
            res = pdb_ref.get((old_chainid, old_resid), None)
            if res is None:
                if not ignore_missing_residues:
                    raise ValueError(f"{old_chainid}{old_resid} is not a valid residue in reference pdb.")
                else:
                    logging.warning(f"{old_chainid}{old_resid} is not a valid residue in reference pdb.")
                    return None
            if is_ligand:
                res["ligmask"] = np.array(0.)
            else:
                res["ligmask"] = np.array(1.)
            if symbol == "$":
                res["seq_gen_mask"] = np.array(1.)
                res["frame_gen_mask"] = np.array(0.)
            elif symbol == "*":
                res["frame_gen_mask"] = np.array(1.)
                res["seq_gen_mask"] = np.array(0.)
            elif symbol == "%":
                res["seq_gen_mask"] = np.array(1.)
                res["frame_gen_mask"] = np.array(1.)
            elif symbol is None:
                res["seq_gen_mask"] = np.array(0.)
                res["frame_gen_mask"] = np.array(0.)
            else:
                raise ValueError(f"unknown symbol {symbol}")
        res["residue_index"] = np.array(new_resid)
        res["chain_id"] = np.array(new_chainid)
        res["hpmask"] = np.array(is_hotspot, dtype=float)
        return res

    if residx is None:
        residues = [
            _build_residue(
                ("%", r[0], r[1]), r[1], PDB_CHAIN_IDS.index(r[0]) + 1
            ) for r in pdb_ref.keys()
        ]
    else:
        residues = []
        for ci, chain_residx in enumerate(residx):
            chain_res = [
                _build_residue(r, ri + 1, ci + 1) for ri, r in enumerate(chain_residx)
            ]
            residues.extend(chain_res)

    residues = [r for r in residues if r is not None]
    sample = collate_numpy_dict(residues)

    # is_ligand = (sample["ligmask"] == 0)
    # sample["residue_index"][is_ligand] += 100
    return sample

def make_pdb_title(
    job_id: str,
    initial_t: float,
    seed: int,
):
    literal = \
f"""
HEADER  MPDESIGNER RESULT {job_id}  {datetime.datetime.now().strftime(r"%Y-%m-%d;%H:%M")}
REMARK   3  INITIAL T: {initial_t:.4f}
REMARK   4  RANDOM SEED: {seed:06d}
"""
    literal = literal.strip().split('\n')
    literal = [l.ljust(80) for l in literal]
    return "\n".join(literal) + "\n"

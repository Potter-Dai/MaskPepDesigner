import argparse
import os
import json
import torch
import time
import tqdm

from absl import logging
from typing import *
from unicore.data.data_utils import numpy_seed
from unifold.modules.frame import Frame
from MPDesigner.nn.mpdesigner import MPDesigner
from MPDesigner.data.process import update_features
from MPDesigner.config_v6 import model_config
from MPDesigner.diffusion.transition_v6 import Diffuser
from MPDesigner.inference import utils, job as J, metrics as M
from MPDesigner.inference.utils import get_pocket_center

logging.set_verbosity("info")

import numpy as np
import torch.nn.functional as F
from SSbuilder.SSbuilder import Structure, read_ssdata, cys_terminal, build_ssbond, merge_s, residues2atoms

def config_and_model(args):
    config = model_config(args.model)
    config.data.frame_augmentation.use_random_rotations = False
    model = MPDesigner(config)

    if args.checkpoint is not None:
        logging.info("start to load params {}".format(args.checkpoint))
        state_dict = torch.load(args.checkpoint)
        if "ema" in state_dict:
            logging.info("ema model exist. using ema.")
            state_dict = state_dict["ema"]["params"]
        else:
            logging.info("no ema model exist. using original params.")
            state_dict = state_dict["model"]
        state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        logging.warning("*** UNRELIABLE RESULTS!!! ***")
        logging.warning("checkpoint not provided. running model with random parameters.")

    model = model.to(args.device)
    model.eval()
    if args.bf16:
        model.bfloat16()

    if args.use_fwd:
        logging.info("use forward activated. model denoising using forward equation.")

    return config, model


def remove_center(*args, mask, eps=1e-12):
    inputs = [Frame.from_tensor_4x4(f) for f in args]
    ref_trans = inputs[0].get_trans()
    ref_center = (ref_trans * mask[..., None]).sum(dim=-2)
    ref_center /= (mask[..., None].sum(dim=-2) + eps)
    outputs = [Frame(f.get_rots(), f.get_trans() - ref_center[..., None, :]) for f in inputs]
    return (o.to_tensor_4x4() for o in outputs)


def add_virtual_CB(s: Structure):
    # Add virtual CB atoms to the Structure object for non-Gly residues
    for res_id in list(s.residues.keys()):
        residue = s.residues[res_id]
        resname = res_id.split('_')[-1]
        if resname == 'G':  # Gly has no CB
            continue

        # Get coords for N, CA, C
        n_coord = None
        ca_coord = None
        c_coord = None
        for atom in residue:
            atom_name = atom[12:16].strip()
            if atom_name == 'N':
                n_coord = get_coord(atom)
            elif atom_name == 'CA':
                ca_coord = get_coord(atom)
            elif atom_name == 'C':
                c_coord = get_coord(atom)

        if n_coord is None or ca_coord is None or c_coord is None:
            logging.warning(f"Missing backbone atoms for {res_id}, skipping CB addition")
            continue

        # Compute virtual CB using ProteinMPNN formula
        b = ca_coord - n_coord
        c = c_coord - ca_coord
        a = np.cross(b, c)
        cb_coord = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_coord

        # Create new atom line for CB, using CA as template
        ca_atom_line = next(a for a in residue if a[12:16].strip() == 'CA')
        cb_atom_line = ca_atom_line[:12] + ' CB ' + ca_atom_line[16:30] + f"{cb_coord[0]:8.3f}{cb_coord[1]:8.3f}{cb_coord[2]:8.3f}" + ca_atom_line[54:76] + ' C\n'

        # Insert CB after CA in the residue's atom list
        ca_idx = next(i for i, a in enumerate(residue) if a[12:16].strip() == 'CA')
        residue.insert(ca_idx + 1, cb_atom_line)

    # Update pdb_atoms, coords, etc.
    s.pdb_atoms = residues2atoms(s.residues)
    s.coords = s.get_pdb_coords()
    s.sort_atoms_by_residue()  # Optional: re-sort if needed
    return s

def get_coord(atom):
    x = float(atom[30:38].strip())
    y = float(atom[38:46].strip())
    z = float(atom[46:54].strip())
    return np.array([x, y, z])


def main(args):
    logging.info("loading jobs ...")
    with open(args.tasks, "r") as f:
        job_configs = json.load(f)

    if isinstance(job_configs, dict):
        jobs = [J.Job(id=k, **v) for k, v in job_configs.items()]
        jobs.sort(key=lambda j: j.id)
    else:
        jobs = [J.Job(id=f"{(i+1):05d}", **v) for i, v in enumerate(job_configs)]

    tic = time.perf_counter()
    config, model = config_and_model(args)
    diffuser = Diffuser(config.data.diffusion)
    gpu_diffuser = Diffuser(config.data.diffusion).to(args.device)
    toc = time.perf_counter() - tic
    logging.info(f"model initialized in {toc:.2f} seconds.")

    for jid, job in enumerate(jobs):
        output_dir = os.path.join(args.outputs, job.id)
        output_traj_dir = os.path.join(output_dir, "traj")
        os.makedirs(output_traj_dir, exist_ok=True)
        with open(os.path.join(output_dir, "task_config.json"), "w") as fp:
            config_dict = job.to_dict()
            config_dict.update({
                "model": args.model,
                "checkpoint": args.checkpoint,
                "use_bf16": args.bf16,
            })
            json.dump(config_dict, fp)

        logging.info(f"running job {jid+1}/{len(jobs)}. job id: {job.id}")

        use_tqdm_for_replica = job.num_replica >= 10 and job.num_steps < 10
        iter_r = (
            tqdm.tqdm(range(job.num_replica), total=job.num_replica)
            if use_tqdm_for_replica
            else range(job.num_replica)
        )

        for replica, seed, initial_t in zip(iter_r, job.seeds, job.initial_t):
            if job.num_replica > 1 and not use_tqdm_for_replica:
                logging.info(f"running replica {replica+1}/{job.num_replica}...")
            batch = J.load_replica_batch(
                pdb=job.pdb,
                contig=job.contig,
                seed=seed,
                init_t=initial_t,
                config=config,
                input_pdb_dir=args.input_pdbs,
                diffuser=diffuser,
                silent=True,
                ignore_missing_residues=job.ignore_missing_residues,
            )
            pocket_center = get_pocket_center(batch)
            if pocket_center is not None:
                pocket_center = pocket_center.to(args.device)
                frames_tensor = batch['frames'].to(args.device)
                frame_gt = Frame.from_tensor_4x4(frames_tensor)
                trans_gt = frame_gt.get_trans() - pocket_center[None, :]
                new_frame_gt = Frame(frame_gt.get_rots(), trans_gt)
                batch['frames'] = new_frame_gt.to_tensor_4x4()

                diffused_frames_tensor = batch['diffused_frames'].to(args.device)
                frame_diffused = Frame.from_tensor_4x4(diffused_frames_tensor)
                trans_diffused = frame_diffused.get_trans() - pocket_center[None, :]
                new_frame_diffused = Frame(frame_diffused.get_rots(), trans_diffused)
                batch['diffused_frames'] = new_frame_diffused.to_tensor_4x4()

            batch = model.convert_input_device(batch, args.device)
            if pocket_center is not None:
                batch = update_features(batch, config.data)
            rep_name = f"{job.id}_rep{replica:03d}_seed{seed:05d}"

            # create denoise time stamps
            time_stamps = torch.linspace(initial_t, 0., job.num_steps+1).float().to(args.device)

            batch_constants = {
                key: batch[key] for key in ("seq_mask", "residue_index", "chain_id")
            }

            utils.save_pdb(
                os.path.join(output_traj_dir, f"input_{rep_name}.pdb"),
                batch["aalogits"], batch["frames"], **batch_constants, b_factor=None
            )
            if job.save_trajectory:
                noised_traj_path = os.path.join(output_traj_dir, f"noised_{rep_name}.pdb")
                predicted_traj_path = os.path.join(output_traj_dir, f"predicted_{rep_name}.pdb")
                with open(noised_traj_path, "w") as f:
                    f.write(utils.make_pdb_title(job.id+" (x_t)", initial_t, seed))
                with open(predicted_traj_path, "w") as f:
                    f.write(utils.make_pdb_title(job.id+" (predicted x_0)", initial_t, seed))

            metrics = []
            iter_t = (
                tqdm.tqdm(range(job.num_steps), total=job.num_steps)
                if job.num_steps >= 10
                else range(job.num_steps)
            )
            s_0, f_0 = batch["aalogits"], batch["frames"]

            for i in iter_t:
                # denoise step
                t, s = time_stamps[i], time_stamps[i+1]
                assert batch["diffusion_t"] == t
                s_t, f_t = batch["diffused_aalogits"], batch["diffused_frames"]
                with torch.no_grad():
                    out = model(batch)
                sh_0, fh_0 = out["aalogits"], out["frames"][-1]     # use last frame
                b_factor = out["plddt"][..., None].tile(37)

                # saving intermediate results
                if job.save_trajectory:
                    with open(noised_traj_path, "a") as f:
                        f.write(utils.to_pdb_string(s_t, f_t, **batch_constants, model_id=i+1,))
                    with open(predicted_traj_path, "a") as f:
                        f.write(utils.to_pdb_string(sh_0, fh_0, **batch_constants, model_id=i+1,))

                if job.save_embedding != "none":
                    single_repr, pair_repr = out["single"], out["pair"]
                    utils.save_embedding(
                        os.path.join(output_dir, f"{rep_name}_emb_single_{i:03d}_{t:.4f}".replace(".", ",") + ".npy"),
                        single_repr,
                    )
                    if job.save_embedding == "single+pair":
                        utils.save_embedding(
                            os.path.join(output_dir, f"{rep_name}_emb_pair_{i:03d}_{t:.4f}".replace(".", ",") + ".npy"),
                            pair_repr,
                        )

                with torch.no_grad():
                    with numpy_seed(seed, i, key="denoise"):
                        if args.use_fwd:
                            denoise_fn = gpu_diffuser.denoise_fwd
                        elif args.use_ode:
                            raise NotImplementedError("use ode not ready currently.")
                            denoise_fn = gpu_diffuser.ode_denoise
                        else:
                            denoise_fn = gpu_diffuser.denoise
                        s_s, f_s = denoise_fn(
                            s_t, f_t, sh_0, fh_0,
                            batch["seq_gen_mask"], batch["frame_gen_mask"],
                            t=t, s=s,
                        )

                if job.calc_metrics:
                    metrics.append(M.eval_metrics(
                        sh_0, fh_0, batch["aalogits"], batch["frames"],
                        batch["seq_mask"], batch["frame_mask"],
                        t, out["plddt"],
                    ))

                batch["diffused_aalogits"] = s_s
                batch["diffused_frames"] = f_s
                batch["diffusion_t"] = s
                batch = update_features(batch, config.data)

            sh_0 = torch.where(
                batch["seq_gen_mask"][..., None] > 0,
                sh_0,
                batch["aalogits"].to(sh_0.dtype),
            )
            total_offset = torch.zeros(3, dtype=fh_0.dtype, device=fh_0.device)
            if pocket_center is not None:
                total_offset = total_offset + pocket_center.to(device=fh_0.device, dtype=fh_0.dtype)
            if "initial_center" in batch and batch["initial_center"] is not None:
                total_offset = total_offset + batch["initial_center"].to(device=fh_0.device, dtype=fh_0.dtype)
            if torch.any(total_offset != 0):
                logging.info("Translating final structure back to original coordinates.")
                frame_final = Frame.from_tensor_4x4(fh_0)
                trans_final = frame_final.get_trans() + total_offset[None, :]
                new_frame_final = Frame(frame_final.get_rots(), trans_final)
                fh_0 = new_frame_final.to_tensor_4x4()

            # save final results
            utils.save_pdb(
                os.path.join(output_dir, f"out_{rep_name}.pdb"),
                sh_0, fh_0, **batch_constants, b_factor=b_factor
            )

            # 计算 entropy (no batch dim, so no[0])
            final_aa_logits = out["aalogits"]  # [L, num_aa_types]
            final_aa_prob = F.softmax(final_aa_logits, dim=-1)
            final_aa_prob = torch.clamp(final_aa_prob, min=1e-10)
            final_aa_entropy = -torch.sum(final_aa_prob * torch.log(final_aa_prob), dim=-1)  # [L]

            entropy = final_aa_entropy.cpu().numpy()  # [L]

            # 获取seq_mask (peptide vs. target) 和seq_gen_mask (real mask vs padding)
            seq_gen_mask = batch["seq_gen_mask"].cpu().numpy()
            seq_mask = batch["seq_mask"].cpu().numpy()
            # 正确的肽链mask是两者的交集，确保只选择真实的、属于肽链的残基
            peptide_mask = (seq_gen_mask * seq_mask).astype(bool)
            # 现在使用修正后的mask来提取熵
            peptide_entropy = entropy[peptide_mask]
            # print(peptide_mask)

            # 保存 entropy 方便调试
            # np.save(os.path.join(output_dir, f"{rep_name}_entropy.npy"), peptide_entropy)
            # np.save(os.path.join(output_dir, f"{rep_name}_full_entropy.npy"), entropy)

            # SSbond builder
            if args.ss_build:
                pdb_path = os.path.join(output_dir, f"out_{rep_name}.pdb")
                lig_chain = args.lig_chain

                # load peptide Structure
                peptide_structure = Structure(pdb_path, chains=lig_chain)

                # add virtual CB atoms
                peptide_structure = add_virtual_CB(peptide_structure)
                cb_pdb_path = pdb_path.replace(".pdb", "_CB.pdb")
                peptide_structure.dump(cb_pdb_path)
                logging.info(f"Virtual CB atoms added and saved to {cb_pdb_path}")

                peptide_len = len(peptide_structure.seq)
                if len(peptide_entropy) != peptide_len:
                    logging.warning(f"Mismatch: entropy len {len(peptide_entropy)} != PDB seq len {peptide_len}. Assuming peptide at first.")
                    peptide_entropy = entropy[:peptide_len]
                    assert len(peptide_entropy) == peptide_len, "Entropy extraction failed after adjustment."

                print(f"Full entropy length: {len(entropy)}")
                print(f"Peptide mask sum: {peptide_mask.sum()}")
                print(f"Peptide entropy length: {len(peptide_entropy)}")
                print(f"Peptide PDB sequence length: {peptide_len}")
                print(f"Peptide PDB sequence: {peptide_structure.seq}")

                ss_frag_lib = read_ssdata(args.ss_frag_lib)

                # check Cys terminal (ss_prob=peptide_entropy, bf_th=threshold)
                if cys_terminal(peptide_structure, peptide_entropy, args.ss_entropy_threshold):
                    new_peptide = build_ssbond(peptide_structure, ss_frag_lib, args.ss_max_bonds, lig_chain)
                    if new_peptide is not None:
                        receptor_chains = ''.join([c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c != lig_chain])
                        receptor_structure = Structure(pdb_path, chains=receptor_chains)
                        ss_pdb_path = pdb_path.replace(".pdb", "_ss.pdb")
                        merge_s(receptor_structure, new_peptide, ss_pdb_path)
                        logging.info(f"SS bonds built and saved to {ss_pdb_path}")
                    else:
                        logging.info(f"Failed to build SS bonds for {pdb_path}")
                else:
                    logging.info(f"No suitable Cys terminals for SS bonds in {pdb_path}")

            if job.calc_metrics:
                metrics = utils.collate_numpy_dict(metrics)
                M.save_metrics(os.path.join(output_traj_dir, f"{rep_name}_metrics.json"), metrics)
                M.plot_metrics(os.path.join(output_traj_dir, f"{rep_name}_metrics.png"), metrics)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path arguments
    parser.add_argument(
        "-t", "--tasks", type=str, default=None, required=True,
        help=\
r"""
path to a json file specifying the task to conduct.
The json file should be organized as:
{
    "tid1": {"kw1": arg1, "kw2": arg2, ...},
    "tid2": {"kw1": arg1, "kw2", arg2, ...},
    ...
}.
The task id `tid` should contain only letters, digits, and `_`.
The keyword arguments are illustrated in MPDESIGNER_README.md.
"""
    )
    parser.add_argument(
        "-i", "--input_pdbs", type=str, default=None,
        help="directory of input reference pdbs."
    )
    parser.add_argument(
        "-o", "--outputs", type=str, default=None, required=True,
        help="directory of outputs."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None, required=True,
        help="path to model checkpoint."
    )
    parser.add_argument(
        "-m", "--model", type=str, default=None, required=True,
        help="the name of the model."
    )

    # SSbond building arguments
    default_ss_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SSbuilder", "SSBLIB")
    parser.add_argument("--ss_build", action="store_true", help="If set, build SS bonds after generation.")
    parser.add_argument("--ss_frag_lib", type=str, default=default_ss_lib, help="Path to SS fragment library directory.")
    parser.add_argument("--ss_entropy_threshold", type=float, default=0.5, help="Entropy threshold for Cys terminal check.")
    parser.add_argument("--ss_max_bonds", type=int, default=1, help="Max number of SS bonds to build.")
    parser.add_argument("--lig_chain", type=str, default="B", help="Chain ID for the peptide binder (default: B for MPDesigner).")

    # model arguments
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="the device to run the model. Please at any chance use GPUs. (default: `cuda:0`)."
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="if set, then bfloat16 is used. (default: not set)."
    )
    parser.add_argument(
        "--use-fwd", action="store_true",
        help="if set, use forward scheme to mimic a reversed sde."
    )
    parser.add_argument(
        "--use-ode", action="store_true",
        help="if set, use probability flow ode to denoise."
    )
    args = parser.parse_args()
    main(args)

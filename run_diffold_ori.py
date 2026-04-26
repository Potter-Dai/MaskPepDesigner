# from unifold.inference import *
import argparse
import os
import json
import torch
import time

from absl import logging

from typing import *

from unicore.data.data_utils import numpy_seed

from unifold.modules.frame import Frame

from diffold.nn.dfv6 import Dfv6
from diffold.data.process import update_features
from diffold.config_v6 import model_config
from diffold.diffusion.transition_v6 import Diffuser

import tqdm

from diffold.inference import utils, job as J, metrics as M

logging.set_verbosity("info")


def config_and_model(args):
    config = model_config(args.model)
    config.data.frame_augmentation.use_random_rotations = False
    model = Dfv6(config)

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
            batch = model.convert_input_device(batch, args.device)
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
                fh_0, f_t = remove_center(fh_0, f_t, mask=batch["frame_mask"])

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

            # save final results
            utils.save_pdb(
                os.path.join(output_dir, f"out_{rep_name}.pdb"),
                sh_0, fh_0, **batch_constants, b_factor=b_factor
            )
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
The keyword arguments are illustrated in DIFFOLD_README.md.
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

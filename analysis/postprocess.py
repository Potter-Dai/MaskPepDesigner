import os
import shutil
from analysis.amber_minimize import AmberRelaxation
from Bio.PDB import *
from pdbfixer import PDBFixer

os.environ["ABSL_CPP_MIN_LOG_LEVEL"] = "2"
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

from openmm.app import PDBFile
from analysis.postprocess_utils import PCLIO
from analysis.utils import write_from_string
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.protocols.rosetta_scripts import RosettaScriptsParser
from pyrosetta.toolbox import py_jobdistributor as jd


RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

P = PDBParser(QUIET=1)
PI = PCLIO()


def fix_pdb(pdb, out_file):
    fixer = PDBFixer(pdb)
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7)
    PDBFile.writeFile(
        fixer.topology, fixer.positions, open(out_file, "w"), keepIds=True
    )


def check_GLY_CB(entity):
    for res in entity.get_residues():
        if res.resname == "GLY" and res.has_id("CB"):
            res.detach_child("CB")
    return entity


class Postprocess:
    def __init__(
        self,
        pdb_file: str,
        lig_chain_id: str,
        ori_dir: str,
        out_dir: str,
        xml=None,
        amber_relax=True,
        rosetta_relax=True,
        do_reconstruct=False,
    ):
        self.pdb_file = pdb_file
        self.lig_chain_id = lig_chain_id
        self.ori_dir = ori_dir
        self.out_dir = out_dir
        self.postprocess_dir = os.path.join(out_dir, "postprocess_results")
        os.makedirs(self.postprocess_dir, exist_ok=True)

        self.pdb_string_relaxed = None
        self.pdb_string_ori_merged = None
        self.data_id = os.path.splitext(os.path.basename(self.pdb_file))[0]
        self.recon_file = os.path.join(self.postprocess_dir, f"{self.data_id}_recon.pdb")
        self.fixed_file = os.path.join(self.postprocess_dir, f"{self.data_id}_fixed.pdb")
        self.do_reconstruct = do_reconstruct
        if self.do_reconstruct:
            self.ori_file = self.find_ori_file()
        else:
            try:
                self.ori_file = self.find_ori_file()
            except FileNotFoundError:
                self.ori_file = self.pdb_file
        self.xml = xml
        self.amber_relax = amber_relax
        self.rosetta_relax = rosetta_relax

    def find_ori_file(self):
        """
        Locate the matching native receptor PDB in self.ori_dir:
        1. Check task_config.json in the generated PDB's directory (or parent) for exact "pdb" key.
        2. Iteratively strip trailing "_..." segments (and optional leading "out_").
        3. If reduced to a PDB ID prefix (>=4 chars), match unique "<prefix>_*.pdb" in self.ori_dir.
        """
        if not self.ori_dir or not os.path.isdir(self.ori_dir):
            raise FileNotFoundError(f"Original PDB directory {self.ori_dir!r} does not exist")

        # 1. Exact lookup from task_config.json written by MPDesigner
        gen_dir = os.path.dirname(os.path.abspath(self.pdb_file))
        for cfg_dir in (gen_dir, os.path.dirname(gen_dir)):
            task_cfg_path = os.path.join(cfg_dir, "task_config.json")
            if os.path.isfile(task_cfg_path):
                try:
                    import json
                    with open(task_cfg_path, "r") as f:
                        tcfg = json.load(f)
                    pdb_key = tcfg.get("pdb")
                    if pdb_key:
                        cand = os.path.join(self.ori_dir, f"{pdb_key}.pdb")
                        if os.path.isfile(cand):
                            return cand
                except Exception:
                    pass

        # 2. Iterative prefix stripping
        prefixes = [self.data_id]
        if self.data_id.startswith("out_"):
            prefixes.append(self.data_id[4:])
        for base_prefix in prefixes:
            prefix = base_prefix
            while True:
                candidate = os.path.join(self.ori_dir, f"{prefix}.pdb")
                if os.path.isfile(candidate):
                    return candidate
                if "_" not in prefix:
                    if len(prefix) >= 4:
                        import glob
                        matches = sorted(glob.glob(os.path.join(self.ori_dir, f"{prefix}_*.pdb")))
                        if len(matches) == 1:
                            return matches[0]
                    break
                prefix = prefix.rsplit("_", 1)[0]

        raise FileNotFoundError(f"No original PDB matching '{self.data_id}' found in {self.ori_dir!r}")

    def _resolve_lig_chain_id(self, model):
        chain_ids = [c.id for c in model.get_chains()]
        if not chain_ids:
            raise ValueError(f"No chains found in {self.pdb_file}")
        if self.lig_chain_id and self.lig_chain_id != "auto" and self.lig_chain_id in chain_ids:
            return self.lig_chain_id
        # Fallback to physical shortest-chain rule
        shortest = min(model.get_chains(), key=lambda c: len(list(c.get_residues())))
        return shortest.id

    def _normalize_and_fix(self, lig_chain_obj, rec_chain_objs):
        """
        Normalize peptide chain to 'A' and receptor chain(s) to 'B', 'C', ...
        so Rosetta interface_analyze.xml (<Chain name="CYC" chains="A"/>, ligandchain="A")
        always scores the peptide as ligand and receptor as target.
        """
        lig_copy = lig_chain_obj.copy()
        lig_copy.detach_parent()
        lig_copy.id = "A"

        avail_ids = iter("BCDEFGHIJKLMNOPQRSTUVWXYZ")
        used_ids = {"A"}
        rec_copies = []
        for rc_obj in rec_chain_objs:
            rc_copy = rc_obj.copy()
            rc_copy.detach_parent()
            if rc_copy.id in used_ids:
                for cand in avail_ids:
                    if cand not in used_ids:
                        rc_copy.id = cand
                        break
            used_ids.add(rc_copy.id)
            rec_copies.append(rc_copy)

        ents = [lig_copy] + rec_copies
        new_ents = [check_GLY_CB(ent) for ent in ents]
        PI.set_structure_multiple(new_ents)
        PI.save(self.recon_file)
        fix_pdb(self.recon_file, self.fixed_file)
        self.pdb_file = self.fixed_file

    def reconstruct(self):
        s_pcl = P.get_structure("s_pcl", self.pdb_file)[0]
        s_ori = P.get_structure("s_ori", self.ori_file)[0]
        lig_id = self._resolve_lig_chain_id(s_pcl)
        lig_chain = s_pcl[lig_id]
        rec_ori = [c for c in s_ori.get_chains() if c.id != lig_id]
        if not rec_ori:
            rec_ori = [c for c in s_ori.get_chains()]
        self._normalize_and_fix(lig_chain, rec_ori)

    def prepare_without_reconstruct(self):
        s_pcl = P.get_structure("s_pcl", self.pdb_file)[0]
        lig_id = self._resolve_lig_chain_id(s_pcl)
        lig_chain = s_pcl[lig_id]
        rec_chains = [c for c in s_pcl.get_chains() if c.id != lig_id]
        self._normalize_and_fix(lig_chain, rec_chains)

    def relax(self):
        ABRlx = AmberRelaxation(
            max_iterations=RELAX_MAX_ITERATIONS,
            tolerance=RELAX_ENERGY_TOLERANCE,
            max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
            use_gpu=False
        )
        self.pdb_string_relaxed, _, = ABRlx.process(open(self.pdb_file).read())
        relaxed_file = os.path.join(self.postprocess_dir, f"{self.data_id}_amber_relaxed.pdb")
        write_from_string(self.pdb_string_relaxed, relaxed_file)
        self.pdb_file = relaxed_file

    def interface_analyze(self):
        init(
            "-mute all "
            "-ignore_zero_occupancy false "
            f"-in:file:native {self.fixed_file} "
            f"-parser:protocol {self.xml}"
        )

        pose = pose_from_pdb(self.pdb_file)
        mover = RosettaScriptsParser().generate_mover(self.xml)
        mover.apply(pose)
        relaxed_file = os.path.join(self.postprocess_dir, f"{self.data_id}_rosetta_relaxed.pdb")
        pose.dump_pdb(relaxed_file)
        self.pdb_file = relaxed_file

        scorefxn  = get_fa_scorefxn()
        scorefxn(pose)
        jd.output_scorefile(
            pose=pose,
            pdb_name=self.ori_file.replace(".pdb", ""),
            current_name=self.data_id,
            scorefxn=scorefxn,
            nstruct=1,
            scorefilepath=os.path.join(self.postprocess_dir, "rosetta_score.sc"),
            json_format=True
        )

    def __call__(self):
        try:
            if self.do_reconstruct:
                self.reconstruct()
            else:
                self.prepare_without_reconstruct()
        except Exception as e:
            print(f"Skipping {self.pdb_file} due to error during preparation/reconstruction: {e}")
            return
        if self.amber_relax:
            try:
                self.relax()
            except Exception as e:
                print(f"Skipping Amber relaxation due to error: {e}")
        if self.rosetta_relax:
            try:
                self.interface_analyze()
            except Exception as e:
                print(f"Skipping Rosetta relaxation due to error: {e}")
        shutil.copy(self.pdb_file, os.path.join(self.out_dir, f"{self.data_id}_final.pdb"))

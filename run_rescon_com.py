import argparse
import os
import numpy as np
import glob
from SSbuilder.SSbuilder import Structure, SVDSuperimposer

def find_matching_original_pdb(gen_base: str, orig_dir: str, gen_dir: str = None) -> str:
    """
    Find the matching native receptor PDB in orig_dir for a generated file name
    such as 'out_8qza_denovo_s50_rep000_seed12345.pdb' or '8qza_rep000.pdb'.
    """
    if gen_dir:
        for cfg_dir in (gen_dir, os.path.dirname(gen_dir)):
            task_cfg_path = os.path.join(cfg_dir, "task_config.json")
            if os.path.isfile(task_cfg_path):
                try:
                    import json
                    with open(task_cfg_path, "r") as f:
                        tcfg = json.load(f)
                    pdb_key = tcfg.get("pdb")
                    if pdb_key:
                        cand = os.path.join(orig_dir, f"{pdb_key}.pdb")
                        if os.path.isfile(cand):
                            return cand
                except Exception:
                    pass

    gen_name, _ = os.path.splitext(gen_base)
    candidates = [gen_name]
    if gen_name.startswith("out_"):
        candidates.append(gen_name[4:])
    for base in candidates:
        prefix = base
        while True:
            cand_path = os.path.join(orig_dir, f"{prefix}.pdb")
            if os.path.isfile(cand_path):
                return cand_path
            if "_" not in prefix:
                if len(prefix) >= 4:
                    matches = sorted(glob.glob(os.path.join(orig_dir, f"{prefix}_*.pdb")))
                    if len(matches) == 1:
                        return matches[0]
                break
            prefix = prefix.rsplit("_", 1)[0]
    return None


def reconstruct_complex(
    original_pdb_path: str,
    generated_pdb_path: str,
    output_pdb_path: str,
    peptide_chain_id: str = 'B',
    original_receptor_chain: str = None,
    replace_receptor: bool = False,
):
    """
    Aligns the generated incomplete protein-peptide complex to the full protein
    and saves the aligned generated complex PDB.

    Args:
        original_receptor_chain: 原始 PDB 中 receptor 的链名。如果提供，
            则只加载该链作为参考（避免 peptide 链残基编号冲突）。
            如果为 None，则加载整个原始 PDB。
        replace_receptor: 若为 True，输出中使用原始完整 receptor 替换生成文件中的截断 receptor。
    """
    print(f"Processing: original={original_pdb_path}, generated={generated_pdb_path}")
    print("Loading structures...")
    if original_receptor_chain:
        full_receptor = Structure(original_pdb_path, chains=original_receptor_chain)
    else:
        full_receptor = Structure(original_pdb_path)
    generated_complex = Structure(generated_pdb_path)

    generated_peptide_atoms = []
    generated_receptor_atoms = []
    for atom in generated_complex.pdb_atoms:
        if atom[21] == peptide_chain_id:
            generated_peptide_atoms.append(atom)
        else:
            generated_receptor_atoms.append(atom)
    
    temp_generated_receptor = Structure(generated_pdb_path, atoms=generated_receptor_atoms)

    print("Identifying common backbone atoms as anchors...")
    # 2. 识别共同的骨架原子作为锚点
    #    生成的 PDB 强制 protein=A, peptide=B，但原始 PDB 链名可能不同
    #    优先用 chain-agnostic 的 (resid, aa) 匹配；若因重编号匹配不足，则回退到序列比对映射
    ref_coords = []      # 来自完整蛋白的锚点原子坐标 (参考)
    mobile_coords = []   # 来自不完整蛋白的锚点原子坐标 (移动)
    
    backbone_atoms = ['N', 'CA', 'C', 'O']

    def strip_chain(res_key):
        """'B_7_G' -> '7_G'"""
        parts = res_key.split('_', 1)
        return parts[1] if len(parts) > 1 else res_key

    full_res_by_nochain = {}
    for rk in full_receptor.residues.keys():
        nochain = strip_chain(rk)
        if nochain not in full_res_by_nochain:
            full_res_by_nochain[nochain] = rk

    for gen_res_key in temp_generated_receptor.residues.keys():
        nochain_key = strip_chain(gen_res_key)
        if nochain_key not in full_res_by_nochain:
            continue
        orig_res_key = full_res_by_nochain[nochain_key]
        for atom_name in backbone_atoms:
            try:
                mobile_coord_key = f"{gen_res_key}_{atom_name}"
                ref_coord_key = f"{orig_res_key}_{atom_name}"
                mobile_coords.append(temp_generated_receptor.coords[mobile_coord_key])
                ref_coords.append(full_receptor.coords[ref_coord_key])
            except KeyError:
                continue

    if len(ref_coords) < 12:
        # Fallback: map residues between generated receptor and full receptor by exact AA subsequence/alignment
        from Bio.Align import PairwiseAligner
        gen_keys = list(temp_generated_receptor.residues.keys())
        full_keys = list(full_receptor.residues.keys())
        gen_seq = "".join(k.split("_")[-1] for k in gen_keys)
        full_seq = "".join(k.split("_")[-1] for k in full_keys)
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -5.0
        aligner.open_gap_score = -2.0
        aligner.extend_gap_score = -0.1
        alns = aligner.align(full_seq, gen_seq)
        if len(alns) > 0:
            ref_coords, mobile_coords = [], []
            aln = alns[0]
            for (f_start, f_end), (g_start, g_end) in zip(*aln.aligned):
                for f_idx, g_idx in zip(range(f_start, f_end), range(g_start, g_end)):
                    if full_seq[f_idx] != gen_seq[g_idx]:
                        continue
                    orig_res_key = full_keys[f_idx]
                    gen_res_key = gen_keys[g_idx]
                    for atom_name in backbone_atoms:
                        mobile_coord_key = f"{gen_res_key}_{atom_name}"
                        ref_coord_key = f"{orig_res_key}_{atom_name}"
                        if mobile_coord_key in temp_generated_receptor.coords and ref_coord_key in full_receptor.coords:
                            mobile_coords.append(temp_generated_receptor.coords[mobile_coord_key])
                            ref_coords.append(full_receptor.coords[ref_coord_key])

    if len(ref_coords) < 3:
        raise ValueError(f"Found only {len(ref_coords)} common anchor atoms. At least 3 are required for stable alignment. Check chain IDs and residue numbering.")

    print(f"Found {len(ref_coords)} common anchor atoms for alignment.")
    ref_coords = np.array(ref_coords)
    mobile_coords = np.array(mobile_coords)

    print("Calculating superposition transform...")
    # 3. 计算变换
    imposer = SVDSuperimposer()
    imposer.set(ref_coords, mobile_coords) # set(reference, mobile)
    imposer.run()
    rms = imposer.get_rms()
    print(f"RMSD after alignment of anchor points: {rms:.4f} Å")

    rot, tran = imposer.get_rotran()

    print("Applying transform to the generated structure...")
    # 4. 应用变换
    all_generated_coords = generated_complex.get_pdb_coords()
    all_generated_coords_matrix = np.array(list(all_generated_coords.values()))

    transformed_coords_matrix = np.dot(all_generated_coords_matrix, rot) + tran

    generated_complex.update_pdb_coords_full(transformed_coords_matrix)

    print("Saving aligned generated structure...")
    # 5. 保存变换后的generated_complex
    if replace_receptor:
        pep_atoms = [a for a in generated_complex.pdb_atoms if a[21] == peptide_chain_id]
        rec_atoms = [a for a in full_receptor.pdb_atoms if a[21] != peptide_chain_id]
        final_pdb_atoms = rec_atoms + pep_atoms
    else:
        final_pdb_atoms = generated_complex.pdb_atoms

    with open(output_pdb_path, 'w') as f:
        if generated_complex.title:
            f.writelines(generated_complex.title)
        for atom_line in final_pdb_atoms:
            f.write(atom_line.strip() + '\n')
        f.write("END\n")

    print(f"Successfully aligned and saved to {output_pdb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align a generated incomplete complex to a full receptor and save the aligned generated structure.")
    parser.add_argument("--original", "-i", required=True, help="Path to the original, full PDB file or directory of the receptor(s).")
    parser.add_argument("--generated", "-g", required=True, help="Path to the cropped PDB file or directory generated by the model.")
    parser.add_argument("--output", "-o", required=True, help="Path for the output aligned PDB file or directory.")
    parser.add_argument("--peptide_chain", "-pc", default="B", help="Chain ID of the peptide in the generated file (default: B).")
    parser.add_argument("--replace_receptor", "-rr", action="store_true", help="Replace cropped receptor with full original receptor in output PDB.")
    
    args = parser.parse_args()

    if os.path.isdir(args.original) and os.path.isdir(args.generated):
        # Batch mode: process directories
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        gen_pdbs = sorted(glob.glob(os.path.join(args.generated, '*.pdb')))
        for gen_pdb in gen_pdbs:
            gen_base = os.path.basename(gen_pdb)
            gen_name, _ = os.path.splitext(gen_base)
            orig_path = find_matching_original_pdb(gen_base, args.original, gen_dir=args.generated)
            if orig_path is None:
                print(f"Skipping {gen_base}: no matching original PDB found in {args.original}.")
                continue
            output_file = gen_name + '_align.pdb'
            output_path = os.path.join(args.output, output_file)
            reconstruct_complex(
                original_pdb_path=orig_path,
                generated_pdb_path=gen_pdb,
                output_pdb_path=output_path,
                peptide_chain_id=args.peptide_chain,
                replace_receptor=args.replace_receptor,
            )
    else:
        # Single mode: assume files
        if os.path.isdir(args.original) or os.path.isdir(args.generated):
            raise ValueError("Mixed input types: both --original and --generated must be files or both must be directories.")
        reconstruct_complex(
            original_pdb_path=args.original,
            generated_pdb_path=args.generated,
            output_pdb_path=args.output,
            peptide_chain_id=args.peptide_chain,
            replace_receptor=args.replace_receptor,
        )

"""
Script for postprocessing procedures.
"""
import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

import os
import argparse
from datetime import datetime
from analysis.metrics import postprocess_metric_parallel

now = datetime.now()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run postprocessing procedures."
    )

    parser.add_argument(
        "--in_pdbs",
        type=str,
        required=True,
        help="Path to the directory containing the test PDB files."
    )

    parser.add_argument(
        "--ori_pdbs",
        type=str,
        default=f"{root}/data/receptor_data/receptor_raw_data",
        help="Path to the directory containing native pdb complexes."
    )
    
    parser.add_argument(
        "--nproc",
        type=int,
        default=os.cpu_count()-1,
        help="Number of processing cores to use."
    )

    parser.add_argument(
        "--postprocess_xml_path",
        type=str,
        default=f"{root}/analysis/interface_analyze.xml",
        help="Path to the XML file containing the postprocessing protocol."
    )

    parser.add_argument(
        "--amber_relax",
        action='store_true',
        help="Relax the structures using Amber force field."
    )

    parser.add_argument(
        "--rosetta_relax",
        action='store_true',
        help="Relax the structures using Rosetta FastRelax."
    )

    parser.add_argument(
        "--do_reconstruct",
        action='store_true',
        help="Reconstruct full complex by combining generated peptide with native receptor from --ori_pdbs."
    )

    parser.add_argument(
        "--lig_chain",
        type=str,
        default="B",
        help="Chain ID of the peptide ligand in input PDBs (default: 'B', or 'auto' for shortest chain)."
    )

    return parser


def main(args):
    test_files = []
    for walk_root, dirs, files in os.walk(args.in_pdbs):
        dirs[:] = [d for d in dirs if d not in ("postprocess_results", "traj")]
        for file in sorted(files):
            if not file.endswith('.pdb'):
                continue
            if file.endswith('_final.pdb') or file.startswith(('input_', 'noised_', 'predicted_')):
                continue
            test_files.append(os.path.join(walk_root, file))
    
    print(f'Start postprocessing of {len(test_files)} peptides...')
    postprocess_metric_parallel(
        files=test_files,
        ori_dir=args.ori_pdbs,
        nproc=args.nproc,
        lig_chain_ids=[args.lig_chain for _ in test_files],
        xml=args.postprocess_xml_path,
        out_path=os.path.join(args.in_pdbs, 'postprocess_results.csv'),
        amber_relax=args.amber_relax,
        rosetta_relax=args.rosetta_relax,
        do_reconstruct=args.do_reconstruct,
    )
    print(f'Finished postprocessing of {len(test_files)} peptides.')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)


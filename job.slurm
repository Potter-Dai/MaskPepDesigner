#!/bin/bash
 
#SBATCH -J ftv66-fixtask
#SBATCH -p gpu-a100-Partition
#SBATCH -N 1
#SBATCH --time 72:00:00
#SBATCH -o /flashfs1/work/btdai/projects/ufcpdiff_cutpock_fixtask/std.out.%j
#SBATCH -e /flashfs1/work/btdai/projects/ufcpdiff_cutpock_fixtask/std.err.%j
#SBATCH --gres=gpu:8

module add gcc/11.2.0
source ~/.bashrc
conda activate ufconf

work_dir="/flashfs1/work/btdai/projects/ufcpdiff_cutpock_fixtask/"
cd $work_dir

sh finetune_MPDesigner_peptide.sh /path/to/cpset_lmdb \
/path/to/save_dir \
/path/to/pretrained_checkpoint.pt denovo > finetune_mpdesigner.log



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

sh finetune_diffold_noema.sh /flashfs1/scratch.global/btdai/traineval_dbt_multimer4CP \
/flashfs1/scratch.global/btdai/diffold_train/CPset_v67_ftv66_denovo_crop10A_fixtask_bz8 \
../ckps/v66_ftv65/checkpoint_1_79000.pt denovo > finetune_v66_denovo_10A_fixtask_bz8.log



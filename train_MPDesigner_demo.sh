#!/usr/bin/env bash
if [ "$#" -ne 5 ]; then
    echo "Usage: bash $0 UFDB DISTILL_DB ESM_DB SAVE_DIR MODEL_NAME" >&2
    exit 2
fi

[ -z "${MASTER_PORT}" ] && MASTER_PORT=1314
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${update_freq}" ] && update_freq=8
[ -z "${total_step}" ] && total_step=10000
[ -z "${warmup_step}" ] && warmup_step=100
[ -z "${decay_step}" ] && decay_step=1000
[ -z "${decay_ratio}" ] && decay_ratio=0.95
[ -z "${lr}" ] && lr=1e-3
[ -z "${seed}" ] && seed=31

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT
echo "ufdb" $1
echo "uf_distill" $2
echo "esmdb" $3
echo "save_dir" $4
echo "decay_step" $decay_step
echo "warmup_step" $warmup_step
echo "decay_ratio" $decay_ratio
echo "lr" $lr
echo "total_step" $total_step
echo "update_freq" $update_freq
echo "seed" $seed
echo "create folder for save"
mkdir -p "$4"
echo "start training"

model_name=$5

tmp_dir=$(mktemp -d) || exit 1
trap 'rm -rf -- "$tmp_dir"' EXIT

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $MASTER_PORT \
       $(which unicore-train) --user-dir MPDesigner \
       --unifold-db-path $1 --uf-distill-db-path $2 --esmatlas-db-path $3 \
       --num-workers 2 --ddp-backend=no_c10d \
       --task MPDesigner --loss MPDesigner --arch MPDesigner_v6 \
       --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0  --per-sample-clip-norm 0.1 --allreduce-fp32-grad  \
       --lr-scheduler exponential_decay --lr $lr --warmup-updates $warmup_step --decay-ratio $decay_ratio --decay-steps $decay_step --stair-decay \
       --update-freq $update_freq --batch-size 1 --seed $seed  --tensorboard-logdir $4/tsb/ \
       --max-update $total_step --max-epoch 1 --log-interval 1 --log-format simple \
       --valid-subset "eval_fold,eval_invfold,eval_denovo" --valid-samples-per-step 4 \
       --save-interval-updates 200 --validate-interval-updates 100 --keep-interval-updates 5 --no-epoch-checkpoints  \
       --save-dir $4 --tmp-save-dir $tmp_dir --required-batch-size-multiple 1 --ema-decay 0.999 --data-buffer-size 32 --model-name $model_name --fixed-validation-seed 38

training_status=$?
exit "$training_status"

[ -z "${MASTER_PORT}" ] && MASTER_PORT=10087
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${update_freq}" ] && update_freq=1
[ -z "${batch_size}" ] && batch_size=1
[ -z "${total_step}" ] && total_step=300000
[ -z "${warmup_step}" ] && warmup_step=1000
[ -z "${decay_step}" ] && decay_step=10000
[ -z "${decay_ratio}" ] && decay_ratio=0.95
[ -z "${lr}" ] && lr=1e-4
[ -z "${seed}" ] && seed=31
[ -z "${valid_step}" ] && valid_step=500
[ -z "${save_step}" ] && save_step=1000
[ -z "${ckpt_ratio}" ] && ckpt_ratio=1.0
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0
[ -z "${load_from_ema}" ] && load_from_ema="--load-from-ema" # --load-from-ema

export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT
echo "ufdb" $1
echo "save_dir" $2
echo "decay_step" $decay_step
echo "warmup_step" $warmup_step
echo "decay_ratio" $decay_ratio
echo "lr" $lr
echo "total_step" $total_step
echo "update_freq" $update_freq
echo "seed" $seed
echo "create folder for save"
mkdir -p $2
echo "start training"

OPTION=""
if [ -f "$2/checkpoint_last.pt" ]; then
    echo "ckp exists."
else
  echo "finetuning from inital training..."
  OPTION=" --finetune-from-model $3 $load_from_ema "
fi

model_name=$4
tmp_dir=`mktemp -d`

torchrun --nproc_per_node=$n_gpu --master_port $MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP \
       $(which unicore-train) --user-dir diffold \
       --unifold-db-path $1 \
       --num-workers 4 --ddp-backend=no_c10d \
       --task diffold --loss diffold --arch diffold_v6 \
       --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0  --per-sample-clip-norm 0.1 --allreduce-fp32-grad  \
       --lr-scheduler exponential_decay --lr $lr --warmup-updates $warmup_step --decay-ratio $decay_ratio --decay-steps $decay_step --stair-decay \
       --update-freq $update_freq --batch-size $batch_size --seed $seed  --tensorboard-logdir $2/tsb/ \
       --max-update $total_step --max-epoch 3 --log-interval 10 --log-format simple \
       --valid-subset "eval_train,eval_init,eval_half,eval_last,eval_fold,eval_invfold" --valid-samples-per-step 128 \
       --save-interval-updates $save_step --validate-interval-updates $valid_step --keep-interval-updates 15 --no-epoch-checkpoints  \
       --save-dir $2 --tmp-save-dir $tmp_dir --required-batch-size-multiple 1 --ema-decay -1 --data-buffer-size 32 --model-name $model_name --fixed-validation-seed 38 \
       --bf16 --bf16-sr  \
       --checkpoint-ratio $ckpt_ratio \
       $OPTION

rm -rf $tmp_dir

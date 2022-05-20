export CUDA_VISIBLE_DEVICES=0,1
dataset=tmt
datapath=../data/$dataset/amr_2
MODEL=$1
interval=1

lr=5e-5

model_size="syllable"

outpath=output/pre-train/${dataset}-bartpho-${model_size}-Unifiedtextinf-JointDenoise-6task-${lr}-AMREOS

mkdir -p $outpath
echo "OutputDir: $outpath"

python pre-train/run_multitask_unified_pretraining.py \
  --train_file $datapath/all_v2_p1.jsonl \
  --val_file $datapath/all_v2_p2.jsonl \
  --test_file $datapath/all_v2_p2.jsonl \
  --output_dir $outpath \
#   --smart_init \
  --bartpho \
  --mlm \
  --mlm_amr \
  --mlm_text \
#   --mlm_amr_plus_text \
  --mlm_text_plus_amr \
#   --mlm_joint_to_amr \
#   --mlm_joint_to_text \
  --block_size 256 \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --model_type "vinai/bartpho-${model_size}" \
  --model_name_or_path $MODEL \
  --save_total_limit 2 \
  --do_train \
  --do_eval \
  --evaluate_during_training  \
  --num_train_epochs 100  \
  --learning_rate $lr \
  --joint_train_interval $interval \
  --warmup_steps 2500 \
  --max_steps 100000 \
  --logging_steps 1000 \
  --overwrite_output_dir 2>&1 | tee $outpath/run.log \
#   --no_cuda \
  --fp16 \
  --local_rank -1
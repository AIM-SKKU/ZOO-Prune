#!/bin/bash
set -x

### Model Selection and Corresponding Ratios
model_name=liuhaotian/llava-v1.5-7b # liuhaotian/llava-v1.5-7b  liuhaotian/llava-v1.5-13b liuhaotian/llava-v1.6-vicuna-7b
pruning_ratio=0.1112 # 64 tokens for 7b, 320 tokens for vicuna-7b


if [[ "$model_name" == "liuhaotian/llava-v1.5-7b" || "$model_name" == "liuhaotian/llava-v1.5-13b" ]]; then
    [[ "$model_name" == "liuhaotian/llava-v1.5-7b" ]] && model_name_short="llava_1.5_7b" || model_name_short="llava_1.5_13b"
    if [[ "$pruning_ratio" == "0.1112" ]]; then token=64; elif [[ "$pruning_ratio" == "0.2223" ]]; then token=128; elif [[ "$pruning_ratio" == "0.3334" ]]; then token=192; fi
elif [[ "$model_name" == "liuhaotian/llava-v1.6-vicuna-7b" ]]; then
    model_name_short="llava_NeXT_7b"
    if [[ "$pruning_ratio" == "0.0556" ]]; then token=160; elif [[ "$pruning_ratio" == "0.1112" ]]; then token=320; elif [[ "$pruning_ratio" == "0.2223" ]]; then token=640; fi
fi

PAPER_TABLE="gqa,mmbench_en_dev,mme,pope,scienceqa_img,textvqa,mmmu_val,seedbench,vqav2"

LOG_DIR=./logs_zoo_prune/

mkdir -p $LOG_DIR

RUN_NAME="ZOO_Prune_${model_name_short}_${token}_Token"

HF_TOKEN="HF_token" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE=OURS \
ARCHIVE_MODE=$model_name \
LAYER_INDEX=0 \
SUBSET_RATIO=$pruning_ratio \
NOISERECOV_NUM=64 \
NOISERECOV_INTENS=0.01 \
python3 -m accelerate.commands.launch \
    --num_processes=4 \
    --main_process_port 1111 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="${model_name}" \
    --tasks "${PAPER_TABLE}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${RUN_NAME}" \
    --output_path "${LOG_DIR}/${RUN_NAME}"

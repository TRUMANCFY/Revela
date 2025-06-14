#!/bin/bash
source activate revela

export TRITON_CACHE_DIR=...
export TRITON_HOME=...
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRITON_PRINT_AUTOTUNING=1

export ROOT_DIR=./
export OUTPUT_DIR=...
export RUN_NAME=...

deepspeed --include localhost:0,1,2,3 --master_port 6022 --module tevatron.llm_retriever.driver.train \
  --deepspeed $ROOT_DIR/deepspeed/ds_zero3_config.json \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --lora \
  --lora_r 256 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 500 \
  --bm25_retrieval_file $DATA_PATH \
  --add_passage_prefix True \
  --add_query_prefix True \
  --first_half True \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --attn_temperature 0.0001 \
  --per_device_train_batch_size 1 \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --passage_max_len 157 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --resume latest \
  --top_k 16 \
  --run_name $RUN_NAME

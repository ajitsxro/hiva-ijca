#!/bin/sh
# export DATA_DIR=./data/squadv2
# export MODEL_DIR=./models

# change batch size depending on gpu 
python ../transformers/examples/legacy/question-answering/run_squad_baseline.py  \
    --model_type distilbert   \
    --model_name_or_path distilbert-base-uncased  \
    --output_dir ./outputs/distilbert-finetuning-baseline \
    --data_dir data/squadv2   \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train  \
    --train_file train-v2.0.json   \
    --version_2_with_negative \
    --do_lower_case  \
    --do_eval   \
    --predict_file dev-v2.0.json   \
    --per_gpu_train_batch_size 128   \
    --per_gpu_eval_batch_size 128   \
    --learning_rate 3e-5   \
    --num_train_epochs 3   \
    --weight_decay 0.01   \
    --max_seq_length 384   \
    --doc_stride 128   \
    --threads 10   \
    --save_steps 100  \
    --logging_steps 100  \
    --evaluate_during_training 
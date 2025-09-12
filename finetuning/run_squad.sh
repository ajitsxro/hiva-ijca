#!/bin/sh
# export DATA_DIR=./data/squadv2
# export MODEL_DIR=./models

# Set USE_MITR_MODEL=1 to use MITR modified DistilBERT, otherwise uses baseline
USE_MITR_MODEL=${USE_MITR_MODEL:-0}

# Determine output directory and MITR flag based on USE_MITR_MODEL
if [ "$USE_MITR_MODEL" = "1" ]; then
    OUTPUT_DIR="./outputs/distilbert-finetuning-mitr"
    MITR_FLAG="--use_mitr_model"
    echo "Using MITR modified DistilBERT model"
else
    OUTPUT_DIR="./outputs/distilbert-finetuning-baseline"
    MITR_FLAG=""
    echo "Using baseline DistilBERT model"
fi

# change batch size depending on gpu 
python ../transformers/examples/legacy/question-answering/run_squad_finetuning.py  \
    --model_type distilbert   \
    --model_name_or_path distilbert-base-uncased  \
    --output_dir "$OUTPUT_DIR" \
    --data_dir ./data/squadv2   \
    --overwrite_output_dir \
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
    --save_steps 1000  \
    --logging_steps 10  \
    --evaluate_during_training \
    $MITR_FLAG 
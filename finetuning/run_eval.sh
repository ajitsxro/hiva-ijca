#!/bin/sh

# Set USE_MITR_MODEL=1 to use MITR modified DistilBERT, otherwise uses baseline
USE_MITR_MODEL=${USE_MITR_MODEL:-0}

# Determine output directory based on USE_MITR_MODEL
if [ "$USE_MITR_MODEL" = "1" ]; then
    OUTPUT_DIR="../outputs/distilbert-finetuning-mitr"
    echo "Evaluating MITR modified DistilBERT model"
else
    OUTPUT_DIR="../outputs/distilbert-finetuning-baseline"
    echo "Evaluating baseline DistilBERT model"
fi

cd evaluation
python get_log.py $USE_MITR_MODEL
python squadv2_eval.py "$OUTPUT_DIR" $USE_MITR_MODEL
cd ..

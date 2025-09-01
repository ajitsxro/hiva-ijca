# HIVA-IJCA

## Installation

### Prerequisites
- Python 3.9+
- `uv` package manager installed

### Installation Steps

```bash
# Clone the repository
git clone [repository-url]

# Navigate to project directory
cd [repo-name]

# Sync dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Install transformers package from director
cd transformers
uv pip install .

# Verify installation
python --version
uv pip list
```

## Running the finetune

### DistilBert + SQuAD v2.0 Training Script

To run DistilBERT fine-tuning on SQuAD v2.0:

```bash
# Navigate to the baseline directory
cd finetuning

# Run the finetuning script for baseline
source run_squad.sh

# To run the finetuning script on distilbert with MITR 
USE_MITR_MODEL=1 source run_squad.sh
```

## Results

After running the training script, to evaluate the model:

```bash
cd finetuning

# Run the evaluation script
source run_eval.sh

# Run the evaluation script on MITR version
USE_MITR_MODEL=1 source run_eval.sh     
```

Results are saved in the directory as the run being evaluated as:
- best_thresh_results.json
- tensorboard_log.json

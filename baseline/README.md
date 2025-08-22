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

# Verify installation
python --version
pip list
```

## Running the Baseline

### DistilBert + SQuAD v2.0 Training Script

To run the baseline DistilBERT fine-tuning on SQuAD v2.0:

```bash
# Navigate to the baseline directory
cd baseline

# Run the finetuning script
source run_squad.py
```

## Results

After running our huggingfaces provided training script for SQuAD v2.0 with perplexity added, here are the results without 

```bash
"exact": 59.37842162890592,
"f1": 63.00454083996302,
"total": 11873,
"HasAns_exact": 59.733468286099864,
"HasAns_f1": 66.99610549812418,
"HasAns_total": 5928,
"NoAns_exact": 59.02439024390244,
"NoAns_f1": 59.02439024390244,
"NoAns_total": 5945, "best_exact": 62.25890676324433,
"best_exact_thresh": -2.2732505798339844,
"best_f1": 64.80409815607187,
"best_f1_thresh": -1.4037442207336426,
"pr_exact_ap": 37.9350185172231,
"pr_f1_ap": 46.061044924994974,
"pr_oracle_ap": 73.83702512765699
```
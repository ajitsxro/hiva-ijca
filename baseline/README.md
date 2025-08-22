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

import tensorflow as tf
import json
import os
import glob
import sys
from collections import defaultdict

# Determine model type based on environment variable or command line argument
use_mitr_model = os.environ.get('USE_MITR_MODEL', '0') == '1'
if len(sys.argv) > 1:
    use_mitr_model = sys.argv[1] == '1'

model_type = "mitr" if use_mitr_model else "baseline"

runs_dir = "../runs"
pattern = os.path.join(runs_dir, f"squadv2_{model_type}_*", "events.out.tfevents.*")
event_files = glob.glob(pattern)

if not event_files:
    raise FileNotFoundError(f"No {model_type} event files found in runs directory, run training script first")

# Select the most recent event file for the specified model type
event_file = max(event_files, key=os.path.getmtime)
print(f"Using event file: {event_file}")

# Dictionary to hold metrics by step
data = defaultdict(dict)

# Go through tensorboard event log
for e in tf.compat.v1.train.summary_iterator(event_file):
    for v in e.summary.value:
        if v.HasField('simple_value'):
            data[e.step][v.tag] = v.simple_value
steps_data = [
    {"step": step, **metrics} for step, metrics in sorted(data.items())
]

with open("tensorboard_log.json", "w") as f:
    json.dump(steps_data, f, indent=4)

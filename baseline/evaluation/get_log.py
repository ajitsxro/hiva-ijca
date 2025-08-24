import tensorflow as tf
import json
import os
import glob
from collections import defaultdict


runs_dir = "../runs"
event_files = glob.glob(os.path.join(runs_dir, "**/events.out.tfevents.*"), recursive=True)
if not event_files:
    raise FileNotFoundError("No event files found in runs directory, run training script first")

event_file = max(event_files, key=os.path.getmtime)

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

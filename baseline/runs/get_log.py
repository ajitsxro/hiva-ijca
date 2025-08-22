import tensorflow as tf
import json
from collections import defaultdict

event_file = "Aug22_11-34-07_69d628c65bfc/events.out.tfevents.1755862447.69d628c65bfc"

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

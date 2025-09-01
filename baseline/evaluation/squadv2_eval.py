import json
import sys
import os
import glob
from transformers.data.metrics.squad_metrics import squad_evaluate
from transformers.data.processors.squad import SquadV2Processor

use_mitr_model = os.environ.get('USE_MITR_MODEL', '0') == '1'
if len(sys.argv) > 2: 
    use_mitr_model = sys.argv[2] == '1'

model_type = "mitr" if use_mitr_model else "baseline"
print(f"Looking for {model_type} model run directory...")

runs_dir = "../runs"
run_dirs = glob.glob(os.path.join(runs_dir, f"squadv2_{model_type}_*"))
if not run_dirs:
    print(f"No {model_type} run directories found, falling back to output directory")
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        run_dir = None
    else:
        output_dir = "../outputs/distilbert-finetuning-baseline"
        run_dir = None
else:
    run_dir = max(run_dirs, key=os.path.getmtime)
    print(f"Using run directory: {run_dir}")
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "../outputs/distilbert-finetuning-baseline"

processor = SquadV2Processor()
examples = processor.get_dev_examples("../data/squadv2/", filename="dev-v2.0.json")
print(len(examples))

# maps
qid_to_example_index = {example.qas_id: i for i, example in enumerate(examples)}
qid_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if has_answer]
no_answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if not has_answer]



# load the predictions we generated earlier
filename = f"{output_dir}/predictions_.json"
preds = json.load(open(filename, 'rb'))

# load the null score differences we generated earlier
filename = f"{output_dir}/null_odds_.json"
null_odds = json.load(open(filename, 'rb'))


# the default threshold is set to 1.0 -- we'll leave it there for now
results_default_thresh = squad_evaluate(examples, 
                                        preds, 
                                        no_answer_probs=null_odds, 
                                        no_answer_probability_threshold=1.0)

# print(results_default_thresh)
# best_thresh = -1.4037442207336426

best_thresh = results_default_thresh['best_f1_thresh']

results_best_thresh = squad_evaluate(examples, 
                                        preds, 
                                        no_answer_probs=null_odds, 
                                        no_answer_probability_threshold=best_thresh)


if run_dir:
    results_file = os.path.join(run_dir, "best_thresh_results.json")
else:
    results_file = "best_thresh_results.json"

with open(results_file, "w") as f:
    json.dump(results_best_thresh, f, indent=4)

print(f"Results saved to: {results_file}")
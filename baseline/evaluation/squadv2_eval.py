import json
from transformers.data.metrics.squad_metrics import squad_evaluate
from transformers.data.processors.squad import SquadV2Processor


processor = SquadV2Processor()
examples = processor.get_dev_examples("../data/squadv2/", filename="dev-v2.0.json")
print(len(examples))

# maps
qid_to_example_index = {example.qas_id: i for i, example in enumerate(examples)}
qid_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if has_answer]
no_answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if not has_answer]



# load the predictions we generated earlier
filename = "../outputs/distilbert-finetuning-baseline/predictions_.json"
preds = json.load(open(filename, 'rb'))

# load the null score differences we generated earlier
filename = "../outputs/distilbert-finetuning-baseline/null_odds_.json"
null_odds = json.load(open(filename, 'rb'))



# the default threshold is set to 1.0 -- we'll leave it there for now
results_default_thresh = squad_evaluate(examples, 
                                        preds, 
                                        no_answer_probs=null_odds, 
                                        no_answer_probability_threshold=1.0)

# print(results_default_thresh)
# best_thresh = -1.4037442207336426

best_thresh = results_default_thresh['best_f1_thresh']
# print(best_thresh)

results_best_thresh = squad_evaluate(examples, 
                                        preds, 
                                        no_answer_probs=null_odds, 
                                        no_answer_probability_threshold=best_thresh)

print(results_best_thresh)

with open("best_thresh_results.json", "w") as f:
    json.dump(results_best_thresh, f, indent=4)
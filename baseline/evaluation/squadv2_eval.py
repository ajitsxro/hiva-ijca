import json
from transformers.data.metrics.squad_metrics import squad_evaluate



# load the predictions we generated earlier
filename = "../outputs/distilbert-finetuning-baseline/predictions_.json"
preds = json.load(open(filename, 'rb'))

# load the null score differences we generated earlier
filename = "../outputs/distilbert-finetuning-baseline/null_odds_.json"
null_odds = json.load(open(filename, 'rb'))

filename = "../data/squadv2/dev-v2.0.json"
examples = json.load(open(filename, 'rb'))


# the default threshold is set to 1.0 -- we'll leave it there for now
results_default_thresh = squad_evaluate(examples, 
                                        preds, 
                                        no_answer_probs=null_odds, 
                                        no_answer_probability_threshold=1.0)

print(results_default_thresh)
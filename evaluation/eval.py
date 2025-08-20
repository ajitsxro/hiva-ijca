from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, pipeline
from datasets import load_dataset
import evaluate
import json
import os
import glob

def load_training_history():
    checkpoint_dirs = glob.glob('../outputs/finetuning-baseline/checkpoint-*')
    training_data = []
    
    for checkpoint_dir in sorted(checkpoint_dirs):
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
            
            checkpoint_num = int(os.path.basename(checkpoint_dir).split('-')[1])
            epoch = trainer_state.get('epoch', 0)
            
            eval_entries = [entry for entry in trainer_state['log_history'] if 'eval_loss' in entry]
            if eval_entries:
                latest_eval = eval_entries[-1]
                eval_loss = latest_eval.get('eval_loss', 0)
                training_data.append({
                    'checkpoint': checkpoint_num,
                    'epoch': epoch,
                    'eval_loss': eval_loss
                })
    
    return training_data


training_history = load_training_history()
if training_history:
    print("Epoch\tCheckpoint\tEval Loss")
    for data in training_history:
        print(f"{data['epoch']:.1f}\t{data['checkpoint']}\t\t{data['eval_loss']:.4f}")


model = DistilBertForQuestionAnswering.from_pretrained('../outputs/finetuning-baseline/checkpoint-22209')
tokenizer = DistilBertTokenizerFast.from_pretrained('../outputs/finetuning-baseline/checkpoint-22209')

pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
validation_split = load_dataset("rajpurkar/squad_v2")["validation"]

squad = evaluate.load('squad_v2')

predictions = []
references = []

for example in validation_split:
    result = pipeline(question=example["question"], context=example["context"])

    pred = {
        "id": example["id"],
        "prediction_text": result["answer"] if result["score"] > 0.5 else "",
        "no_answer_probability": 1 - result["score"] if result["score"] <= 0.5 else 0
    }

    predictions.append(pred)

    ref = {
        "id": example["id"],
        "answers": example["answers"]
    }

    references.append(ref)

results = squad.compute(predictions=predictions, references=references)
print(f"F1 score: {results['f1']:.4f}")
print(f"Exact match: {results['exact']:.4f}")

predictions_dict = {pred["id"]: pred["prediction_text"] for pred in predictions}

with open('predictions.json', 'w') as f:
    json.dump(predictions_dict, f, indent=2)



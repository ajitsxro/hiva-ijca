from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, pipeline
from datasets import load_dataset
import evaluate

model = DistilBertForQuestionAnswering.from_pretrained('./outputs/finetuning-baseline/checkpoint-22209')
tokenizer = DistilBertTokenizerFast.from_pretrained('./outputs/finetuning-baseline/checkpoint-22209')

pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
test_split = load_dataset("rajpurkar/squad_v2")["test"]

squad = evaluate.load('squad_v2')

predictions = []
references = []

for example in test_split:
    result = pipeline(question=example["question"], context=example["context"])

    pred = {
        "id": example["id"],
        "prediction_text": result["answer"] if result["score"] > 0.5 else "",
        "no_answer_probability": 1 - result["score"] if result["score"] < 0.5 else 0
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


from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
from datasets import load_from_disk, DatasetDict
import math


class PerplexityCallback(TrainerCallback):
    """A callback to compute and log perplexity after evaluation."""
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            try:
                perplexity = math.exp(metrics["eval_loss"])
                metrics["eval_perplexity"] = perplexity
                print(f"Perplexity: {perplexity:.4f}")
            except OverflowError:
                metrics["eval_perplexity"] = float("inf")
                print("Perplexity: inf")


train_dataset = load_from_disk(
    "data/squadv2/tokenized_squadv2/train")
validation_dataset = load_from_disk(
    "data/squadv2/tokenized_squadv2/validation")
test_dataset = load_from_disk(
    "data/squadv2/tokenized_squadv2/test")

dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})


tokenized_train = dataset["train"]
tokenized_val = dataset["validation"]
tokenized_test = dataset["test"]
print("Data Loaded Correctly.")


model = DistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased")
print("Model Loaded Correctly")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
print("Tokenizer Loaded Correctly.")
args = TrainingArguments(
    output_dir="./outputs/finetuning-baseline",
    eval_strategy="steps",
    eval_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_perplexity",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer, 
    callbacks=[PerplexityCallback()]
)

trainer.train()

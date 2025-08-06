from transformers import TrainingArguments, Trainer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_from_disk, DatasetDict


train_dataset = load_from_disk(
    "data/hotpotqa/tokenized_hotpotqa_fullwiki/train")
validation_dataset = load_from_disk(
    "data/hotpotqa/tokenized_hotpotqa_fullwiki/validation")
test_dataset = load_from_disk(
    "data/hotpotqa/tokenized_hotpotqa_fullwiki/test")

dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})


tokenized_train = dataset["train"]
tokenized_val = dataset["validation"]
tokenized_test = dataset["test"]
print("Data Loaded Correctly.")


model = AutoModelForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased")
print("Model Loaded Correctly")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer Loaded Correctly.")
args = TrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer
)

trainer.train()

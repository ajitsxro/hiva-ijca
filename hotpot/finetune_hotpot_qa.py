from datasets import load_dataset

dataset = load_dataset("hotpot_qa", "fullwiki")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(example):
    return tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

tokenized_train = dataset["train"].map(preprocess, batched=True)
tokenized_val = dataset["validation"].map(preprocess, batched=True)

from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="epoch",
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

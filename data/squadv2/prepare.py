# https://www.kaggle.com/code/amansherjadakhan/fine-tuning-distilbert-for-question-answering
# https://huggingface.co/datasets/rajpurkar/squad_v2


from tqdm import tqdm
import numpy as np
import os
from datasets import load_dataset
from transformers import DistilBertTokenizerFast

# Load the squad dataset
dataset = load_dataset("rajpurkar/squad_v2")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# Create test split from train data
test_size = len(dataset['validation'])  # Use same size as validation (11873)
train_dataset = dataset['train']

# Split the train dataset
train_split = train_dataset.select(range(test_size, len(train_dataset)))
test_split = train_dataset.select(range(test_size))

# Update dataset with new splits
from datasets import DatasetDict

dataset = DatasetDict({
    'train': train_split,
    'validation': dataset['validation'],
    'test': test_split
})


# features: ['id', 'title', 'context', 'question', 'answers']

# print(dataset['train'])
# print('Sample id:', dataset['train']['id'][0])
# print('Sample title:', dataset['train']['title'][0])
# print('Sample question:', dataset['train']['question'][0])
# print('Sample answer:', dataset['train']['answers'][0])
# print('Sample context:', dataset['train']['context'][0])



def preprocess(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    # tokenize
    inputs = tokenizer(
        questions,
        contexts,
        max_length=256,
        truncation="only_second",
        stride=128,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # start and end positions for the answers
    start_positions = []
    end_positions = []

    for i, offset in enumerate(inputs["offset_mapping"]):
        answer = examples["answers"][i]
        
        # Handle unanswerable questions (empty answer lists in SQuAD v2)
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue
            
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        token_start_idx = token_end_idx = 0
        for idx, (start, end) in enumerate(offset):
            if start <= start_char and end >= start_char:  
                token_start_idx = idx
            if start <= end_char and end >= end_char:  
                token_end_idx = idx
                break 

        start_positions.append(token_start_idx)
        end_positions.append(token_end_idx)


    # Add the labels
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs



'''
Use the preprocess function on the data
3 splits [train, validation, test]
this makes a directory, /tokenized_squadv2
in this directory we have 3 folders for each split
data is stored as a .arrow file
'''

tokenized_datasets = {}

# process each split 
for split in ['train', 'validation', 'test']:
    tokenized_datasets[split] = dataset[split].map(
        preprocess,
        batched=True,
        batch_size=32,       
        remove_columns=dataset[split].column_names
    )

# save to tokenized_hotpotqa_fullwiki dir and make it if it does not exist
save_dir = "./tokenized_squadv2"
os.makedirs(save_dir, exist_ok=True)

for split in tokenized_datasets:
    tokenized_datasets[split].save_to_disk(os.path.join(save_dir, split))
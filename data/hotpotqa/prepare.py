from tqdm import tqdm
import numpy as np
import torch
import os
from datasets import load_dataset
from transformers import DistilBertTokenizerFast


'''
Load the dataset using huggingface datasets
Workaround:
    forked archive of hotpo_qa, original author is no longer maintaining dataset
'''
dataset = load_dataset('vincentkoc/hotpot_qa_archive', 'fullwiki') 
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# check dataset
# features: ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context']

# already has train, test, val, split
# print('Train:', len(dataset['train']))
# print('Test:', len(dataset['test']))
# print('Val:', len(dataset['validation']))

# print(dataset['train'])
# print('Sample id:', dataset['train']['id'][0])
# print('Sample question:', dataset['train']['question'][0])
# print('Sample answer:', dataset['train']['answer'][0])
# print('Sample type:', dataset['train']['type'][0])
# print('Sample level:', dataset['train']['level'][0])
# print('Sample supporting_facts:', dataset['train']['supporting_facts'][0])
# print('Sample context:', dataset['train']['context'][0])



'''
DistilBERT takes in
    {input_ids, attention_mask, start_positions, end_positions}
'''

def formating_context(context_dict):
    '''
    Currently the context is stored as a list of titles and a list of sentences lists
    that correspond to each title.
    This combines the title and corresponding sentences into a string using zip
    '''
    context = ''
    titles = context_dict['title']
    sentences = context_dict['sentences']
    
    for title, sent_list in zip(titles, sentences):
        context += f"{title}: " + " ".join(sent_list) + "\n"
    
    return context.strip()  # remove leading and trailing whitespace


def preprocess(examples):
    '''
    Preprocess the data to work for distilBERT
    we need the question, context, and the index for the start and end of the answer
    '''
    questions = examples['question']
    answers = examples['answer']
    contexts_original = examples['context']
    
    # Convert context into string using formating_context function
    context = [formating_context(c) for c in contexts_original]

    # Tokenize 
    inputs = tokenizer(
        questions,
        context,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_offsets_mapping=True
    )

    # start and end positions for the answers
    start_positions = []
    end_positions = []

    # get the span of the answer
    for i, (answer, context, offsets) in enumerate(zip(answers, context, inputs["offset_mapping"])):
        answer = answer.strip().lower()
        context_lower = context.lower()

        answer_start_char = context_lower.find(answer)
        answer_end_char = answer_start_char + len(answer)

        if answer_start_char == -1:
            # If answer not found in context use the CLS token position
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Find token positions from character offsets
        token_start_idx = token_end_idx = 0
        for idx, (start, end) in enumerate(offsets):
            if start <= answer_start_char < end:
                token_start_idx = idx
            if start < answer_end_char <= end:
                token_end_idx = idx
                break

        start_positions.append(token_start_idx)
        end_positions.append(token_end_idx)

    # Remove offset mappings (not needed by model)
    inputs.pop("offset_mapping")

    # Add the labels
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs





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

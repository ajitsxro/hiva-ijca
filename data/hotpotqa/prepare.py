from tqdm import tqdm
import numpy as np
import torch
import os
from datasets import load_dataset
from transformers import DistilBertTokenizerFast


if __name__ == '__main__':

    '''
    Workaround:
        forked archive of hotpo_qa, originally author is no longer maintaining dataset
    '''
    dataset = load_dataset('vincentkoc/hotpot_qa_archive', 'fullwiki') 

    
    # check data
    # features: ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context']

    # already has train, test, val, split
    print('Train:', len(dataset['train']))
    print('Test:', len(dataset['test']))
    print('Val:', len(dataset['validation']))

    print(dataset['train'])
    print('Sample id:', dataset['train']['id'][0])
    print('Sample question:', dataset['train']['question'][0])
    print('Sample answer:', dataset['train']['answer'][0])
    print('Sample type:', dataset['train']['type'][0])
    print('Sample level:', dataset['train']['level'][0])
    print('Sample supporting_facts:', dataset['train']['supporting_facts'][0])
    print('Sample context:', dataset['train']['context'][0])



    
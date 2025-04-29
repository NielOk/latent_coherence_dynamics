import torch
import numpy as np
import torch.nn.functional as F
import argparse
import os

SINGLE_PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))

def read_dataset(data_dir):

    dataset = []
    for file in os.listdir(data_dir):
        if not file.endswith(".pt"):
            continue

        file_path = os.path.join(data_dir, file)
        data = torch.load(file_path)

        step = data['step']
        tokens = data['tokens']
        top_k_values = data['topk_values']
        top_k_indices = data['topk_indices']

        entry = {
            'step': data['step'],
            'tokens': data['tokens'],
            'topk_values': data['topk_values'],
            'topk_indices': data['topk_indices'],
        }
        dataset.append(entry)

    # Sort the dataset by step number
    dataset = sorted(dataset, key=lambda x: x['step'])

    return dataset

if __name__ == '__main__':

    data_dir = os.path.join(SINGLE_PROMPT_DIR, "single_prompt_collected_dataset")

    dataset = read_dataset(data_dir)

    # Print sorted dataset
    for entry in dataset:
        print(f"Step: {entry['step']}")
        print(f"Tokens: {entry['tokens']}")
        print(f"Top-k Values: {entry['topk_values']}")
        print(f"Top-k Indices: {entry['topk_indices']}")
        print("-" * 50)
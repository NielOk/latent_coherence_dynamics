import torch
import numpy as np
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import json
from dotenv import load_dotenv
import os
from huggingface_hub import login

# Hugging face login
load_dotenv()
hugging_face_token = os.getenv("hugging_face_api_key")
login(token=hugging_face_token)

def read_architecture():
    '''
    Read the architecture of the instruct model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instruct model
    model_name = 'GSAI-ML/LLaDA-8B-Instruct'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    for name, module in model.named_modules():
        print(name, type(module))

def main():
    read_architecture()

if __name__ == "__main__":
    main()
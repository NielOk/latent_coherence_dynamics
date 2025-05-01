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


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate_with_logits(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                         cfg_scale=0., remasking='low_confidence', mask_id=126336, collect_logits=False):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    num_blocks = gen_length // block_length
    steps = steps // num_blocks

    collected = []

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length:
                              prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits  # shape: (1, seq_len, vocab_size)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            if collect_logits:
                collected.append({
                    'logits': logits[0].float().cpu(),      # shape: [seq_len, vocab_size]
                    'tokens': x[0].clone().cpu(),           # shape: [seq_len]
                    'step': num_block * steps + i
                })

    return x, collected if collect_logits else x


def format_questions(ds):
    formatted_questions = []
    labels = []

    for sample in ds:
        premise = sample["premises"]
        conclusion = sample["conclusion"]
        label = sample["label"]

        formatted_question = f'{premise}. Based on the above, is the conclusion "{conclusion}" true? Think out loud carefully and answer "False", "Uncertain", or "True".'
        formatted_questions.append((formatted_question))
        labels.append(label)

    return formatted_questions, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_variant', choices=['base', 'instruct'], default='instruct',
                        help="Choose 'base' or 'instruct' model variant.")
    args = parser.parse_args()

    device = 'cuda'

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("yale-nlp/FOLIO", split="train")

    # Get formatted questions and labels
    formatted_questions, labels = format_questions(ds)

    # Load the model and tokenizer
    if args.model_variant == 'instruct':
        model_name = 'GSAI-ML/LLaDA-8B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

        input_ids_list = []
        for prompt in formatted_questions:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt)['input_ids']
            input_ids_list.append(input_ids)
    else:
        model_name = 'GSAI-ML/LLaDA-8B-Base'
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        
        input_ids_list = []
        for prompt in formatted_questions:
            input_ids = tokenizer(prompt)['input_ids']
            input_ids_list.append(input_ids)

    overall_save_dir_name = "collected_logits"
    if not os.path.exists(overall_save_dir_name):
        os.makedirs(overall_save_dir_name)

    # Collect logits
    for i in range(len(input_ids_list)):

        save_dir_name = f"prompt_id_{i}"
        save_dir = os.path.join(overall_save_dir_name, save_dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save prompt in directory
        prompt = formatted_questions[i]
        with open(os.path.join(save_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        # Generate logits
        input_ids = torch.tensor(input_ids_list[i]).to(device).unsqueeze(0)
        out, collected = generate_with_logits(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='random', collect_logits=True)

        



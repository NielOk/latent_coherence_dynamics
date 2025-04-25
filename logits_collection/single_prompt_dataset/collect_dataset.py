import torch
import numpy as np
import torch.nn.functional as F
import argparse
import os

from transformers import AutoTokenizer, AutoModel


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_variant', choices=['base', 'instruct'], default='instruct',
                        help="Choose 'base' or 'instruct' model variant.")
    args = parser.parse_args()

    device = 'cuda'

    if args.model_variant == 'instruct':
        model_name = 'GSAI-ML/LLaDA-8B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

        prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt)['input_ids']
    else:
        model_name = 'GSAI-ML/LLaDA-8B-Base'
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

        prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
        input_ids = tokenizer(prompt)['input_ids']

    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    collect_logits = True
    mask_id = 126336
    out, collected = generate_with_logits(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', collect_logits=collect_logits, mask_id=mask_id)

    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])

    # Save directory
    save_dir = "single_prompt_collected_dataset"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save tokens and top-k logits at each step
    k = 64  # top-k sparsity level for interpretability
    for entry in collected:
        step = entry['step']
        tokens = entry['tokens']         # shape: (seq_len,)
        logits = entry['logits']         # shape: (seq_len, vocab_size)

        # Get top-k per token position
        topk_values, topk_indices = torch.topk(logits, k, dim=-1)  # shapes: (seq_len, k)

        # Optional: convert to float16 to save space
        topk_values = topk_values.half()

        # Save step as a compact dict
        torch.save({
            'step': step,
            'tokens': tokens,               # still useful for masking later
            'topk_values': topk_values,     # (seq_len, k)
            'topk_indices': topk_indices    # (seq_len, k)
        }, os.path.join(save_dir, f'step_{step}.pt'))

if __name__ == '__main__':
    main()
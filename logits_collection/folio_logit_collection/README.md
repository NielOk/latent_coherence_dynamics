# LLaDA 8B FOLIO Collected Logits Dataset

This dataset contains logits collected from the `GSAI-ML/LLaDA-8B-Instruct` model on the training set of the [FOLIO](https://huggingface.co/datasets/yale-nlp/FOLIO) dataset. For each prompt, we record:

- `prompt_id`: unique prompt directory
- `prompt`: natural language input question
- `step`: inference step during generation
- `tokens`: the token sequence at that step
- `topk_values`: top-k logits (float32) at each position. k is 64 for this particular dataset.
- `topk_indices`: corresponding token IDs for top-k logits. k is 64 for this particular dataset.

This is intended for **latent decomposition of token dynamics** using sparse autoencoders for semantic interpretability in masked denoising diffusion inference, specifically for LLaDA. 

Train test split is intended to happen after model download, so this model only has a train split. 

## Usage

```python
from datasets import load_dataset

ds = load_dataset("NielOk/LLaDA_8B_folio_collected_logits_dataset", split="train")
print(ds[0])
'''
Once you run this, delete the `hf_ready_dataset` folder to save space.
'''

import os
import torch
from datasets import Dataset, DatasetDict, Features, Value, Array2D, Sequence, concatenate_datasets
from huggingface_hub import login, create_repo, upload_folder
from dotenv import load_dotenv

BATCH_SIZE = 100  # You can tune this for memory use
FOLIO_LOGIT_COLLECTION_DIR = os.path.dirname(os.path.abspath(__file__))

# Hugging Face login
load_dotenv()
hugging_face_token = os.getenv("niel_hugging_face_token")
login(token=hugging_face_token)

def collect_data_batch(root_dir, start_prompt_idx, end_prompt_idx):
    prompt_ids = sorted(os.listdir(root_dir))[start_prompt_idx:end_prompt_idx]
    entries = []

    for prompt_id in prompt_ids:
        prompt_path = os.path.join(root_dir, prompt_id)
        if not os.path.isdir(prompt_path):
            continue

        with open(os.path.join(prompt_path, "prompt.txt")) as f:
            prompt = f.read()

        for fname in sorted(os.listdir(prompt_path)):
            if fname.startswith("step_") and fname.endswith(".pt"):
                step_path = os.path.join(prompt_path, fname)
                data = torch.load(step_path)

                entries.append({
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "step": data['step'],
                    "tokens": data['tokens'].tolist(),
                    "topk_values": data['topk_values'].tolist(),
                    "topk_indices": data['topk_indices'].tolist(),
                })

    return entries

def main():
    repo_id = "NielOk/LLaDA_8B_folio_collected_logits_dataset"
    make_dataset_private = False
    local_dir = os.path.join(FOLIO_LOGIT_COLLECTION_DIR, "collected_logits")

    all_prompt_ids = sorted([pid for pid in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, pid))])
    num_prompts = len(all_prompt_ids)

    # Define dataset schema
    features = Features({
        "prompt_id": Value("string"),
        "prompt": Value("string"),
        "step": Value("int32"),
        "tokens": Sequence(Value("int32")),
        "topk_values": Sequence(Sequence(Value("float32"))), 
        "topk_indices": Sequence(Sequence(Value("int32"))),
    })

    dataset_splits = []

    print(f"Processing {num_prompts} prompts in batches of {BATCH_SIZE}...")

    for i in range(0, num_prompts, BATCH_SIZE):
        batch = collect_data_batch(local_dir, i, min(i + BATCH_SIZE, num_prompts))
        print(f"Loaded prompts {i} to {min(i + BATCH_SIZE, num_prompts)} - {len(batch)} total examples")
        ds_batch = Dataset.from_list(batch, features=features)
        dataset_splits.append(ds_batch)

    print("Concatenating all batches...")
    full_dataset = concatenate_datasets(dataset_splits)
    dataset_dict = DatasetDict({"train": full_dataset})

    # Save dataset
    print("Saving dataset to disk...")
    save_path = "hf_ready_dataset"
    dataset_dict.save_to_disk(save_path)


    # Upload to Hub
    print(f"Uploading to Hugging Face Hub: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="dataset", private=make_dataset_private, exist_ok=True)
    upload_folder(repo_id=repo_id, folder_path=save_path, repo_type="dataset")

    print("Upload complete.")

if __name__ == "__main__":
    main()
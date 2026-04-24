import torch
import os
from datasets import Dataset, load_dataset
from transformers import default_data_collator


def _load_cached_wikitext_split(split):
    cache_dir = (
        "/home/wmq/.cache/huggingface/datasets/wikitext/"
        "wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3"
    )
    arrow_path = os.path.join(cache_dir, f"wikitext-{split}.arrow")
    if os.path.isfile(arrow_path):
        return Dataset.from_file(arrow_path)
    return None


def _preprocess(tokenizer, examples, max_length=128):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=max_length
    )


def get_dataset(dataset_name, subset, split, size=None, start=0):
    if dataset_name == "wikitext" and subset == "wikitext-2-raw-v1":
        dataset = _load_cached_wikitext_split(split)
        if dataset is not None:
            if size is None:
                return dataset
            end = min(start + size, len(dataset))
            return dataset.select(range(start, end))
    if size is None:
        dataset = load_dataset(dataset_name, subset)[split]
    else:
        dataset = load_dataset(dataset_name, subset, streaming=True)[split]
        dataset = dataset.skip(start).take(size)

    return dataset


def get_dataloader(dataset, tokenizer, batch_size, num_workers=4, max_length=128):
    dataset = dataset.map(
        lambda examples: _preprocess(tokenizer, examples, max_length),
        batched=True,
        batch_size=batch_size,
        remove_columns=["text", "timestamp", "url"],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=default_data_collator,
    )
    return dataloader

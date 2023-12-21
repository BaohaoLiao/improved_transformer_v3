import math
import argparse
from itertools import chain
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, GPTQConfig
from models import OPTForCausalLM, ImprovedQOPTForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help='model name or path')
    parser.add_argument('--dataset_name', type=str, default="wiki40b", help='dataset name')
    parser.add_argument('--dataset_config_name', type=str, default="en", help='dataset config name')
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--bit_width', type=int, default=16)
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_type', type=str, default="vanilla")
    args = parser.parse_args()
    return args

def build_model_and_tokenizer(model_name_or_path, args):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}

    if args.model_type == "qopt":
        auto_model = OPTForCausalLM
    elif args.model_type == "improved_qopt":
        auto_model = ImprovedQOPTForCausalLM
    else:
        auto_model = AutoModelForCausalLM

    if args.bit_width < 16:
        gptq_config = GPTQConfig(
            bits=args.bit_width,
            group_size=args.group_size,
            dataset="c4",
            desc_act=False,
        )
        model = auto_model.from_pretrained(
            model_name_or_path,
            quantization_config=gptq_config,
            **kwargs
        )
    else:
        model = auto_model.from_pretrained(
            model_name_or_path,
            **kwargs
        )
    return model, tokenizer

def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name_or_path, args)

    # Load dataset
    raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name, split="validation")
    raw_datasets = DatasetDict()
    raw_datasets["validation"] = raw_dataset

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    column_names = raw_datasets["validation"].column_names
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // args.seq_len) * args.seq_len
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + args.seq_len] for i in range(0, total_length, args.seq_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {args.seq_len}",
    )

    if args.num_samples > 0:
        eval_dataset = lm_datasets["validation"].select(range(args.num_samples))
    else:
        eval_dataset = lm_datasets["validation"]
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

    # Inference
    model.eval()
    losses = []
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            for key, value in batch.items():
                batch[key] = value.to("cuda")
            outputs = model(**batch)

        loss = outputs.loss
        losses += [loss.item() for _ in range(len(batch["input_ids"]))]

    eval_loss = np.mean(losses)
    perplexity = math.exp(eval_loss)
    print(f"eval_loss = {eval_loss} | perplexity = {perplexity}")


if __name__ == "__main__":
    main()
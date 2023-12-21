# borrowed from https://github.com/mit-han-lab/smoothquant

import math
import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from fake_quant.w8a8linear import W8A8Linear
from fake_quant.smooth import smooth_lm

from models import OPTForCausalLM as QOPTForCausalLM
from models import ImprovedQOPTForCausalLM
from models.modeling_opt import (
    OPTAttention as QOPTAttention,
    OPTDecoderLayer as QOPTDecoderLayer,
)
from models.modeling_improved_qopt import (
    OPTAttention as ImprovedQOPTAttention,
    OPTDecoderLayer as ImprovedQOPTDecoderLayer,
)


def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer) or isinstance(m, QOPTDecoderLayer) or isinstance(m, ImprovedQOPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention) or isinstance(m, QOPTAttention) or isinstance(m, ImprovedQOPTAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model


class Evaluator:
    def __init__(self, dataset, tokenizer, device, args):
        self.tokenizer = tokenizer
        self.device = device
        self.num_samples = args.num_samples

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text']) #, max_length=args.seq_len, truncation=True)
            #example["labels"] = example["input_ids"].copy()
            return example

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // args.seq_len) * args.seq_len
            result = {
                k: [t[i: i + args.seq_len] for i in range(0, total_length, args.seq_len)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        self.dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=args.num_workers,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        self.dataset = self.dataset.map(
            group_texts,
            batched=True,
            num_proc=args.num_workers,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {args.seq_len}",
        )
        self.dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "labels"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        losses = []
        for index, batch in enumerate(tqdm(self.dataset, total=len(self.dataset))):
            for key, value in batch.items():
                batch[key] = value.to(self.device).unsqueeze(0)
            outputs = model(**batch)
            loss = outputs.loss
            losses += [loss.item() for _ in range(len(batch["input_ids"]))]

            if index > self.num_samples-2:
                break

        mean_loss = np.mean(losses)
        perplexity = math.exp(mean_loss)
        return mean_loss, perplexity


def build_model_and_tokenizer(model_name, args):
    # TODO: check whether to use GPT2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    if args.model_type == "opt":
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    elif args.model_type == "qopt":
        model = QOPTForCausalLM.from_pretrained(model_name, **kwargs)
    elif args.model_type == "improved_qopt":
        model = ImprovedQOPTForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='facebook/opt-1.3b', help='model name')
    parser.add_argument('--dataset_name', type=str, default='wiki40b', help='evaluate on the validation set')
    parser.add_argument('--dataset_config_name', type=str, default='en', help='dataset config name')
    parser.add_argument('--act_scales_path', type=str, help='activation scales')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--model_type', type=str, default="vanilla", choices=["opt", "qopt", "improved_qopt"])
    parser.add_argument('--no_smooth', action="store_true")
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    model, tokenizer = build_model_and_tokenizer(args.model_name, args)

    # fake quantization
    if args.act_scales_path is not None and not args.no_smooth:
        act_scales = torch.load(args.act_scales_path)
        smooth_lm(model, act_scales, args.alpha)
        model = quantize_model(model)
    elif args.no_smooth:
        model = quantize_model(model)
    print(model)

    if args.num_samples > 0:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=f"validation[:{10*args.num_samples}]")
    else:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, split="validation")
    evaluator = Evaluator(dataset, tokenizer, "cuda:0", args)

    loss, perplexity = evaluator.evaluate(model)
    print(f"loss = {loss} | perplexity = {perplexity}")


if __name__ == "__main__":
    main()
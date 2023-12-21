# borrowed from https://github.com/mit-han-lab/smoothquant

import os
import argparse

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from fake_quant.calibration import get_act_scales
from models import OPTForCausalLM as QOPTForCausalLM
from models import ImprovedQOPTForCausalLM


def build_model_and_tokenizer(model_name, args):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    if args.model_type == "opt":
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    elif args.model_type == "qopt":
        model = QOPTForCausalLM.from_pretrained(model_name, **kwargs)
    elif args.model_type == "improved_qopt":
        model = ImprovedQOPTForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='facebook/opt-1.3b', help='model name')
    parser.add_argument('--output_path', type=str, default='act_scales/opt-1.3b.pt',
                        help='where to save the act scales')
    parser.add_argument('--dataset_name', type=str, default='wiki40b', help='calibration dataset')
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--model_type', type=str, default="vanilla", choices=["opt", "qopt", "improved_qopt"])
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name, args)
    act_scales = get_act_scales(model, tokenizer, args.dataset_name, args.num_samples, args.seq_len)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == '__main__':
    main()
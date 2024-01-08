#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import json
import warnings
from itertools import chain
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from pprint import pformat
from typing import Optional
from dataclasses import dataclass, field

import datasets
import evaluate
import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets, load_metric, load_from_disk
from torch.utils.data import DataLoader
from timm.utils import AverageMeter

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from run_clm_registered_tokens import ModelArguments, DataTrainingArguments
from models.opt_attention import OPTAttentionWithExtras
from models.quantized_opt import QuantizedOPTForCausalLM
from models.quant_configs import get_quant_config
from utils import kurtosis, count_params, pass_data_for_range_estimation, val_qparams
from quantization.range_estimators import OptMethod, RangeEstimators
from quantization.quantizers import QMethods


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.35.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class NewTrainingArguments(TrainingArguments):
    quantize: Optional[bool] = field(
        default=False,
        metadata={"help": "Quantization for evaluation"},
    )
    est_num_batches: Optional[int] = field(
        default=1,
        metadata={"help": "Number of batch for calibration"},
    )
    n_bits: Optional[int] = field(
        default=8,
        metadata={"help": "n-bit for quantization"},
    )
    n_bits_act: Optional[int] = field(
        default=8,
        metadata={"help": "n-bit for acttivation"},
    )
    no_weight_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "No quantization for weight"},
    )
    no_act_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "No quantization for activation"},
    )
    qmethod_acts: Optional[str] = field(
        default="asymmetric_uniform",
        metadata={"help": "Asymmetric uniform"},
    )
    ranges_weights: Optional[str] = field(
        default="minmax",
        metadata={"help": "Minmax for weights"},
    )
    ranges_acts: Optional[str] = field(
        default="running_minmax",
        metadata={"help": "Running minmax for activation"},
    )
    percentile: Optional[float] = field(
        default=None,
        metadata={"help": "Percentile (in %) for range estimation"},
    )
    quant_setup: Optional[str] = field(
        default="all",
        metadata={"help": ""},
    )
    def __post_init__(self):
        super().__post_init__()

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, NewTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args, training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        if "opt" in model_args.config_name:
            config.init_std = 0.006

    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        logger.info(model)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # replace self-attention module with ours
    # NOTE: currently assumes OPT
    logger.info(f"Attention parameters, alpha: {model_args.alpha}, eta: {model_args.eta}, beta: {model_args.beta}")
    for layer_idx in range(len(model.model.decoder.layers)):
        old_attn = model.model.decoder.layers[layer_idx].self_attn
        new_attn = OPTAttentionWithExtras(
            embed_dim=old_attn.embed_dim,
            num_heads=old_attn.num_heads,
            dropout=old_attn.dropout,
            is_decoder=old_attn.is_decoder,
            bias=True,
            # YB
            alpha=model_args.alpha,
            max_seq_length=data_args.block_size,
            # Baohao
            eta=model_args.eta,
            beta=model_args.beta
        )
        # copy loaded weights
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)
        model.model.decoder.layers[layer_idx].self_attn = new_attn

    logger.info("FP Model:")
    logger.info(model)

    # Display num params
    n_embeddings = count_params(model.model.decoder.embed_tokens) + count_params(model.model.decoder.embed_positions)
    n_decoder = count_params(model.model.decoder) - n_embeddings
    n_head = count_params(model.lm_head)
    logger.info(
        f"\nNumber of parameters:\n"
        f"\t* Embeddings:\t{n_embeddings}\n"
        f"\t* Decoder:\t{n_decoder}\n"
        f"\t* Head:\t{n_head}\n"
        f"\t= Total (pre-training):\t{n_embeddings + n_decoder + n_head}\n"
        f"\t= Total (decoder only):\t{n_embeddings + n_decoder}\n"
    )

    # Get the datasets
    tokenized_book_wiki_path = Path(data_args.data_cache_dir) / f"tokenized_book_wiki_{data_args.block_size}"
    if data_args.dataset_name == "wiki+book" and tokenized_book_wiki_path.exists():
        logger.info(f"Loading tokenized dataset from {str(tokenized_book_wiki_path)}")
        lm_datasets = load_from_disk(str(tokenized_book_wiki_path))
    else:
        # Downloading and loading a dataset from the hub.
        if data_args.dataset_name == "wiki+book":
            bookcorpus = load_dataset("bookcorpus", split="train")
            wiki_train = load_dataset("wiki40b", "en", split="train")
            wiki_eval = load_dataset("wiki40b", "en", split="validation")
            wiki_train = wiki_train.remove_columns([col for col in wiki_train.column_names if col != "text"])
            wiki_eval = wiki_eval.remove_columns([col for col in wiki_eval.column_names if col != "text"])
            assert bookcorpus.features.type == wiki_train.features.type == wiki_eval.features.type

            raw_datasets = DatasetDict()
            raw_datasets["train"] = concatenate_datasets([bookcorpus, wiki_train])
            raw_datasets["validation"] = wiki_eval
        else:
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                )

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if training_args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["validation"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > config.max_position_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                block_size = min(1024, config.max_position_embeddings)
        else:
            if data_args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(data_args.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )

        # Save tokenized dataset for saving time if re-running
        if data_args.dataset_name == "wiki+book":
            lm_datasets.save_to_disk(Path(data_args.data_cache_dir) / f"tokenized_book_wiki_{data_args.block_size}")


    if config.num_registered_tokens > 0:
        logger.info(f"Adding registered tokens ...")
        new_tokens = [f"<s{i}>" for i in range(config.num_registered_tokens)]
        registered_tokens = tokenizer("".join(new_tokens))
        for k, v in registered_tokens.items():
            registered_tokens[k] = v[1:]  # delete BOS
        registered_tokens["labels"] = [-100] * config.num_registered_tokens

        def insert_registered_tokens(example, tokens):
            new_example = example.copy()
            interval = len(example) // (len(tokens) + 1)
            for i, element in enumerate(tokens):
                new_example.insert((i + 1) * interval + i, element)
            return new_example

        def add_registered_tokens(examples):
            result = {}
            for k, examples in examples.items():
                new_examples = []
                for example in examples:
                    new_examples.append(insert_registered_tokens(example, registered_tokens[k]))
                result[k] = new_examples
            return result

        # speed up the data processing
        lm_datasets["train"] = lm_datasets["train"].shuffle(seed=training_args.seed).select(range(1000))
        #lm_datasets["validation"] = lm_datasets["validation"].select(range(1000))
        with training_args.main_process_first(desc="adding registered tokens"):
            if not data_args.streaming:
                lm_datasets = lm_datasets.map(
                    add_registered_tokens,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=False,
                    desc=f"Adding registered tokens",
                )
            else:
                lm_datasets = lm_datasets.map(
                    add_registered_tokens,
                    batched=True,
                )

    if training_args.quantize:
        click_config = get_quant_config()

        # override number of batches
        click_config.act_quant.num_batches = training_args.est_num_batches
        click_config.quant.n_bits = training_args.n_bits
        click_config.quant.n_bits_act = training_args.n_bits_act
        click_config.quant.quant_setup = training_args.quant_setup
        if training_args.no_weight_quant:
            click_config.quant.weight_quant = False
        if training_args.no_act_quant:
            click_config.quant.act_quant = False

        # use MSE for weights (ignore `args.ranges_weights`)
        # click_config.quant.weight_quant_method = RangeEstimators.current_minmax
        click_config.quant.weight_quant_method = RangeEstimators.MSE
        click_config.quant.weight_opt_method = OptMethod.grid

        # qmethod acts
        if training_args.qmethod_acts == "symmetric_uniform":
            click_config.quant.qmethod_act = QMethods.symmetric_uniform
        elif training_args.qmethod_acts == "asymmetric_uniform":
            click_config.quant.qmethod_act = QMethods.asymmetric_uniform
        else:
            raise NotImplementedError(f"Unknown qmethod_act setting, '{training_args.qmethod_acts}'")

        # Acts ranges
        if training_args.percentile is not None:
            click_config.act_quant.options["percentile"] = training_args.percentile

        if training_args.ranges_acts == "running_minmax":
            click_config.act_quant.quant_method = RangeEstimators.running_minmax

        elif training_args.ranges_acts == "MSE":
            click_config.act_quant.quant_method = RangeEstimators.MSE
            if training_args.qmethod_acts == "symmetric_uniform":
                click_config.act_quant.options = dict(opt_method=OptMethod.grid)
            elif training_args.qmethod_acts == "asymmetric_uniform":
                click_config.act_quant.options = dict(opt_method=OptMethod.golden_section)

        elif training_args.ranges_acts.startswith("L"):
            click_config.act_quant.quant_method = RangeEstimators.Lp
            p_norm = float(training_args.ranges_acts.replace("L", ""))
            options = dict(p_norm=p_norm)
            if training_args.qmethod_acts == "symmetric_uniform":
                options["opt_method"] = OptMethod.grid
            elif training_args.qmethod_acts == "asymmetric_uniform":
                options["opt_method"] = OptMethod.golden_section
            click_config.act_quant.options = options

        else:
            raise NotImplementedError(f"Unknown range estimation setting, '{training_args.ranges_acts}'")

        qparams = val_qparams(click_config)
        qparams["quant_dict"] = {}
        model = QuantizedOPTForCausalLM(model, **qparams)
        model.set_quant_state(weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant)
        logger.info("Quantized model:")
        logger.info(model)

        # Range estimation
        logger.info("** Estimate quantization ranges on training data **")
        logger.info(lm_datasets)
        train_dataset = lm_datasets["train"]
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=training_args.per_device_train_batch_size,
            num_workers=data_args.preprocessing_num_workers,
        )

        pass_data_for_range_estimation(
            loader=train_dataloader,
            model=model,
            act_quant=click_config.quant.act_quant,
            max_num_batches=click_config.act_quant.num_batches,
        )
        model.fix_ranges()
        model.set_quant_state(weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant)


    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy") # load_metric("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, config.num_registered_tokens + 1:].reshape(-1)
            preds = preds[:, config.num_registered_tokens:-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=default_data_collator,
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=data_args.preprocessing_num_workers,
        )

        # attach hooks for activation stats
        def attach_act_hooks(model):
            act_dict = OrderedDict()

            def _make_hook(name):
                def _hook(mod, inp, out):
                    if isinstance(inp, tuple) and len(inp) > 0:
                        inp = inp[0]
                    if isinstance(out, tuple) and len(out) > 0:
                        out = out[0]
                    act_dict[name] = (inp, out)

                return _hook

            for name, module in model.named_modules():
                module.register_forward_hook(_make_hook(name))
            return act_dict

        act_dict = attach_act_hooks(model)
        num_layers = len(model.model.decoder.layers)

        ACT_KEYS = [
            "model.decoder.final_layer_norm",
            *[f"model.decoder.layers.{j}" for j in range(num_layers)],
            *[f"model.decoder.layers.{j}.fc2" for j in range(num_layers)],
            *[f"model.decoder.layers.{j}.final_layer_norm" for j in range(num_layers)],
            *[f"model.decoder.layers.{j}.self_attn.out_proj" for j in range(num_layers)],
            *[f"model.decoder.layers.{j}.self_attn_layer_norm" for j in range(num_layers)],
        ]

        act_inf_norms = OrderedDict()
        act_kurtoses = OrderedDict()

        # -----------------------------------------------------------------
        # *** Evaluation ***
        has_cuda = torch. cuda. is_available()
        logger.info(f"Validate on GPU: {has_cuda}")
        if has_cuda:
            device = "cuda"
        else:
            device = "cpu"
        model.to(device)
        model.eval()
        losses = []
        for batch_idx, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)

            loss = outputs.loss
            loss_ = loss.repeat(training_args.per_device_eval_batch_size)
            losses.append(loss_)

            # compute inf norms
            if not training_args.quantize:
                for name in ACT_KEYS:
                    if name in act_dict:
                        x_inp, x_out = act_dict[name]
                        x = x_out
                        x = x.view(x.size(0), -1)

                        # compute inf norm
                        inf_norms = x.norm(dim=1, p=np.inf)
                        if not name in act_inf_norms:
                            act_inf_norms[name] = AverageMeter()
                        for v in inf_norms:
                            act_inf_norms[name].update(v.item())

                        # compute kurtosis
                        if batch_idx <= 100:
                            kurt = kurtosis(x)
                            if not name in act_kurtoses:
                                act_kurtoses[name] = AverageMeter()
                            for v in kurt:
                                act_kurtoses[name].update(v.item())

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        logger.info(f"perplexity: {perplexity:.4f}")

        # metrics
        metrics = OrderedDict([("perplexity", perplexity)])

        if not training_args.quantize:
            for name, v in act_inf_norms.items():
                metrics[name] = v.avg

            max_inf_norm = max(v.avg for v in act_inf_norms.values())
            max_ffn_inf_norm = max(v.avg for k, v in act_inf_norms.items() if ".fc" in k)
            max_layer_inf_norm = max(
                act_inf_norms[f"model.decoder.layers.{j}"].avg for j in range(num_layers)
            )

            avg_kurtosis = sum(v.avg for v in act_kurtoses.values()) / len(act_kurtoses.values())
            max_kurtosis = max(v.avg for v in act_kurtoses.values())
            max_kurtosis_layers = max(
                act_kurtoses[f"model.decoder.layers.{j}"].avg for j in range(num_layers)
            )

            metrics["max_inf_norm"] = max_inf_norm
            metrics["max_ffn_inf_norm"] = max_ffn_inf_norm
            metrics["max_layer_inf_norm"] = max_layer_inf_norm

            metrics["avg_kurtosis"] = avg_kurtosis
            metrics["max_kurtosis"] = max_kurtosis
            metrics["max_kurtosis_layers"] = max_kurtosis_layers

            logger.info(f"Max inf norm: {max_inf_norm:.1f}")
            logger.info(f"Max FFN inf norm: {max_ffn_inf_norm:.1f}")
            logger.info(f"Max layer inf norm: {max_layer_inf_norm:.1f}")

            logger.info(f"Avg Kurtosis: {avg_kurtosis:.2f}")
            logger.info(f"Max Kurtosis: {max_kurtosis:.1f}")
            logger.info(f"Max Kurtosis layers: {max_kurtosis_layers:.1f}")

            logger.info(f"\nAll metrics:\n{pformat(metrics)}")

        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
                json.dump(metrics, f)


if __name__ == "__main__":
    main()

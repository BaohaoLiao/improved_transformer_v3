import os
import sys
import math
import json
import warnings
import logging
from itertools import chain
from typing import Optional
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import numpy as np
from timm.utils import AverageMeter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoConfig,
    CONFIG_MAPPING,
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from transformers.utils import check_min_version, send_example_telemetry
import datasets
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets

from models.bert_attention import BertSelfAttentionWithExtras
from run_mlm_registered_tokens import ModelArguments, DataTrainingArguments
from models.quant_configs import get_quant_config
from quantization.range_estimators import OptMethod, RangeEstimators
from utils import count_params, val_qparams, pass_data_for_range_estimation, kurtosis
from models.quantized_bert import QuantizedBertForMaskedLM
from data_collator import DataCollatorForLanguageModeling

EXTRA_METRICS = True
logger = logging.getLogger(__name__)


def attach_act_hooks(model):
    act_dict = OrderedDict()

    def _make_hook(name):
        def _hook(mod, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp = inp[0]
            act_dict[name] = (inp, out)

        return _hook

    for name, module in model.named_modules():
        module.register_forward_hook(_make_hook(name))
    return act_dict


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

    send_example_telemetry("validate_clm", model_args, data_args, training_args)

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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
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
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.num_registered_tokens > 0:
        logger.info(f"Vocabulary size before adding registered tokens: {len(tokenizer.vocab)}")
        new_tokens = [f"<s{i}>" for i in range(model_args.num_registered_tokens)]
        assert len(set(new_tokens) - set(tokenizer.vocab.keys())) == model_args.num_registered_tokens
        tokenizer.add_tokens(new_tokens)
        registered_tokens = tokenizer("".join(new_tokens))
        for k, v in registered_tokens.items():
            registered_tokens[k] = v[1:] # delete BOS
        registered_tokens["labels"] = registered_tokens["input_ids"].copy()
        config.num_registered_tokens = model_args.num_registered_tokens
        logger.info(f"Added registered tokens: {new_tokens}")
        logger.info(f"Vocabulary size after adding registered tokens: {len(tokenizer.vocab)}")

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=model_args.trust_remote_code)

    # >> replace Self-attention module with ours
    # NOTE: currently assumes BERT
    for layer_idx in range(len(model.bert.encoder.layer)):
        old_self = model.bert.encoder.layer[layer_idx].attention.self
        new_self = BertSelfAttentionWithExtras(
            config,
            ## from YB
            alpha=model_args.alpha,
            max_seq_length=data_args.max_seq_length,
            # Baohao
            eta=model_args.eta,
            beta=model_args.beta
        )

        # copy loaded weights
        if model_args.model_name_or_path is not None:
            new_self.load_state_dict(old_self.state_dict(), strict=False)
        model.bert.encoder.layer[layer_idx].attention.self = new_self

    # Display num params
    n_embeddings = count_params(model.bert.embeddings)
    n_encoder = count_params(model.bert.encoder)
    n_head = count_params(model.cls)
    logger.info(
        f"\nNumber of parameters:\n"
        f"\t* Embeddings:\t{n_embeddings}\n"
        f"\t* Encoder:\t{n_encoder}\n"
        f"\t* Head:\t{n_head}\n"
        f"\t= Total (pre-training):\t{n_embeddings + n_encoder + n_head}\n"
        f"\t= Total (encoder):\t{n_embeddings + n_encoder}\n"
    )

    # Get the datasets
    tokenized_book_wiki_path = Path(data_args.data_cache_dir) / f"tokenized_book_wiki_{data_args.max_seq_length}"
    if data_args.dataset_name == "wiki+book" and tokenized_book_wiki_path.exists():
        logger.info(f"Loading tokenized dataset from {str(tokenized_book_wiki_path)}")
        tokenized_datasets = load_from_disk(str(tokenized_book_wiki_path))
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
            """
            raw_datasets = DatasetDict()
            raw_datasets["train"] = load_dataset("bookcorpus", split="train").select(range(1000))
            raw_datasets["validation"] = load_dataset("wiki40b", "en", split="validation").select(range(1000))
            """
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

        if data_args.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
                max_seq_length = 1024
        else:
            if data_args.max_seq_length > tokenizer.model_max_length:
                logger.warning(
                    f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


        # we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
            else:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )

        # Save tokenized dataset for saving time if re-running
        if data_args.dataset_name == "wiki+book":
            tokenized_datasets.save_to_disk(
                Path(data_args.data_cache_dir) / f"tokenized_book_wiki_{data_args.max_seq_length}")

    logger.info("FP model:")
    logger.info(model)

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            registered_tokens=registered_tokens if model_args.num_registered_tokens > 0 else None,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=data_args.preprocessing_num_workers,
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

            # Weight Ranges
            if training_args.ranges_weights == "minmax":
                pass
            elif training_args.ranges_weights in ("mse", "MSE"):
                click_config.quant.weight_quant_method = RangeEstimators.MSE
                click_config.quant.weight_opt_method = OptMethod.grid
            else:
                raise ValueError(f"Unknown weight range estimation: {training_args.ranges_weights}")

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
                raise NotImplementedError(f"Unknown act range estimation setting, '{training_args.ranges_acts}'")

            qparams = val_qparams(click_config)
            qparams["quant_dict"] = {}
            model = QuantizedBertForMaskedLM(model, **qparams)
            model.set_quant_state(weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant)
            logger.info("Quantized model:")
            logger.info(model)

            # Range estimation
            logger.info("** Estimate quantization ranges on training data **")
            pass_data_for_range_estimation(
                loader=eval_dataloader,
                model=model,
                act_quant=click_config.quant.act_quant,
                max_num_batches=click_config.act_quant.num_batches,
            )
            model.fix_ranges()
            model.set_quant_state(weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant)

        # attach hooks for activation stats (if needed)
        act_dict = {}
        if EXTRA_METRICS:
            act_dict = attach_act_hooks(model)

        num_layers = len(model.bert.encoder.layer)
        act_inf_norms = OrderedDict()
        act_kurtoses = OrderedDict()

        # *** Evaluation ***
        has_cuda = torch.cuda.is_available()
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
            if EXTRA_METRICS:
                for j in range(num_layers):
                    for name in (
                            f"bert.encoder.layer.{j}.output.dense",  # FFN output
                            f"bert.encoder.layer.{j}.output.LayerNorm",  # LN(FFN output + input)
                    ):
                        x_inp, x_out = act_dict[name]

                        x = x_out

                        # inf-norm
                        x = x.view(x.size(0), -1)
                        inf_norms = x.norm(dim=1, p=np.inf)
                        if not name in act_inf_norms:
                            act_inf_norms[name] = AverageMeter()
                        for v in inf_norms:
                            act_inf_norms[name].update(v.item())

                        # kurtosis
                        if batch_idx <= 256:
                            kurt = kurtosis(x)
                            if not name in act_kurtoses:
                                act_kurtoses[name] = AverageMeter()
                            for v in kurt:
                                act_kurtoses[name].update(v.item())

                        # compute inf norm also for input
                        if "LayerNorm" in name:
                            x = x_inp
                            x = x.view(x.size(0), -1)
                            inf_norms = x.norm(dim=1, p=np.inf)
                            name += ".input"
                            if not name in act_inf_norms:
                                act_inf_norms[name] = AverageMeter()
                            for v in inf_norms:
                                act_inf_norms[name].update(v.item())

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        logger.info(f"perplexity: {perplexity:.4f}")

        # metrics
        metrics = OrderedDict([("perplexity", perplexity)])

        if EXTRA_METRICS:
            for name, v in act_inf_norms.items():
                metrics[name] = v.avg

            max_ffn_out_inf_norm = max(v.avg for k, v in act_inf_norms.items() if "dense" in k)
            max_LN_out_inf_norm = max(
                v.avg for k, v in act_inf_norms.items() if k.endswith("LayerNorm")
            )
            max_LN_inp_inf_norm = max(v.avg for k, v in act_inf_norms.items() if "input" in k)
            avg_kurtosis = sum(v.avg for v in act_kurtoses.values()) / len(act_kurtoses.values())
            max_kurtosis = max(v.avg for v in act_kurtoses.values())

            metrics["max_ffn_out_inf_norm"] = max_ffn_out_inf_norm
            metrics["max_LN_out_inf_norm"] = max_LN_out_inf_norm
            metrics["max_LN_inp_inf_norm"] = max_LN_inp_inf_norm
            metrics["avg_kurtosis"] = avg_kurtosis
            metrics["max_kurtosis"] = max_kurtosis

            logger.info(f"max FFN output inf norm: {max_ffn_out_inf_norm:.1f}")
            logger.info(f"max FFN input + output inf norm: {max_LN_inp_inf_norm:.1f}")
            logger.info(f"max LN(FFN i + o) inf norm: {max_LN_out_inf_norm:.1f}")
            logger.info(f"Avg Kurtosis: {avg_kurtosis:.2f}")
            logger.info(f"Max Kurtosis: {max_kurtosis:.1f}")

        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
                json.dump(metrics, f)

if __name__ == "__main__":
    main()








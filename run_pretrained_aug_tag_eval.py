#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import itertools
import json
import logging
import math
import os
import random
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import datasets
import nltk
import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from filelock import FileLock
from huggingface_hub import Repository
from peft import LoraConfig
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.utils.peft_types import TaskType
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    Adafactor,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name, is_offline_mode
from transformers.utils.versions import require_version

import semeval
from diff_knn_retriever import Seq2SeqModelWithRetriever
from diff_knn_retriever.retriever import KNNEmbeddingRetriever, RandomEmbeddingRetriever
from loss import LabelSmoother
from utils.tacred_utils import TACRED_LABEL_TEMPLATES
from utils.trie import PredictTrie

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If passed, use only small examples.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument("--test_file", type=str, default=None, help="A csv or a json file containing the test data.")
    parser.add_argument("--type_file", type=str, default=None, help="A csv or a json file containing the type data.")
    parser.add_argument(
        "--template_file", type=str, default=None, help="A csv or a json file containing the type data."
    )
    parser.add_argument(
        "--type_constraint_file", type=str, default=None, help="A csv or a json file containing the type data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--min_target_length",
        type=int,
        default=0,
        help="The minimum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="model", help="Where to store the final model.")
    # parser.add_argument("--load_dir", type=str, help="Where to load the model before training.")
    parser.add_argument("--eval_load_dir", type=str, help="Where to load the model for evaluation.")
    parser.add_argument(
        "--eval_load_last", action="store_true", help="Whether to load the last checkpoint for evaluation."
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--use_label_smoother", action="store_true", help="Whether or not to push the model to the Hub."
    )
    parser.add_argument("--use_adafactor", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")

    # early stopping
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--early_stopping_warmup", type=int, default=0, help="Early stopping warmup")
    parser.add_argument("--early_stopping_metric", type=str, default="dev/loss", help="Early stopping metric")
    parser.add_argument(
        "--early_stopping_maximize", action="store_true", help="Early stopping objective become maximize"
    )

    parser.add_argument("--eval_loss_only", action="store_true", help="Only evaluate loss during training")
    parser.add_argument("--eval_steps", type=int, default=100, help="Run an evaluation every X steps.")
    parser.add_argument("--log_steps", type=int, default=10, help="Run an evaluation every X steps.")

    # control behavior of script
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    # resave
    parser.add_argument("--do_resave", action="store_true", help="Whether to resave the model.")

    parser.add_argument("--do_analysis", action="store_true", help="Whether to run analysis on the dev set.")

    # wandb
    parser.add_argument(
        "--run_name",
        type=str,
        default="debug",
        help="A descriptor for the run. Typically the objective.",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable logging to Weights & Biases. By default, logging is enabled.",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="ETRASK", help="The name of the project to log to on Weights & Biases."
    )
    parser.add_argument("--wandb_tags", nargs="+", help="Tags to add to the run on Weights & Biases.")
    parser.add_argument("--wandb_group", type=str, help="The group of the run on Weights & Biases.")
    parser.add_argument("--wandb_id", type=str, help="The ID of the run on Weights & Biases to resume from.")

    # lora
    parser.add_argument("--lora", action="store_true", help="Use LORA")
    parser.add_argument(
        "--lora_r", type=int, default=8, help="The number of hops to use for the LORA model. Defaults to 8."
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="The alpha value to use for the LORA model. Defaults to 32."
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="The dropout rate to use for the LORA model. Defaults to 0.1."
    )
    parser.add_argument("--target_modules_embedding", action="store_true", help="Add LoRA to embedding layer")

    # retriever
    parser.add_argument("--n", "-n", type=int, default=0, help="The number of examples to retrieve. Defaults to 0.")
    parser.add_argument("--embedding_retriever", type=str, help="A type of embedding retriever. Defaults to None.")
    parser.add_argument("--embedding_retriever_distance", type=str, default="cos", help="Defaults to 'cos'.")
    parser.add_argument("--embedding_retriever_batch_size", type=int, default=1, help="Defaults to 1.")
    parser.add_argument("--embedding_retriever_device_map", type=str, help="Defaults to False.")
    parser.add_argument("--embedding_retriever_model", type=str, help="Defaults to None.")
    parser.add_argument("--embedding_retriever_load_path", type=str, help="Defaults to None.")
    parser.add_argument("--embedding_retriever_sampling", type=float, help="Defaults to None.")
    parser.add_argument("--embedding_retriever_update_step", type=int, help="Defaults to None.")
    parser.add_argument("--embedding_retriever_add_distance_token", action="store_true", help="Defaults to None.")
    parser.add_argument("--embedding_retriever_detect_span", action="store_true", help="Defaults to None.")
    parser.add_argument("--embedding_retriever_keep_prompt", action="store_true", help="Defaults to None.")
    parser.add_argument("--embedding_retriever_insert", action="store_true", help="Defaults to None.")

    parser.add_argument("--knn_embedding_retriever_max_sampling", type=int, help="Defaults to None.")
    parser.add_argument("--knn_embedding_retriever_top_k", type=int, help="Defaults to None.")
    parser.add_argument("--knn_embedding_retriever_differentiable", action="store_true", help="Defaults to None.")
    parser.add_argument("--knn_embedding_retriever_temperature", type=float, default=1.0, help="Defaults to None.")
    parser.add_argument("--knn_embedding_retriever_epsilon", type=float, default=0.0, help="Defaults to None.")
    parser.add_argument(
        "--knn_embedding_retriever_distance_reduction", type=str, default="mean", help="Defaults to mean."
    )
    parser.add_argument("--knn_embedding_retriever_distance_combination", action="store_true", help="Defaults to None.")

    parser.add_argument("--embedding_retriever_lora", action="store_true", help="Use LORA")
    parser.add_argument(
        "--embedding_retriever_lora_r",
        type=int,
        default=8,
        help="The number of hops to use for the LORA model. Defaults to 8.",
    )
    parser.add_argument(
        "--embedding_retriever_lora_alpha",
        type=int,
        default=32,
        help="The alpha value to use for the LORA model. Defaults to 32.",
    )
    parser.add_argument(
        "--embedding_retriever_lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate to use for the LORA model. Defaults to 0.1.",
    )

    parser.add_argument("--embedding_retriever_warmup_steps", type=int, default=0, help="Defaults to 0.")
    parser.add_argument("--embedding_retriever_lr", type=float, default=1e-3, help="Defaults to 1e-3.")
    parser.add_argument("--embedding_retriever_weight_decay", type=float, default=0.0, help="Defaults to 0.0.")

    parser.add_argument("--semeval", action="store_true", help="The dataset is semeval")

    args = parser.parse_args()

    # # Sanity checks
    # if args.dataset_name is None and args.train_file is None and args.validation_file is None:
    #     raise ValueError("Need either a dataset name or a training/validation file.")
    # else:
    #     if args.train_file is not None:
    #         extension = args.train_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #     if args.validation_file is not None:
    #         extension = args.validation_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
    args.output_dir = os.path.join(args.output_dir, args.run_name)

    return args


def main():
    args = parse_args()

    TARGET_MODULES = {
        "t5": ["q", "k", "v", "o", "wi", "wo", "wi_0", "wi_1"],
        "gpt2": ["attn.c_attn", "attn.c_proj"],
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bart": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "roberta": ["query", "key", "value", "dense"],
        "pegasus": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    }
    if args.target_modules_embedding:
        TARGET_MODULES["t5"].append("shared")
        TARGET_MODULES["gpt2"].append("wte")
        TARGET_MODULES["gpt2"].append("wpe")
        TARGET_MODULES["llama"].append("embed_tokens")
        TARGET_MODULES["bart"].append("shared")
        TARGET_MODULES["roberta"].append("word_embeddings")
        TARGET_MODULES["roberta"].append("position_embeddings")
        TARGET_MODULES["pegasus"].append("shared")

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(log_with="wandb")
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # wandb
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=args,
        init_kwargs={
            "wandb": {
                "tags": args.wandb_tags,
                "group": args.wandb_group,
                "id": args.wandb_id,
                "resume": "must" if args.wandb_id else None,
                "name": args.run_name,
                "mode": "disabled" if args.disable_wandb else None,
            }
        },
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    model_class = AutoModelForSeq2SeqLM
    # if args.model_name_or_path:
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            # config=config,
        )
    except:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_config(config)
        try:
            model = PeftModel.from_pretrained(model, model_id=args.model_name_or_path)
        except:
            pass

    # lora
    if args.lora and not hasattr(model, "peft_config"):
        target_modules = None
        for k, v in TARGET_MODULES.items():
            if k in args.model_name_or_path:
                target_modules = v
                break

        lora_config = LoraConfig(
            # task_type=TASKS[args.task_type],
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    additional_special_tokens = ["<e1>", "</e1>", "<e2>", "</e2>"]
    if args.type_file:
        with open(args.type_file) as f:
            types = json.load(f)
        for t in types:
            additional_special_tokens.append(f"<e1-{t}>")
            additional_special_tokens.append(f"</e1-{t}>")
            additional_special_tokens.append(f"<e2-{t}>")
            additional_special_tokens.append(f"</e2-{t}>")
    else:
        types = None
    special_tokens_dict = {"additional_special_tokens": additional_special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if args.template_file:
        with open(args.template_file) as f:
            templates = json.load(f)
        relation_map = {rel: i for i, rel in enumerate(list(templates.keys()))}
        relation_num = len(relation_map)

    if args.type_constraint_file:
        with open(args.type_constraint_file) as f:
            type_constraint_dict = json.load(f)
    else:
        type_constraint_dict = None

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # load dataset
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    assert isinstance(raw_datasets, datasets.DatasetDict)

    # Preprocessing the datasets.
    if "train" in raw_datasets or "validation" in raw_datasets:
        # do preprocess
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names

        # Get the column names for input/target.
        dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
        if args.text_column is None:
            text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            text_column = args.text_column
            if text_column not in column_names:
                raise ValueError(
                    f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
                )
        if args.summary_column is None:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            summary_column = args.summary_column
            if summary_column not in column_names:
                raise ValueError(
                    f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
                )

        # Temporarily set max_target_length for training.
        max_target_length = args.max_target_length
        padding = "max_length" if args.pad_to_max_length else False

        def preprocess_function(examples):
            inputs = examples[text_column]
            targets = examples[summary_column]
            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # small dataset for debugging
        if args.debug:
            raw_datasets["train"] = raw_datasets["train"].select(range(10))
            if "validation" in raw_datasets:
                raw_datasets["validation"] = raw_datasets["validation"].select(range(10))

        with accelerator.main_process_first():
            processed_dataset = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

    model_load_kwargs = {}

    # add retriever to the model
    retriever_optimizer = None
    if args.embedding_retriever and args.n:

        retriever_model = transformers.AutoModel.from_pretrained(args.embedding_retriever_model)

        retriever_tokenizer = transformers.AutoTokenizer.from_pretrained(args.embedding_retriever_model)

        # lora
        if args.embedding_retriever_lora and not hasattr(retriever_model, "peft_config"):

            # TASKS = {"seq2seq": peft.TaskType.SEQ_2_SEQ_LM, "lm": peft.TaskType.CAUSAL_LM, "seq_cls": peft.TaskType.SEQ_CLS}
            target_modules = None
            for k, v in TARGET_MODULES.items():
                if k in args.embedding_retriever_model:
                    target_modules = v
                    break

            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=args.embedding_retriever_lora_r,
                lora_alpha=args.embedding_retriever_lora_alpha,
                lora_dropout=args.embedding_retriever_lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            retriever_model = get_peft_model(retriever_model, lora_config)

        sep_tok = (
            retriever_tokenizer.sep_token
            if hasattr(retriever_tokenizer, "sep_token")
            else retriever_tokenizer.eos_token
        )
        collator = transformers.DataCollatorWithPadding(tokenizer=retriever_tokenizer)
        assert "train" in raw_datasets
        database: datasets.Dataset = processed_dataset["train"].select_columns(["input_ids", "labels"])
        # reconstruct text
        database_to_text = (
            lambda x: tokenizer.decode(x["input_ids"], skip_special_tokens=True)
            + " "
            + sep_tok
            + " "
            + tokenizer.decode(x["labels"], skip_special_tokens=True)
        )
        database = database.map(
            lambda x: retriever_tokenizer(database_to_text(x)),
            # batched=True,
            remove_columns=["labels"],
        )
        database = database.add_column("relation", raw_datasets["train"]["relation"])
        # subj and obj
        database = database.add_column("subj", raw_datasets["train"]["subj"])
        database = database.add_column("obj", raw_datasets["train"]["obj"])
        database = database.add_column("text", raw_datasets["train"]["text"])

        def span_detect_function(input_ids, *args, tokenizer=retriever_tokenizer, **kwds) -> dict:
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            text = tokenizer.decode(input_ids)
            # text between 'The head entity is ' and 'The tail entity is' is head entity
            # span : [start_index, tail_index]
            head_text_span = re.search(r"The head entity is(.*)The tail entity is", text).span(1)
            # text between 'The tail entity is ' and 'The type of' is tail entity
            try:
                tail_text_span = re.search(r"The tail entity is(.*)The type of", text).span(1)
            except:
                tail_text_span = re.search(r"The tail entity is(.*)\. ", text).span(1)
            # convert to input_ids index
            head_span = [
                len(tokenizer.encode(text[: head_text_span[0]])),
                len(tokenizer.encode(text[: head_text_span[1]])),
            ]
            tail_span = [
                len(tokenizer.encode(text[: tail_text_span[0]])),
                len(tokenizer.encode(text[: tail_text_span[1]])),
            ]
            # after first sep is label
            label_span = [input_ids.index(tokenizer.sep_token_id), len(input_ids) - 1]
            return {
                "head_span": torch.tensor(head_span),
                "tail_span": torch.tensor(tail_span),
                "label_span": torch.tensor(label_span),
            }

        if args.embedding_retriever == "knn":
            # TODO database, span_keys
            retriever = KNNEmbeddingRetriever(
                database=database,
                encoder=retriever_model,
                tokenizer=retriever_tokenizer,
                parent_tokenizer=tokenizer,
                distance=args.embedding_retriever_distance,
                update_batch_size=args.embedding_retriever_batch_size,
                collator=collator,
                sampling=args.embedding_retriever_sampling,
                seed=args.seed,
                n=args.n,
                max_sampling=args.knn_embedding_retriever_max_sampling,
                top_k=args.knn_embedding_retriever_top_k,
                add_distance_token=args.embedding_retriever_add_distance_token,
                differentiable=args.knn_embedding_retriever_differentiable,
                temperature=args.knn_embedding_retriever_temperature,
                epsilon=args.knn_embedding_retriever_epsilon,
                span_detect_function=span_detect_function if args.embedding_retriever_detect_span else None,
                span_keys=["head_span", "tail_span", "label_span"] if args.embedding_retriever_detect_span else None,
                distance_reduction=args.knn_embedding_retriever_distance_reduction,
                distance_combination=args.knn_embedding_retriever_distance_combination,
            )
            model_load_kwargs["retriever_class"] = KNNEmbeddingRetriever
        elif args.embedding_retriever == "random":
            retriever = RandomEmbeddingRetriever(
                database=database,
                parent_tokenizer=tokenizer,
                encoder=retriever_model,
                tokenizer=retriever_tokenizer,
                distance=args.embedding_retriever_distance,
                sampling=args.embedding_retriever_sampling,
                add_distance_token=args.embedding_retriever_add_distance_token,
                seed=args.seed,
                n=args.n,
                span_detect_function=span_detect_function if args.embedding_retriever_detect_span else None,
                span_keys=["head_span", "tail_span", "label_span"] if args.embedding_retriever_detect_span else None,
            )
            model_load_kwargs["retriever_class"] = RandomEmbeddingRetriever
        else:
            raise NotImplementedError("embedding_retriever {} not im`plemented".format(args.embedding_retriever))
        model_load_kwargs["retriever_kwargs"] = {
            "span_detect_function": span_detect_function if args.embedding_retriever_detect_span else None,
        }
        model_load_kwargs["insert"] = args.embedding_retriever_insert
        model_class = Seq2SeqModelWithRetriever
        model = model_class(model, retriever, insert=args.embedding_retriever_insert)

        # retriever optimizer
        # parameters for retriever optimizer is one which does not have `base_model` in its name
        no_decay = ["bias", "LayerNorm.weight"]
        # retriever_optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model.named_parameters() if not "base_model" in n],
        #         "weight_decay": args.embedding_retriever_weight_decay,
        #     }
        # ]
        retriever_optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not "base_model" in n and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.embedding_retriever_weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if not "base_model" in n and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        retriever_optimizer = AdamW(retriever_optimizer_grouped_parameters, lr=args.embedding_retriever_lr, eps=1e-8)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if "base_model" in n and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "base_model" in n and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    else:
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if "pegasus" in args.model_name_or_path and args.use_adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=args.learning_rate, relative_step=False, scale_parameter=False
            )
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    logger.info(f"Optimizer: {str(optimizer)}\n")

    # if args.load_dir:
    #     # Load from a local directory
    #     logger.info(f"Loaded model from {args.load_dir}")
    #     model = model_class.from_pretrained(args.load_dir, **model_load_kwargs)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Metric
    # metric = load_metric("rouge")
    trie = PredictTrie(tokenizer, force_end="bart" in args.model_name_or_path)
    eval_data = None
    if args.validation_file:
        eval_data = []
        with open(args.validation_file) as f:
            line = f.readline()
            while line:
                eval_data.append(json.loads(line))
                line = f.readline()
        if args.debug:
            eval_data = eval_data[:10]

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if args.use_label_smoother:
        label_smoother = LabelSmoother()

    def predict(model, tokenizer, dataset, return_retrieval=False, generator=False):
        model.eval()
        trues = []
        preds = []
        retrieval_results = []
        for each in tqdm(dataset):
            subj = each["subj"]
            obj = each["obj"]
            true = each["relation"]
            sentence = each["text"]

            inputs = tokenizer.batch_encode_plus(
                [sentence], return_tensors="pt", max_length=256, padding="max_length", truncation=False
            ).to(accelerator.device)

            if types:
                subj_type = each["subj_type"]
                obj_type = each["obj_type"]
                if type_constraint_dict:
                    temp = {"no_relation": templates["no_relation"]}
                    for rel in templates:
                        if rel in type_constraint_dict[f"{subj_type}|{obj_type}"]:
                            temp[rel] = templates[rel]
                else:
                    temp = templates
            else:
                temp = templates

            score = trie.predict(
                subj,
                obj,
                temp,
                inputs,
                model,
                keep_prompt=args.embedding_retriever_keep_prompt and args.embedding_retriever,
                return_retrieval=return_retrieval,
            )
            if return_retrieval:
                score, retrieval = score
            p = sorted(score.items(), key=lambda x: -x[1])
            if p[0][0] == "no_relation":
                p_score = p[1]
            else:
                p_score = p[0]
            na_score = score["no_relation"] if "no_relation" in score else score["Other"]

            if na_score > p_score[1]:
                p = relation_map["no_relation"]
            else:
                p = relation_map[p_score[0]]
            if generator:
                ret = {"gold": relation_map[true], "pred": p}
                if return_retrieval:
                    ret["retrieval"] = retrieval
                yield ret
            else:
                preds.append(p)
                trues.append(relation_map[true])
                if return_retrieval:
                    retrieval_results.append(retrieval)
        ret = {"gold": trues, "pred": preds}
        if return_retrieval:
            ret["retrieval"] = retrieval_results
        if generator:
            return
        return ret

    def eval(model, tokenizer, eval_data) -> dict[str, Any]:
        """eval using pred function"""
        o = predict(model, tokenizer, eval_data)
        trues = o["gold"]
        preds = o["pred"]
        precision = precision_score(trues, preds, labels=range(1, relation_num), average="micro")
        recall = recall_score(trues, preds, labels=range(1, relation_num), average="micro")
        f1 = f1_score(trues, preds, labels=range(1, relation_num), average="micro")
        ret: dict[str, Any] = {"precision": precision, "recall": recall, "f1": f1}
        if args.semeval:
            from semeval.evaluate import evaluate

            # convert label to string
            rev_rel_map = {v: k for k, v in relation_map.items()}
            trues = [rev_rel_map[i] for i in trues]
            preds = [rev_rel_map[i] for i in preds]
            metric = evaluate(trues, preds)
            # ret["official_raw"] = metric
            of = {
                "official/" + k: v
                for k, v in metric["official"].items()
                if k not in ["confusion_matrix", "other", "class_scores"]
            }
            ret.update(of)
        return ret

    if args.do_resave:
        # load model parameter and save it again to update format
        # first hold the dataset
        from diff_knn_retriever.retriever.base import save_dataset

        database = retriever.database

        retriv_kwds = model_load_kwargs.get("retriever_kwargs", {})
        model_load_kwargs["retriever_kwargs"]["legacy"] = True
        kwds = deepcopy(model_load_kwargs)
        kwds["retriever_kwargs"] = retriv_kwds
        mdl = model_class.from_pretrained(args.output_dir, **kwds)
        mdl.retriever.database = database
        mdl.save_pretrained(args.output_dir)
        del mdl

    if args.do_train:
        label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
        train_dataloader = DataLoader(
            processed_dataset["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
        )
        eval_dataloader = (
            DataLoader(
                processed_dataset["validation"], collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
            )
            if "validation" in processed_dataset
            else None
        )
        if eval_dataloader is None:
            model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        else:
            # Prepare everything with our `accelerator`.
            model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader
            )

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
        retriever_lr_scheduler = (
            get_scheduler(
                name=args.lr_scheduler_type,
                optimizer=retriever_optimizer,
                num_warmup_steps=args.embedding_retriever_warmup_steps,
                num_training_steps=args.max_train_steps,
            )
            if retriever_optimizer
            else None
        )
        logger.info("***** Running training *****")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        global_step = 0

        if hasattr(model, "update_retriever"):
            model.update_retriever()
        best_score = float("inf")
        global_step = 0
        eval_steps = 0
        patience = 0
        should_stop = False
        for epoch in range(args.num_train_epochs):
            model.train()
            loss_for_log = 0.0
            log_dict = {}
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                if args.use_label_smoother:
                    loss = label_smoother(outputs, batch["labels"])
                else:
                    loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                loss_for_log += loss.item()
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    # at warmup stage, only update retriever
                    # therefore, skip base model optimizer at warmup stage
                    if (
                        args.embedding_retriever_warmup_steps is not None
                        and global_step < args.embedding_retriever_warmup_steps
                        and retriever_optimizer
                    ):
                        assert retriever_lr_scheduler is not None
                        retriever_optimizer.step()
                        retriever_lr_scheduler.step()
                        retriever_optimizer.zero_grad()
                    else:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        if retriever_optimizer:
                            assert retriever_lr_scheduler is not None
                            retriever_optimizer.step()
                            retriever_lr_scheduler.step()
                            retriever_optimizer.zero_grad()
                    progress_bar.update(1)

                    log_dict = {"train/loss": loss_for_log}
                    loss_for_log = 0.0

                    # run update for retriever
                    if (
                        args.embedding_retriever_update_step is not None
                        and global_step % args.embedding_retriever_update_step == 0
                    ):
                        if hasattr(model, "update_retriever"):
                            model.update_retriever()

                    # evaluation term
                    with torch.no_grad():
                        if global_step % args.eval_steps == 0 and global_step > 0:
                            all_loss = 0.0
                            steps = 0.0
                            if eval_dataloader is not None:
                                for step, batch in enumerate(eval_dataloader):
                                    outputs = model(**batch)
                                    loss = outputs.loss
                                    all_loss += loss.item()
                                    steps += 1
                                logger.info(f"eval loss: {all_loss / steps}\n")
                                log_dict.update({"dev/loss": all_loss / steps})
                            if not args.eval_loss_only and eval_data is not None:
                                metric = eval(model, tokenizer, eval_data)
                                logger.info(f"metric: {metric}\n")
                                # rewrite key name
                                metric = {f"dev/{k}": v for k, v in metric.items()}
                                log_dict.update(metric)

                                # if f1 > best_f1:
                                #     best_f1 = f1
                                #     accelerator.wait_for_everyone()
                                #     unwrapped_model = accelerator.unwrap_model(model)
                                #     unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                                #     if accelerator.is_main_process:
                                #         tokenizer.save_pretrained(args.output_dir)
                                #     logger.info(f"saving pretrained models with f1 {f1}\n")
                            # early stopping
                            if args.early_stopping and global_step > 0 and eval_steps >= args.early_stopping_warmup:
                                score = log_dict[args.early_stopping_metric]
                                if args.early_stopping_maximize:
                                    score = -score
                                if score < best_score:
                                    best_score = score
                                    patience = 0
                                    accelerator.wait_for_everyone()
                                    unwrapped_model = accelerator.unwrap_model(model)
                                    unwrapped_model.save_pretrained(args.output_dir)
                                    tokenizer.save_pretrained(args.output_dir)
                                    logger.info(f"saving pretrained models with score {score}\n")
                                else:
                                    patience += 1
                                    if patience > args.early_stopping_patience:
                                        logger.info(f"early stopping with score {score}\n")
                                        should_stop = True
                                        break
                            eval_steps += 1
                        # log
                        if accelerator.is_local_main_process and global_step % args.log_steps == 0:
                            accelerator.log(log_dict, step=global_step)

                    global_step += 1
                    if global_step >= args.max_train_steps:
                        should_stop = True
                        break
                    if should_stop:
                        break
                if should_stop:
                    break
            if should_stop:
                break
        # save last model
        last_dir = os.path.join(args.output_dir, "last")
        os.makedirs(last_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)
        logger.info(f"saving pretrained models with final step {global_step}\n")

    if args.eval_load_dir is None:
        eval_load_dir = args.output_dir
        if args.eval_load_last:
            eval_load_dir = os.path.join(args.output_dir, "last")
    else:
        eval_load_dir = args.eval_load_dir
    if args.do_eval:
        # evaluation
        # load best model
        model = model_class.from_pretrained(eval_load_dir, **model_load_kwargs)
        model = accelerator.prepare(model)
        model.eval()
        if hasattr(model, "update_retriever"):
            model.update_retriever()
        metric = eval(model, tokenizer, eval_data)
        logger.info(f"metric: {metric}\n")
        # rewrite key name
        metric = {f"dev_eval/{k}": v for k, v in metric.items()}
        accelerator.log(metric, step=global_step)

    if args.do_test:
        if not args.do_eval:
            # load best model
            model = model_class.from_pretrained(eval_load_dir, **model_load_kwargs)
            model = accelerator.prepare(model)
            if hasattr(model, "update_retriever"):
                model.update_retriever()
        test_data = []
        with open(args.test_file) as f:
            line = f.readline()
            while line:
                test_data.append(json.loads(line))
                line = f.readline()
        model.eval()
        metric = eval(model, tokenizer, test_data)
        logger.info(f"metric: {metric}\n")
        # rewrite key name
        metric = {f"test/{k}": v for k, v in metric.items()}
        accelerator.log(metric, step=global_step)

    # analysis
    def analyse(model, tokenizer, dataset):
        # predict
        # o = predict(model, tokenizer, dataset, return_retrieval=True)
        # trues = np.array(o["gold"])
        # preds = np.array(o["pred"])
        # retrieval_results = o["retrieval"]

        # retrieved_examples = []
        # retrieved_classes = []
        # for ret in retrieval_results:
        #     # select exmaples
        #     idx = ret.indices.flatten().tolist()
        #     # c = [relation_map[e["relation"]] for e in ret.examples]
        #     e = [ret.examples[x] for x in idx]
        #     retrieved_examples.append(e)
        #     c = [relation_map[ret.examples[x]["relation"]] for x in idx]
        #     retrieved_classes.append(c)

        trues = []
        preds = []
        retrieved_examples = []
        retrieved_classes = []
        for o in predict(model, tokenizer, dataset, return_retrieval=True, generator=True):
            trues.append(o["gold"])
            preds.append(o["pred"])
            ret = o["retrieval"]
            # select exmaples
            idx = ret.indices.flatten().tolist()
            # c = [relation_map[e["relation"]] for e in ret.examples]
            e = [ret.examples[x] for x in idx]
            retrieved_examples.append(e)
            c = [relation_map[ret.examples[x]["relation"]] for x in idx]
            retrieved_classes.append(c)

        report = {}

        # analyse
        # confusion matrix with wandb module
        confusion_matrix = wandb.plot.confusion_matrix(y_true=trues, preds=preds, class_names=list(relation_map.keys()))
        report["confusion_matrix"] = confusion_matrix

        # retriever analysis
        # is_same_label : shape (len(dataset), len(retrieved_examples[0]))
        is_same_label = np.array(list([[t == c for c in retrieved_classes[i]] for i, t in enumerate(trues)]))
        # create as confusion matrix
        retrieves = np.array(list(itertools.chain.from_iterable(retrieved_classes)))
        queries = np.array(
            list(itertools.chain.from_iterable([[trues[i]] * len(c) for i, c in enumerate(retrieved_classes)]))
        )
        retriever_confusion_matrix = wandb.plot.confusion_matrix(
            y_true=queries, preds=retrieves, class_names=list(relation_map.keys())
        )
        report["retriever_confusion_matrix"] = retriever_confusion_matrix
        # top-1 confusion matrix
        top1_retrieves = np.array(list(itertools.chain.from_iterable([c[:1] for c in retrieved_classes])))
        top1_retriever_confusion_matrix = wandb.plot.confusion_matrix(
            y_true=trues, preds=top1_retrieves, class_names=list(relation_map.keys())
        )
        report["retriever_top1_confusion_matrix"] = top1_retriever_confusion_matrix

        # accuracy
        retriever_accuracy = accuracy_score(queries, retrieves)
        report["retriever_accuracy"] = retriever_accuracy

        # top-n accuracy
        for n in [1, 3, 5, 10]:
            if len(retrieved_classes[0]) < n:
                continue
            topn_retrieves = np.array(list(itertools.chain.from_iterable([c[:n] for c in retrieved_classes])))
            topn_trues = np.array(list(itertools.chain.from_iterable([t] * n for t in trues)))
            topn_retriever_accuracy = accuracy_score(topn_trues, topn_retrieves)
            report[f"retriever_top{n}_accuracy"] = topn_retriever_accuracy

        # entity analysis
        # subject inclusion and object inclusion
        subject_inclusion = []
        object_inclusion = []
        has_same_entity = []
        for i, e in enumerate(retrieved_examples):
            subj = dataset[i]["subj"]
            obj = dataset[i]["obj"]
            sbj_inc = [subj in x["text"] for x in e]
            obj_inc = [obj in x["text"] for x in e]
            subject_inclusion.append(sbj_inc)
            object_inclusion.append(obj_inc)
            # has same entity
            # dataset[i]['subj'] in {e['subj'], e['obj']}
            has_same_entity.append(
                [dataset[i]["subj"] in {x["subj"], x["obj"]} or dataset[i]["obj"] in {x["subj"], x["obj"]} for x in e]
            )
        # convert to report
        # top-n subject and object coverage
        for n in [1, 3, 5, 10]:
            if len(retrieved_examples[0]) < n:
                continue
            topn_subject_cov = np.array(list([any(c[:n]) for c in subject_inclusion]))
            topn_object_cov = np.array(list([any(c[:n]) for c in object_inclusion]))
            report[f"retriever_top{n}_subject_inclusion"] = topn_subject_cov.mean()
            report[f"retriever_top{n}_object_inclusion"] = topn_object_cov.mean()
            # take or
            report[f"retriever_top{n}_entity_inclusion"] = np.array(
                list([any(s[:n]) or any(o[:n]) for s, o in zip(subject_inclusion, object_inclusion)])
            ).mean()

        # subject and object cov
        subject_cov = np.array(list([any(c) for c in subject_inclusion]))
        object_cov = np.array(list([any(c) for c in object_inclusion]))
        report["retriever_subject_inclusion"] = subject_cov.mean()
        report["retriever_object_inclusion"] = object_cov.mean()
        # entity inclusion
        # take or
        report["retriever_entity_inclusion"] = np.array(
            list([any(s) or any(o) for s, o in zip(subject_inclusion, object_inclusion)])
        ).mean()

        # top-n subject and object accuracy
        for n in [1, 3, 5, 10]:
            if len(retrieved_examples[0]) < n:
                continue
            # subject
            topn_subject_acc = np.array(list([any(c[:n]) for c in subject_inclusion]))
            report[f"retriever_top{n}_subject_accuracy"] = topn_subject_acc.mean()
            # object
            topn_object_acc = np.array(list([any(c[:n]) for c in object_inclusion]))
            report[f"retriever_top{n}_object_accuracy"] = topn_object_acc.mean()
            # take or
            report[f"retriever_top{n}_entity_accuracy"] = np.array(
                list([any(s[:n]) or any(o[:n]) for s, o in zip(subject_inclusion, object_inclusion)])
            ).mean()
        # subject and object accuracy
        subject_acc = np.array(list([any(c) for c in subject_inclusion]))
        object_acc = np.array(list([any(c) for c in object_inclusion]))
        report["retriever_subject_accuracy"] = subject_acc.mean()
        report["retriever_object_accuracy"] = object_acc.mean()
        # entity accuracy
        report["retriever_entity_accuracy"] = np.array(
            list([any(s) or any(o) for s, o in zip(subject_inclusion, object_inclusion)])
        ).mean()

        # top-n entity coverage
        for n in [1, 3, 5, 10]:
            if len(retrieved_examples[0]) < n:
                continue
            topn_entity_cov = np.array(list([any(c[:n]) for c in has_same_entity]))
            report[f"retriever_top{n}_entity_coverage"] = topn_entity_cov.mean()
        # entity coverage
        entity_cov = np.array(list([any(c) for c in has_same_entity]))
        report["retriever_entity_coverage"] = entity_cov.mean()

        # include entity (subject or object) or have same relation label (is_same_label)
        # top-n
        for n in [1, 3, 5, 10]:
            if len(retrieved_examples[0]) < n:
                continue
            topn_entity_or_label = np.array(
                list(
                    [
                        any(s[:n]) or any(o[:n]) or any(l[:n])
                        for s, o, l in zip(subject_inclusion, object_inclusion, is_same_label)
                    ]
                )
            )
            report[f"retriever_top{n}_entity_or_label"] = topn_entity_or_label.mean()
        # all
        entity_or_label = np.array(
            list([any(s) or any(o) or any(l) for s, o, l in zip(subject_inclusion, object_inclusion, is_same_label)])
        )
        report["retriever_entity_or_label"] = entity_or_label.mean()

        # top-n entity or label accuracy
        # marge same_label and subject_inclusion and object_inclusion
        has_label_or_entity = np.array(
            [
                [s or o or l for s, o, l in zip(subject_inclusion[i], object_inclusion[i], is_same_label[i])]
                for i in range(len(dataset))
            ]
        )

        for n in [1, 3, 5, 10]:
            if len(retrieved_examples[0]) < n:
                continue
            report[f"retriever_top{n}_entity_or_label_accuracy"] = has_label_or_entity[:n].mean()
        # all
        report["retriever_entity_or_label_accuracy"] = has_label_or_entity.mean()

        def get_sent(text: str) -> str:
            sents = nltk.sent_tokenize(text)
            return " ".join(sents[4:])

        # retrieved examples attribute
        pairs = []
        retrieved_example_attributes = [[set() for _ in range(len(retrieved_examples[0]))] for _ in range(len(dataset))]
        for i, e in enumerate(retrieved_examples):
            for j, x in enumerate(e):
                query = dataset[i]
                # whether the retrieved example has same subj
                if query["subj"] in {x["subj"], x["obj"]}:
                    retrieved_example_attributes[i][j].add("subj")
                # whether the retrieved example has same obj
                if query["obj"] in {x["subj"], x["obj"]}:
                    retrieved_example_attributes[i][j].add("obj")
                # whether the retrieved example has same relation
                if query["relation"] == x["relation"]:
                    retrieved_example_attributes[i][j].add("relation")
                # whether the retrieved example has subj in text
                if query["subj"] in x["text"]:
                    retrieved_example_attributes[i][j].add("subj_in_text")
                # whether the retrieved example has obj in text
                if query["obj"] in x["text"]:
                    retrieved_example_attributes[i][j].add("obj_in_text")
                # vocab overlap
                query_text = get_sent(query["text"])
                words = set(nltk.word_tokenize(query_text))
                retriv_text = get_sent(x["text"])
                # retriv_words = set(nltk.word_tokenize(retriv_text))
                # Jaccard similarity for n-gram vocabulary
                # ratio = 0.2
                # for n in [1, 2, 3]:
                #     words_ngram = set(nltk.ngrams(words, n))
                #     retriv_words_ngram = set(nltk.ngrams(retriv_words, n))
                #     # for ratio in [0.9, 0.7, 0.5, 0.3]:
                #     if len(words_ngram.intersection(retriv_words_ngram)) / len(words_ngram) > ratio:
                #         retrieved_example_attributes[i][j].add(f"vocab_{n}-gram_{ratio}")
                #         pairs.append((query_text, retriv_text))
                #         break
                if len(retrieved_example_attributes[i][j]) == 0:
                    pairs.append((query_text, retriv_text))
        # save non atributed examples
        dat = []
        for i, e in enumerate(retrieved_examples):
            query = dataset[i]
            query_text = get_sent(query["text"])
            retriv = []
            for j, x in enumerate(e):
                retriv_text = get_sent(x["text"])
                if len(retrieved_example_attributes[i][j]) == 0:
                    retriv.append({"text": retriv_text, "relation": x["relation"], "subj": x["subj"], "obj": x["obj"]})
            dat.append(
                {
                    "query": {
                        "text": query_text,
                        "relation": query["relation"],
                        "subj": query["subj"],
                        "obj": query["obj"],
                    },
                    "retrieved": retriv,
                }
            )
        a = np.array([[len(r) > 0 for r in ret] for ret in retrieved_example_attributes]).mean()

        # vocabulary analysis
        # sim = defaultdict(list)
        # for i in range(len(dataset)):
        #     words = nltk.word_tokenize(dataset[i]["text"])
        #     retriv_words = [nltk.word_tokenize(e["text"]) for e in retrieved_examples[i]]
        #     # Jaccard similarity for n-gram vocabulary
        #     for n in [1, 2, 3]:
        #         words_ngram = set(nltk.ngrams(words, n))
        #         retriv_words_ngram = [set(nltk.ngrams(x, n)) for x in retriv_words]
        #         sim[n].append(
        #             [
        #                 len(words_ngram.intersection(x)) / len(words_ngram.union(x))
        #                 for x in retriv_words_ngram
        #                 if len(words_ngram.union(x)) > 0
        #             ]
        #         )
        # # convert to report
        # # top-k n-gram vocabulary sim
        # for k in [1, 3, 5, 10]:
        #     if len(sim[1][0]) < k:
        #         continue
        #     for n in [1, 2, 3]:
        #         sim_n = np.array(list([np.mean(c[:k]) for c in sim[n]]))
        #         # report[f"retriever_top{k}_{n}_gram_sim"] = sim_n.mean()

        return report

    if args.do_analysis:
        # load best model
        if not (args.do_eval or args.do_test):
            model = model_class.from_pretrained(eval_load_dir, **model_load_kwargs)
            model = accelerator.prepare(model)
            if hasattr(model, "update_retriever"):
                model.update_retriever()
        model.eval()
        report = analyse(model, tokenizer, eval_data)
        # report to wandb with appending prefix
        report = {f"analysis/{k}": v for k, v in report.items()}
        accelerator.log(report)
        print(report)

    # end tracker
    accelerator.end_training()


if __name__ == "__main__":
    main()

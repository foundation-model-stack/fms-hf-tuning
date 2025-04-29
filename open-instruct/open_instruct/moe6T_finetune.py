# !/usr/bin/env python
# coding=utf-8
# Copyright 2024 AllenAI. All rights reserved.
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

import functools
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Dict, List, Literal, Optional, Union

import datasets
import deepspeed
import torch
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
# == DQA Note: original OpenInstruct uses cuda 12.1, but ete team suggests to use 12.6, which makes Pytorch 2.6
# This causes the error in load_state() of a prev.checkpoint due to error: _pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, [1mdo those steps only if you trust the source of the checkpoint[0m.
# (1) In PyTorch 2.6, we changed the default value of the weights_only argument in torch.l
# So, solution: either re-install OpenInstruct conda env to torch==2.5.1+cu121 (in requirements.txt),
# or adding below 2 lines to work with new Pytorch 2.6
# NOTE: if this addition does not work for checkpoint resuming in Pytorch 2.6, resinstall open-instruct-env with its original requirement.txt
# from deepspeed.runtime.zero.config import ZeroStageEnum
# torch.serialization.add_safe_globals([ZeroStageEnum])
# == DQA
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from huggingface_hub import HfApi
from open_instruct.dataset_transformation import (INPUT_IDS_KEY,
                                                  TOKENIZED_SFT_DATASET_KEYS,
                                                  TokenizerConfig,
                                                  get_cached_dataset_tulu,
                                                  visualize_token)
from open_instruct.model_utils import (push_folder_to_hub, save_state,
                                       save_with_accelerate,
                                       save_with_accelerate_for_moe_kernels)
from open_instruct.utils import (ArgumentParserPlus, clean_last_n_checkpoints,
                                 get_last_checkpoint_path, get_wandb_tags,
                                 is_beaker_job, launch_ai2_evals_on_weka,
                                 maybe_get_beaker_config,
                                 maybe_use_ai2_hf_entity,
                                 maybe_use_ai2_wandb_entity)
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from rich.pretty import pprint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, PreTrainedTokenizer,
                          get_scheduler)

## ::FMS-ACCELERATION:: MoE Kernels
from tuning.config.acceleration_configs.acceleration_framework_config import \
    AccelerationFrameworkConfig
from tuning.config.acceleration_configs.fast_moe import FastMoe, FastMoeConfig

logger = get_logger(__name__)


@dataclass
class FlatArguments:
    """
    Full arguments class for all fine-tuning jobs.
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention in the model training"},
    )
    model_revision: Optional[str] = field(
        default=None,
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_mixer: Optional[dict] = field(
        default=None,
        metadata={"help": "A dictionary of datasets (local or HF) to sample from."},
    )

    dataset_mixer_list: List[str] = field(
        default_factory=lambda: ["allenai/tulu-3-sft-personas-algebra", "1.0"]
    )
    """A list of datasets (local or HF) to sample from."""

    dataset_mixer_list_splits: List[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""

    # == DQA: FUNCTIONS USED TO TRANSFORM TRAINING DATASET (USED IN dataset_transformation.py)
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: [
            "sft_tulu_tokenize_and_truncate_v1",
            "sft_tulu_filter_v1",
        ]
    )
    """The list of transform functions to apply to the dataset."""

    dataset_target_columns: List[str] = field(
        default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS
    )
    """The columns to use for the dataset."""

    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""

    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""

    dataset_config_hash: Optional[str] = None
    """The hash of the dataset configuration."""

    dataset_skip_cache: bool = False  # == DQA: NOT to read data from cache?!
    """Whether to skip the cache."""

    dataset_mix_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory to save the mixed dataset to disk."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a json/jsonl file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    clip_grad_norm: float = field(
        default=-1,
        metadata={
            "help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer."},
    )
    logging_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Log the training loss and learning rate every logging_steps steps."
        },
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."},
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use for learning rate adjustment.",
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
        },
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Total number of training epochs to perform."},
    )
    output_dir: str = field(
        default="output/",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    use_lora: bool = field(
        default=False,
        metadata={
            "help": "If True, will use LORA (low-rank parameter-efficient training) to train the model."
        },
    )
    use_qlora: bool = field(
        default=False,
        metadata={
            "help": "Use qLoRA training - initializes model in quantized form. Not compatible with deepspeed."
        },
    )
    use_8bit_optimizer: bool = field(
        default=False,
        metadata={
            "help": "Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed."
        },
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."},
    )
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the training process in seconds."
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    reduce_loss: str = field(
        default="mean",
        metadata={
            "help": "How to reduce loss over tokens. Options are 'mean' or 'sum'."
            "Using 'sum' can improve chat model performance."
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "If the training should continue from a checkpoint folder."},
    )
    report_to: Union[str, List[str]] = field(
        default="all",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    save_to_hub: Optional[str] = field(
        default=None,
        metadata={
            "help": "Save the model to the Hub under this name. E.g allenai/your-model"
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Turn on gradient checkpointing. Saves memory but slows training."
        },
    )
    use_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether to use LigerKernel for training."},
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, overrides the number of training steps. Otherwise, num_train_epochs is used."
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization and dataset shuffling."},
    )
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."  # noqa
        },
    )
    keep_last_n_checkpoints: int = field(
        default=3,
        metadata={
            "help": "How many checkpoints to keep in the output directory. -1 for all."
        },
    )
    fused_optimizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use fused AdamW or not.",
        },
    )
    load_balancing_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to include a load balancing loss (for OLMoE) or not.",
        },
    )
    load_balancing_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for load balancing loss if applicable."},
    )

    # == DQA: Add new special tokens:
    add_special_tokens: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of additional special tokens to add to the tokenizer"},
    )

    # ::FMS-ACCELERATION:: MoE Kernels
    fast_moe: str = field(
        default=None,
        metadata={"help": "activate fast moe kernels and EP"},
    )

    # Experiment tracking
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = False  # DQA: from True
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    try_launch_beaker_eval_jobs: bool = False  # DQA: from True
    """Whether to launch beaker evaluation jobs after training"""
    hf_metadata_dataset: Optional[str] = ""  # from "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""

    # Ai2 specific settings
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: Optional[str] = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: Optional[List[str]] = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""

    def __post_init__(self):
        if self.reduce_loss not in ["mean", "sum"]:
            raise ValueError("reduce_loss must be either 'mean' or 'sum'")
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.dataset_mixer is None
            and self.dataset_mixer_list is None
        ):
            raise ValueError(
                "Need either a dataset name, dataset mixer, or a training file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "json",
                    "jsonl",
                ], "`train_file` should be a json or a jsonl file."
        if (
            (
                self.dataset_name is not None
                and (
                    self.dataset_mixer is not None
                    or self.dataset_mixer_list is not None
                )
            )
            or (self.dataset_name is not None and self.train_file is not None)
            or (
                (self.dataset_mixer is not None or self.dataset_mixer_list is not None)
                and self.train_file is not None
            )
            or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")
        if self.try_launch_beaker_eval_jobs and not self.push_to_hub:
            raise ValueError(
                "Cannot launch Beaker evaluation jobs without pushing to the Hub."
            )


def debug_apply_chat_template_and_tokenize(
    tokenizer: PreTrainedTokenizer, messages: Optional[List[Dict[str, str]]] = None
) -> None:
    """
    Applies the chat template to messages and tokenizes the resulting text.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer instance.
        messages (Optional[List[Dict[str, str]]]): List of message dictionaries with "role" and "content".
    """
    print(
        f"\n== Vocab: ({len(tokenizer):,} - {tokenizer.vocab_size:,}) "
        f"tokenizer.special_tokens_map (len={len(tokenizer.special_tokens_map)}): {tokenizer.special_tokens_map}"
    )

    if messages is None:
        messages = [
            {"role": "user", "content": "Who?"},
            {"role": "assistant", "content": "LLM"},
        ]

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"\n== messages:\n{messages}")
    print(f"\n== apply_chat_template(messages):\n{text}")

    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]
    zipped = list(zip(tokens, decoded_tokens))
    print(f"{repr(text)} \n-> (len:{len(tokens)}) {tokens}\n")
    for i, (token_id, token_str) in enumerate(zipped):
        print(f"{token_id:6d} -> `{token_str}`")


def debug_STOP(accelerator: Accelerator) -> None:
    """
    Stops debugging and cleans up distributed resources.

    Args:
        accelerator (Accelerator): The accelerator instance managing distributed training.
    """
    print("== STOP DEBUGGING AND CLEANING UP RESOURCES ==")
    accelerator.wait_for_everyone()
    # Attempt to clean up distributed resources
    accelerator.end_training()
    accelerator.free_memory()
    sys.exit(0)


def main(args: FlatArguments, tc: TokenizerConfig):
    # ------------------------------------------------------------
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    # TODO: chk args (goal: tc.chat_template_name)
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )

    ## ::FMS-ACCELERATION:: MoE Kernels
    fast_moe_config = None
    if args.fast_moe is not None:
        if args.fast_moe.isdigit():
            args.fast_moe = int(args.fast_moe)
        elif args.fast_moe.lower() == "true":
            args.fast_moe = True
        elif args.fast_moe.lower() == "false":
            args.fast_moe = False
        fast_moe_config = FastMoeConfig(fast_moe=FastMoe(ep_degree=args.fast_moe))
    framework = AccelerationFrameworkConfig.from_dataclasses(
        fast_moe_config,
        None,
        None,
        None,
    ).get_framework()
    # default case
    if framework is not None:
        logger.info("NOTE: fast moe kernels is activated.")
    model_loader = AutoModelForCausalLM.from_pretrained
    if framework is not None and framework.requires_custom_loading:
        model_loader = framework.model_loader  # drop-in new loader
    # ------------------------------------------------------------
    # Setup tokenizer
    tc.tokenizer_revision = (
        args.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    )
    tc.tokenizer_name_or_path = (
        args.model_name_or_path
        if tc.tokenizer_name_or_path is None
        else tc.tokenizer_name_or_path
    )
    if (
        tc.tokenizer_revision != args.model_revision
        and tc.tokenizer_name_or_path != args.model_name_or_path
    ):
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{args.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{args.model_name_or_path=}`."""
        logger.warning(warning)

    # == DQA: add special chat tokens:
    if args.add_special_tokens is not None:
        existing_special_tokens = tc.tokenizer.special_tokens_map.get(
            "additional_special_tokens", []
        )
        new_special_tokens = [
            t for t in args.add_special_tokens if t not in existing_special_tokens
        ]
        if new_special_tokens:
            all_special_tokens = existing_special_tokens + new_special_tokens
            tc.tokenizer.add_special_tokens(
                {"additional_special_tokens": all_special_tokens}
            )
            if accelerator.is_main_process:
                print(
                    f"\n== Updated special tokens ({len(existing_special_tokens)} -> {len(all_special_tokens)}): {all_special_tokens}"
                )

    tokenizer = tc.tokenizer

    if accelerator.is_main_process:
        # == DQA: quick test on applying chat template + tokenizer on a simple example:
        debug_apply_chat_template_and_tokenize(tokenizer)

    # ------------------------------------------------------------
    # Set up runtime variables
    # == DQA DEFINE CHECKPOINT FOLDER:
    # args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.run_name = f"checkpoint"  # == DQA: for a fixed folder
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = (
            "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        )
    if args.push_to_hub and accelerator.is_main_process:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = (
            f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
        )
        if is_beaker_job():
            beaker_config = maybe_get_beaker_config()

    # ------------------------------------------------------------
    # Initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]

        # (Optional) Ai2 internal tracking
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
        if accelerator.is_main_process and is_beaker_job():
            experiment_config.update(vars(beaker_config))
        experiment_config.update(vars(tc))
        accelerator.init_trackers(
            args.wandb_project_name,
            experiment_config,
            init_kwargs={
                "wandb": {
                    "name": args.run_name,
                    "entity": args.wandb_entity,
                    "tags": [args.exp_name] + get_wandb_tags(),
                }
            },
        )
        wandb_tracker = accelerator.get_tracker("wandb")

    if accelerator.is_main_process:
        # == DQA: PRINT OUT FINAL PARAMS
        # pprint([args, tc])
        print(
            "\n============================= FlatArguments (args) ============================="
        )
        for key, value in asdict(args).items():
            print(f"{key:30s}: {value}")

        print(
            "\n============================= TokenizerConfig (tc) ============================="
        )
        for key, value in asdict(tc).items():
            print(f"{key:30s}: {value}")
        print(
            "\n================================================================================"
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()  # == DQA: added to avoid below makedirs creating multiple chkpt folders

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.dataset_mixer is not None:
        # == 00-DQA: Show how to define data mixture (ds: pct)
        args.dataset_mixer_list = [
            item for pair in args.dataset_mixer.items() for item in pair
        ]
    with accelerator.main_process_first():
        transform_fn_args = [
            {"max_seq_length": args.max_seq_length},
            {},
        ]

        if accelerator.is_main_process:

            def tokenize_and_print(text, tokenizer):
                print(
                    f"\n== len(tokenizer): {len(tokenizer):,}\n== tokenizer.vocab_size: {tokenizer.vocab_size:,}\n== tokenizer.special_tokens_map: {tokenizer.special_tokens_map}"
                )
                tokens = tokenizer.encode(text, add_special_tokens=False)
                decoded_tokens = [tokenizer.decode([t]) for t in tokens]
                zipped = [
                    f"({token_id}, `{token_str}`)"
                    for token_id, token_str in zip(tokens, decoded_tokens)
                ]
                formatted_text = repr(text)  # Compute repr() first
                print(
                    f"{formatted_text} \n-> (len:{len(tokens)}) {tokens} \n-> {zipped}"
                )

            messages = [
                {"role": "user", "content": "Who?"},
                {"role": "assistant", "content": "an AI"},
            ]
            applied_ct_smp = tokenizer.apply_chat_template(messages, tokenize=False)
            print(f"\n== applied_ct_smp:\n{applied_ct_smp}")
            tokenize_and_print(applied_ct_smp, tokenizer)

            # applied_ct_smp = tc.tokenizer.apply_chat_template(messages, tokenize=False)
            # print(f"\n== B2. applied_ct_smp:\n{applied_ct_smp}")
            # tokenize_and_print(applied_ct_smp, tc.tokenizer)

        # == 00-DQA: DATA PREPARATION:
        train_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=args.dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.dataset_target_columns,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_hash,
            hf_entity=args.hf_entity,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )

        # == 00-DQA: PRINT A FEW SAMPLES (only on main process) FOR TESTING:
        if accelerator.is_main_process:
            print(
                f"\n========== DEBUGGING: PRINT FIRST 3 SAMPLES AFTER CHAT TEMPLATE APPLIED by get_cached_dataset_tulu(): ==========\n"
            )
            num_samples = min(3, len(train_dataset))
            for i in range(num_samples):
                sample = train_dataset[i]
                print(f"***  Sample {i + 1} ***")
                # Decode tokens to text if INPUT_IDS_KEY exists, otherwise print raw sample
                if INPUT_IDS_KEY in sample:
                    # NOTE: skip_special_tokens=True to NOT display eos_token but THIS TOKEN is INSIDE the seq of tokenIDs used to train the model
                    decoded_text = tokenizer.decode(
                        sample[INPUT_IDS_KEY], skip_special_tokens=False
                    )
                    # print(f"== sample: {sample}")
                    print(f"== DECODED TEXT({len(decoded_text)}):\n{decoded_text}")
                    print(
                        f"== input_ids: ({len(sample[INPUT_IDS_KEY])}):\n{sample[INPUT_IDS_KEY]}"
                    )
                    print(
                        f"== attention_mask: ({len(sample['attention_mask'])}):\n{sample['attention_mask']}"
                    )
                    print(f"== labels: ({len(sample['labels'])}):\n{sample['labels']}")
                else:
                    print(f"== RAW SAMPLE: {sample}")
                print("-" * 50)

            print(
                f"========== END DEBUGGING: PRINT FIRST 3 SAMPLES AFTER CHAT TEMPLATE APPLIED by get_cached_dataset_tulu() ==========\n"
            )

        # debug_STOP()

        # == 00-DQA: DATA SAMPLE SHUFFLING:
        train_dataset = train_dataset.shuffle(seed=args.seed)
        train_dataset.set_format(type="pt")
    if accelerator.is_main_process:
        visualize_token(train_dataset[0][INPUT_IDS_KEY], tokenizer)

    if args.cache_dataset_only:
        return

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=tc.trust_remote_code,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
                if args.use_flash_attn
                else "eager",
            )
        elif args.use_liger_kernel:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM

            fused_linear_cross_entropy = args.reduce_loss == "mean"
            logger.info(
                f"Attempting to apply liger-kernel. {fused_linear_cross_entropy=}"
            )

            # Supported models: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py#L948
            model = AutoLigerKernelForCausalLM.from_pretrained(
                args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=tc.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                # liger-kernel specific args
                fused_linear_cross_entropy=fused_linear_cross_entropy,
            )
        else:
            ## ::FMS-ACCELERATION:: MoE Kernels
            model = model_loader(
                args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=tc.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
                if args.use_flash_attn
                else "eager",
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # get_callbacks_and_ready_for_train is called after augmentation
    # there are no callbacks for moe kernels
    # however we would still need to call the get_callbacks_and_ready_for_train
    # set (ignore params) within for moe plugin for EP
    if framework is not None:
        framework.get_callbacks_and_ready_for_train(model, accelerator)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    print("before", embeddings.weight.shape)
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
        print("after", embeddings.weight.shape)
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    print("after resize", embeddings.weight.shape)
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
        print("after resize and gather", embeddings.weight.shape)
    # capture model config just after resize
    # since this model config will be replaced with ds config after accelerate prep
    # we would need to save this in hf_converted to support eval
    model_config = model.config
    peft_config = None
    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing
            )

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "o_proj",
                "v_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # for k,v in model.named_parameters():
    #     print(k, v.shape, v)
    ## ::FMS-ACCELERATION:: MoE Kernels
    params_to_gather = []
    for p in model.parameters():
        params_to_gather.append(p)
    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=None):
        if framework is not None and framework.requires_augmentation:
            print("moe kernels augmentation step")
            # ideally train_args should be passed
            # not needed for moe kernels
            model, (peft_config,) = framework.augmentation(
                model, None, modifiable_args=(peft_config,)
            )

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"
        ),
        batch_size=args.per_device_train_batch_size,
    )
    # == DQA: DATA FORMAT PRIOR TO TRAINING HERE
    if accelerator.is_main_process:
        print(
            f"\n========== DEBUGGING: PRINT SAMPLES FROM train_dataloader AFTER DataCollatorForSeq2Seq: ==========\n"
        )
        for i, sample in enumerate(train_dataloader):
            print(f"***  Sample {i + 1} ***")
            print(
                f"== input_ids: ({len(sample['input_ids'][0])}):\n{sample['input_ids'][0]}"
            )
            print(
                f"== attention_mask: ({len(sample['attention_mask'][0])}):\n{sample['attention_mask'][0]}"
            )
            print(f"== labels: ({len(sample['labels'][0])}):\n{sample['labels'][0]}")
            if i == 2:
                break
        print(
            f"========== END DEBUGGING: PRINT SAMPLES FROM train_dataloader AFTER DataCollatorForSeq2Seq ==========\n"
        )

    # debug_STOP()

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            fused=args.fused_optimizer,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler
    # for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set.
    # In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set.
    # So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the
    # entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of
    # updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info(
        "\n =========================== RUNNING TRAINING ==========================="
    )
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(
        "\n ================================================================================"
    )

    # #=== 00-DQA: DEBUG JUST BEFORE TRAINING:
    # sys.exist(0)
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(args)

    if last_checkpoint_path:
        if accelerator.is_main_process:
            accelerator.print(
                f"Found and Resumed from checkpoint DQA: {last_checkpoint_path}"
            )

        # == DQA: LOAD LAST CHECKPOINT:
        # == As weights_only=True is DEFAULT in Pytorch 2.6, below trick is to disable it so that
        # == the checkpoint can be loaded
        original_torch_load = torch.load
        torch.load = functools.partial(original_torch_load, weights_only=False)
        try:
            accelerator.load_state(last_checkpoint_path)
        finally:
            torch.load = original_torch_load

        # accelerator.load_state(last_checkpoint_path)

        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
    if accelerator.is_main_process:
        print(f"Starting from epoch {starting_epoch} and step {completed_steps}.")

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    local_total_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    total_token_including_padding = torch.tensor(
        0, dtype=torch.int64, device=accelerator.device
    )
    start_time = time.time()

    if accelerator.is_main_process:
        logger.info(f"\n== DQA: START MODEL TRAINING:")

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        total_loss = 0
        total_aux_loss = 0
        if last_checkpoint_path and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            local_total_tokens += batch["attention_mask"].sum()
            total_token_including_padding += batch["attention_mask"].numel()

            with accelerator.accumulate(model):

                # == DQA Passes the batch to the model for forward propagation:
                if args.load_balancing_loss:
                    outputs = model(**batch, use_cache=False, output_router_logits=True)
                else:
                    # TODO: we have calculated the mean loss here anyway, so doubling the calculation
                    outputs = model(**batch, use_cache=False)
                if args.reduce_loss == "mean":
                    loss = outputs.loss
                else:
                    # == DQA custom loss:
                    # reduce loss is sum
                    # this ensures that we weight all tokens in the dataset equally,
                    # rather than weighting each overall example equally when
                    # using high amounts of gradient accumulation.
                    # this can result in > 5 point improvements in AlpacaEval
                    # see https://github.com/huggingface/transformers/issues/24725 for
                    # more discussion and details.
                    logits = outputs.logits
                    labels = batch["labels"]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                    shift_logits = shift_logits.view(-1, embedding_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    if args.load_balancing_loss:
                        aux_loss = args.load_balancing_weight * outputs.aux_loss
                        loss += aux_loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                if args.load_balancing_loss:
                    total_aux_loss += aux_loss.detach().float()
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item()
                        / args.gradient_accumulation_steps
                        / args.logging_steps
                    )
                    total_tokens = accelerator.gather(local_total_tokens).sum().item()
                    total_tokens_including_padding = (
                        accelerator.gather(total_token_including_padding).sum().item()
                    )
                    metrics_to_log = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "train_loss": avg_loss,
                        "total_tokens": total_tokens,
                        "per_device_tps": total_tokens
                        / accelerator.num_processes
                        / (time.time() - start_time),
                        "total_tokens_including_padding": total_tokens_including_padding,
                        "per_device_tps_including_padding": total_tokens_including_padding
                        / accelerator.num_processes
                        / (time.time() - start_time),
                    }
                    if args.load_balancing_loss:
                        avg_aux_loss = (
                            accelerator.gather(total_aux_loss).mean().item()
                            / args.gradient_accumulation_steps
                            / args.logging_steps
                        )
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, Aux Loss: {avg_aux_loss}, TPS: {total_tokens / (time.time() - start_time)}"
                        )
                        metrics_to_log["aux_loss"] = avg_aux_loss
                    else:
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, "
                            f"Loss: {avg_loss:.5f}, total_tokens: {total_tokens:,}, total_tokens_including_padding: {total_tokens_including_padding:,}, "
                            f"TPS: {total_tokens / (time.time() - start_time):.2f}"
                        )

                        # logger.info(
                        #     f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, TPS: {total_tokens / (time.time() - start_time)}"
                        # )
                    if args.with_tracking:
                        accelerator.log(
                            metrics_to_log,
                            step=completed_steps,
                        )
                    total_loss = 0
                    total_aux_loss = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)

                        # == DQA: SAVE A NEW CHECKPOINT:
                        accelerator.save_state(output_dir)
                        ## ::FMS-ACCELERATION:: MoE Kernels
                        if accelerator.is_local_main_process:
                            if fast_moe_config is not None:
                                converted = os.path.join(output_dir, "hf_converted")
                                os.mkdir(converted)
                                # save train states like optimizers rng etc
                                save_state(
                                    accelerator=accelerator, output_dir=converted
                                )
                                logger.info(f"accelerator state saved at {converted}")
                                logger.info(
                                    "converting moe kernels state dict to original model state dict"
                                )
                                from fms_acceleration_moe.utils.checkpoint_utils import \
                                    recover_safetensors_from_dcp

                                recover_safetensors_from_dcp(
                                    output_dir,
                                    args.model_name_or_path,
                                    converted,
                                    is_deepspeed=True,
                                )
                                logger.info(f"converted model is saved at {converted}")
                                tokenizer.save_pretrained(converted)
                                logger.info(f"tokenizer saved at {converted}")
                                model_config.save_pretrained(converted)
                                logger.info(f"config.json saved at {converted}")

                        # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
                        with open(
                            os.path.join(
                                get_last_checkpoint_path(args, incomplete=True),
                                "COMPLETED",
                            ),
                            "w",
                        ) as f:
                            f.write(
                                "COMPLETED"
                            )  # annoyingly, empty files arent uploaded by beaker.
                        if (
                            accelerator.is_local_main_process
                        ):  # TODO: in mason local model this is gonna error out if using something like output/test; because mason used the same shared file ssytem.
                            clean_last_n_checkpoints(
                                args.output_dir, args.keep_last_n_checkpoints
                            )
                        accelerator.wait_for_everyone()

                if completed_steps >= args.max_train_steps:
                    break

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            if accelerator.is_local_main_process:
                if fast_moe_config is not None:
                    converted = os.path.join(output_dir, "hf_converted")
                    os.mkdir(converted)
                    # save train states like optimizers rng etc
                    save_state(accelerator=accelerator, output_dir=converted)
                    logger.info(f"accelerator state saved at {converted}")
                    logger.info(
                        "converting moe kernels state dict to original model state dict"
                    )
                    from fms_acceleration_moe.utils.checkpoint_utils import \
                        recover_safetensors_from_dcp

                    recover_safetensors_from_dcp(
                        output_dir,
                        args.model_name_or_path,
                        converted,
                        is_deepspeed=True,
                    )
                    logger.info(f"converted model is saved at {converted}")
                    tokenizer.save_pretrained(converted)
                    logger.info(f"tokenizer saved at {converted}")
                    model_config.save_pretrained(converted)
                    logger.info(f"config.json saved at {converted}")
            # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
            with open(
                os.path.join(
                    get_last_checkpoint_path(args, incomplete=True), "COMPLETED"
                ),
                "w",
            ) as f:
                f.write(
                    "COMPLETED"
                )  # annoyingly, empty files arent uploaded by beaker.
            if accelerator.is_local_main_process:
                clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info(f"\n== DQA: END MODEL TRAINING")

    if args.output_dir is not None:
        # save with accelerate
        # wont work with moe kernels
        # since we need to recover original state dict
        # from modified moe kernel state dict
        if fast_moe_config is None:
            save_with_accelerate(
                accelerator,
                model,
                tokenizer,
                args.output_dir,
                args.use_lora,
            )
        else:
            output_dir = os.path.join(args.output_dir, "final")
            accelerator.save_state(output_dir)
            if accelerator.is_local_main_process:
                converted = os.path.join(output_dir, "hf_converted")
                os.mkdir(converted)
                # save train states like optimizers rng etc
                save_state(accelerator=accelerator, output_dir=converted)
                logger.info(f"accelerator state saved at {converted}")
                logger.info(
                    "converting moe kernels state dict to original model state dict"
                )
                from fms_acceleration_moe.utils.checkpoint_utils import \
                    recover_safetensors_from_dcp

                recover_safetensors_from_dcp(
                    output_dir, args.model_name_or_path, converted, is_deepspeed=True
                )
                logger.info(f"converted model is saved at {converted}")
                tokenizer.save_pretrained(converted)
                logger.info(f"tokenizer saved at {converted}")
                model_config.save_pretrained(converted)
                logger.info(f"config.json saved at {converted}")

        # else:
        #     save_with_accelerate_for_moe_kernels(
        #         accelerator=accelerator,
        #         output_dir=args.output_dir,
        #         tokenizer=tokenizer,
        #         model_name_or_path=args.model_name_or_path,
        #     )

    # remove all checkpoints to save space
    if accelerator.is_local_main_process:
        clean_last_n_checkpoints(
            args.output_dir, keep_last_n_checkpoints=args.keep_last_n_checkpoints
        )

    if (
        args.try_auto_save_to_beaker
        and accelerator.is_main_process
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)

    if (
        is_beaker_job()
        and accelerator.is_main_process
        and args.try_launch_beaker_eval_jobs
    ):
        launch_ai2_evals_on_weka(
            path=args.output_dir,
            leaderboard_name=args.hf_repo_revision,
            oe_eval_max_length=args.oe_eval_max_length,
            wandb_url=wandb_tracker.run.get_url(),
            oe_eval_tasks=args.oe_eval_tasks,
            gs_bucket_path=args.gs_bucket_path,
        )
    if args.push_to_hub:
        push_folder_to_hub(
            accelerator,
            args.output_dir,
            args.hf_repo_id,
            args.hf_repo_revision,
        )
    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()

    main(args, tc)

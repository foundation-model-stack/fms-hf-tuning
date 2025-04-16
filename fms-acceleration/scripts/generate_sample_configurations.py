# Copyright The IBM Tuning Team
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

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Standard
from typing import Any, Dict, List
import argparse
import logging
import os
import re
from copy import deepcopy

# Third Party
from ruamel.yaml import YAML

logger = logging.getLogger()

# use https://yaml.readthedocs.io/en/latest/overview/
# because it can preserve style, comments, etc
yaml = YAML()

# configuration template
TEMPLATE = """\
# FMS Acceleration Plugin Configuration. 
#
# Each stanza incorporates various configurations for 
# different fine-tuning / training tasks.
plugins:
"""

TEMPLATE = yaml.load(TEMPLATE)


# if we can find a proper utility to do this then remove this function
# this will search the .ca elements, example below
# and increment the column
# Comment(
#   start=[None, [CommentToken('# PEFT-related acceleration\n', line: 0, col: 0)], []],
#   items={
#     peft: [None, None, None, [CommentToken('\n', line: 2, col: 2), CommentToken('# quantization-releated acceleration\n', line: 3, col: 4), CommentToken('# e.g., kernels for quantized base weights\n', line: 4, col: 4)]]
#   })
def indent_yaml(data: Dict, n: int = 2):
    "helper utility to indent the comments in the yaml."

    # in order to track that the indentation is done once
    indented = set()

    # values here be like
    # values = [None, [CommentToken('# PEFT-related acceleration\n', line: 0, col: 0)], []],
    # values = [None, None, CommentToken('\n\n      # If true, then will already expect quantized checkpoint \n      # passed into TrainingArguments.model_name_or_path\n', line: 16, col: 6), None]

    def _update_column(values: List[List]):
        # extract out the None's
        for xs in values:
            if xs is None:
                continue
            elif not isinstance(xs, list):
                xs = [xs]

            for x in xs:
                _id = id(x)
                if _id not in indented:

                    # this is a wierd edge case, if a value starts with
                    # \n's followed by spaces and #,
                    # then the start_mark has no effect
                    # so we manually replace spaces
                    x.value = re.sub(
                        r"(\n\s+)#",  # any "\n    #" pattern
                        r"\1" + (" " * n) + "#",  # replace with n extra spaces
                        x.value,
                    )
                    # even if we had replaced spaces, we still shift the
                    # start mark for consistency
                    x.start_mark.column += n
                indented.add(_id)

    def _indent(node):
        # if the node has a comment attribute
        if hasattr(node, "ca"):
            ca = node.ca  # get the comment attribute
            if ca.comment is not None:
                _update_column([x for x in ca.comment if x is not None])
            for values in ca.items.values():
                _update_column([x for x in values if x is not None])

        # if the node is a dict and has other children nodes
        if hasattr(node, "values"):
            for v in node.values():
                _indent(v)

    _indent(data)


def update_configuration_contents(
    configuration_contents: Dict,
    augment_at_path: str,
    augmented_contents: Any,
):
    "helper function to replace configuration contents at augment_at_path with augmented_contents"

    contents = configuration_contents
    augment_at_path = augment_at_path.split(".")
    for k in augment_at_path[:-1]:
        contents = contents[k]
    key = augment_at_path[-1]
    if isinstance(contents[key], dict):
        d = contents[key]
        del contents[key]
        contents[augmented_contents] = d
    else:
        contents[key] = augmented_contents
    return configuration_contents


def read_configuration(path: str) -> Dict:
    "helper function to read yaml config into json"

    with open(path) as f:
        return yaml.load(f)


# map of keys -> configuration specs
# the spec can either be
# 1. a (str) path to a plugin config
# 2. a (str, list) tuple of (path to plugin, list of augmentations)
#
# NOTE: an augmentation (path, value) will augment a config at the
# specified key path, with the value.
KEY_AUTO_GPTQ = "auto-gptq"
KEY_BNB_NF4 = "bnb-nf4"
KEY_BNB_NF4_BASELINE = "baseline-bnb-nf4"
KEY_AUTO_GPTQ_FOAK = "auto-gptq-foak"
KEY_AUTO_GPTQ_FOAK_LIGER = "auto-gptq-foak-liger"
KEY_BNB_NF4_FOAK = "bnb-nf4-foak"
KEY_BNB_NF4_FOAK_LIGER = "bnb-nf4-foak-liger"
KEY_AADP_PADDING_FREE = "aadp-padding-free"
KEY_AADP_MULTIPACK = "aadp-multipack"
KEY_FAST_KERNELS = "foak-fast-kernels"
KEY_FAST_KERNELS_LIGER = "foak-fast-kernels-liger"
KEY_SCATTERMOE_EP1 = "moe-scattermoe-ep1"
KEY_SCATTERMOE_EP2 = 'moe-scattermoe-ep2'
KEY_SCATTERMOE_EP4 = 'moe-scattermoe-ep4'
KEY_SCATTERMOE_EP8 = 'moe-scattermoe-ep8'

CONFIGURATIONS = {
    KEY_AUTO_GPTQ: "plugins/accelerated-peft/configs/autogptq.yaml",
    KEY_BNB_NF4: (
        "plugins/accelerated-peft/configs/bnb.yaml",
        [("peft.quantization.bitsandbytes.quant_type", "nf4")],
    ),
    KEY_BNB_NF4_BASELINE: (
        "plugins/accelerated-peft/configs/bnb.yaml",
        [
            ("peft.quantization.bitsandbytes.quant_type", "nf4"),
            ("peft.quantization.bitsandbytes.no_peft_model", True),
        ],
    ),
    KEY_AUTO_GPTQ_FOAK: (
        "plugins/fused-ops-and-kernels/configs/fast_quantized_peft.yaml",
        [("peft.quantization.fused_ops_and_kernels.base_layer", "auto_gptq")],
    ),
    KEY_AUTO_GPTQ_FOAK_LIGER: (
        "plugins/fused-ops-and-kernels/configs/fast_quantized_peft_liger.yaml",
        [("peft.quantization.fused_ops_and_kernels.base_layer", "auto_gptq")],
    ),
    KEY_BNB_NF4_FOAK: (
        "plugins/fused-ops-and-kernels/configs/fast_quantized_peft.yaml",
        [("peft.quantization.fused_ops_and_kernels.base_layer", "bitsandbytes")],
    ),
    KEY_BNB_NF4_FOAK_LIGER: (
        "plugins/fused-ops-and-kernels/configs/fast_quantized_peft_liger.yaml",
        [("peft.quantization.fused_ops_and_kernels.base_layer", "bitsandbytes")],
    ),
    KEY_AADP_PADDING_FREE: "plugins/attention-and-distributed-packing/configs/padding_free.yaml",
    KEY_AADP_MULTIPACK: "plugins/attention-and-distributed-packing/configs/multipack.yaml",
    KEY_FAST_KERNELS: "plugins/fused-ops-and-kernels/configs/fast_kernels.yaml",
    KEY_FAST_KERNELS_LIGER: "plugins/fused-ops-and-kernels/configs/fast_kernels_liger.yaml",
    KEY_SCATTERMOE_EP1: "plugins/accelerated-moe/configs/scattermoe.yaml",
    KEY_SCATTERMOE_EP2: (
        "plugins/accelerated-moe/configs/scattermoe.yaml",
        [("training.moe.scattermoe.ep_degree", 2)],
    ),
    KEY_SCATTERMOE_EP4: (
        "plugins/accelerated-moe/configs/scattermoe.yaml",
        [("training.moe.scattermoe.ep_degree", 4)],
    ),
    KEY_SCATTERMOE_EP8: (
        "plugins/accelerated-moe/configs/scattermoe.yaml",
        [("training.moe.scattermoe.ep_degree", 8)],
    ),
}

# list of (tag, combi) tuples
# - the tag will be used to name the sample config
# - the combi is a tuple of CONFIGURATIONS keys (see above). Used to
#   determine the CONFIGURATIONS that will be merged to form the sample
#   config.
COMBINATIONS = [
    ("accelerated-peft-autogptq", (KEY_AUTO_GPTQ,)),
    ("accelerated-peft-bnb-nf4", (KEY_BNB_NF4,)),
    ("baseline-peft-bnb-nf4", (KEY_BNB_NF4_BASELINE,)),
    ("accelerated-peft-autogptq-foak", (KEY_AUTO_GPTQ, KEY_AUTO_GPTQ_FOAK)),
    ("accelerated-peft-bnb-nf4-foak", (KEY_BNB_NF4, KEY_BNB_NF4_FOAK)),
    ("accelerated-peft-autogptq-foak-liger", (KEY_AUTO_GPTQ, KEY_AUTO_GPTQ_FOAK_LIGER)),
    ("accelerated-peft-bnb-nf4-foak-liger", (KEY_BNB_NF4, KEY_BNB_NF4_FOAK_LIGER)),
    ("aadp-padding-free", (KEY_AADP_PADDING_FREE,)),
    ("accelerated-peft-autogptq-padding-free", (KEY_AADP_PADDING_FREE,KEY_AUTO_GPTQ)),
    ("accelerated-peft-bnb-nf4-padding-free", (KEY_AADP_PADDING_FREE,KEY_BNB_NF4)),
    ("accelerated-peft-autogptq-foak-padding-free", (KEY_AADP_PADDING_FREE,KEY_AUTO_GPTQ, KEY_AUTO_GPTQ_FOAK)),
    ("accelerated-peft-bnb-nf4-foak-padding-free", (KEY_AADP_PADDING_FREE,KEY_BNB_NF4, KEY_BNB_NF4_FOAK)),
    ("aadp-padding-free-multipack", (KEY_AADP_PADDING_FREE, KEY_AADP_MULTIPACK)),
    ("foak-fast-kernels", (KEY_FAST_KERNELS,)),
    ("foak-fast-kernels-liger", (KEY_FAST_KERNELS_LIGER,)),
    ("moe-scattermoe-granite-ep1", (KEY_SCATTERMOE_EP1,)),
    ("moe-scattermoe-granite-ep1-padding-free", (KEY_AADP_PADDING_FREE, KEY_SCATTERMOE_EP1,)),
    ("moe-scattermoe-granite-ep1-padding-free-foak", (KEY_AADP_PADDING_FREE, KEY_FAST_KERNELS, KEY_SCATTERMOE_EP1,)),
    ("moe-scattermoe-granite-ep2", (KEY_SCATTERMOE_EP2,)),
    ("moe-scattermoe-granite-ep2-padding-free", (KEY_AADP_PADDING_FREE, KEY_SCATTERMOE_EP2,)),
    ("moe-scattermoe-granite-ep2-padding-free-foak", (KEY_AADP_PADDING_FREE, KEY_FAST_KERNELS, KEY_SCATTERMOE_EP2,)),
    ("moe-scattermoe-granite-ep4", (KEY_SCATTERMOE_EP4,)),
    ("moe-scattermoe-granite-ep4-padding-free", (KEY_AADP_PADDING_FREE, KEY_SCATTERMOE_EP4,)),
    ("moe-scattermoe-granite-ep4-padding-free-foak", (KEY_AADP_PADDING_FREE, KEY_FAST_KERNELS, KEY_SCATTERMOE_EP4,)),
    ("moe-scattermoe-granite-ep8", (KEY_SCATTERMOE_EP8,)),
    ("moe-scattermoe-granite-ep8-foak", (KEY_FAST_KERNELS, KEY_SCATTERMOE_EP8,)),
]

# TODO: throw error if merge conflicts
def merge_configs(config_contents: List[Dict]):
    "helper function to merge configuration contents."

    # merge in place
    def _merge(result: Dict, new_contents: Dict):
        for k, v in new_contents.items():
            if k not in result:
                # if k is not in result, it means v does not
                # exist as a subtree under result, so we just do
                # an assingment
                result[k] = v
            else:
                # otherwise we call the merge
                _merge(result[k], v)

    if len(config_contents) == 0:
        return {}

    result = config_contents[0]
    if len(config_contents) == 1:
        return result

    for new_contents in config_contents[1:]:
        _merge(result, new_contents)

    return result


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description="utility for generating sample configurations"
    )
    parser.add_argument(
        "output_dir",
        help="directory where the sample configurations will be placed. "
        "If directory does not exist, will be created.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        logger.warn(
            f"Sample configuration directory '{args.output_dir}' does not exist. "
            "Creating the directory."
        )
        os.makedirs(args.output_dir)

    # read the configuration contents.
    # - read directly and modify the CONFIGURATIONS object (since we do not require it anymore)
    # - after reading we just replace into the CONFIGURATIONS
    for key in CONFIGURATIONS:

        logger.info(f"Reading and updating contents of configuration '{key}'.")

        if isinstance(CONFIGURATIONS[key], tuple):
            path, replacements = CONFIGURATIONS[key]
        elif isinstance(CONFIGURATIONS[key], str):
            path = CONFIGURATIONS[key]
            replacements = []
        else:
            raise NotImplementedError(
                "CONFIGURATIONS can only contain str paths or (path (str), augmentations (list))."
            )

        # read contents and update CONFIGURATIONS object for use later
        CONFIGURATIONS[key] = read_configuration(path)

        # update contents in CONFIGURATIONS
        for update_path, update_val in replacements:
            CONFIGURATIONS[key] = update_configuration_contents(
                CONFIGURATIONS[key],
                update_path,
                update_val,
            )

    # now merge contents in CONFIGURATIONS to form the final
    # sample configuration
    for combi_tag, combi in COMBINATIONS:
        # merging the configuration contents for this particular combination
        # if keys are not the same separate the merges
        config = merge_configs([deepcopy(CONFIGURATIONS[tag]) for tag in combi])
        indent_yaml(config)  # add the indent

        # writing the configuration contents
        filename = os.path.join(
            args.output_dir, f"{combi_tag}-sample-configuration.yaml"
        )
        with open(filename, "w") as f:
            TEMPLATE["plugins"] = config
            yaml.dump(TEMPLATE, f)

        logger.info(f"Wrote sample-configuration '{filename}'.")

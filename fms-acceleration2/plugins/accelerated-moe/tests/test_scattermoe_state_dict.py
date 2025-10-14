# Copyright The FMS HF Tuning Authors
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

# Standard
from typing import List

# Third Party
# pylint: disable=import-error
import pytest

# First Party
from fms_acceleration_moe.utils.scattermoe_constants import (
    PARAM_NAME_ROUTER_SCATTERMOE,
    PARAM_NAME_WEIGHT_SCATTERMOE,
)
from fms_acceleration_moe.utils.scattermoe_state_dict import (
    get_checkpoint_meta_from_sharded_safetensor,
)

# just a dummy sample value
ST_SHARD = "model-00001-of-00001.safetensors"


# --------------------------- HELPERS ------------------------------
# - builds a weight dict for checkpoints where MoE is sharded (i.e.,
#   one linear by expert).
# - this is like Mixtral style
def build_dummy_weight_map_sharded_moe(
    prefix: str,
    module_name: str,
    router_name: str,
    expert_name: str,
    num_layers: int,
    num_experts: int,
    expert_keys: List[str],
):

    # - ST_SHARD entries are not impt for the test
    weight_map = {}
    for i in range(num_layers):
        layer_map = {
            f"{prefix}.{i}.{module_name}.{router_name}.weight": ST_SHARD,
        }
        for j in range(num_experts):
            expert_map = {}

            for n in expert_keys:
                expert_map.update(
                    {
                        f"{prefix}.{i}.{module_name}.{expert_name}.{j}.{n}.weight": ST_SHARD
                    }
                )

            layer_map.update(expert_map)

        weight_map.update(layer_map)

    return weight_map


# - this is like granite style
def build_dummy_weight_map_non_sharded_moe(
    prefix: str,
    module_name: str,
    router_name: str,
    num_layers: int,
    expert_keys: List[str],
):
    # - ST_SHARD entries are not impt for the test
    weight_map = {}
    for i in range(num_layers):
        layer_map = {
            f"{prefix}.{i}.{module_name}.{router_name}.weight": ST_SHARD,
        }
        for n in expert_keys:
            layer_map.update({f"{prefix}.{i}.{module_name}.{n}.weight": ST_SHARD})

        weight_map.update(layer_map)

    return weight_map


# --------------------------- TEST ---------------------------------

PARAMETERS = [
    (
        True,
        "model.layers",
        "block_sparse_moe",
        "gate",
        "experts",
        2,
        8,
        ["w1", "w2", "w3"],
    ),
    (
        False,
        "model.layers",
        "block_sparse_moe",
        "gate",
        "input_linear|output_linear|input_linear",
        2,
        None,
        ["input_linear", "output_linear"],
    ),
]


@pytest.mark.parametrize(
    (
        "sharded_ckpt,prefix,module_name,router_name,expert_name,"
        "num_layers,num_experts,expert_keys"
    ),
    PARAMETERS,
)
def test_get_metadata_from_sharded_safetensor_correctly(
    sharded_ckpt: bool,
    prefix: str,
    module_name: str,
    router_name: str,
    expert_name: str,
    num_layers: int,
    num_experts: int,
    expert_keys: List[str],
):

    if sharded_ckpt:
        weight_map = build_dummy_weight_map_sharded_moe(
            prefix,
            module_name,
            router_name,
            expert_name,
            num_layers,
            num_experts,
            expert_keys,
        )
    else:
        weight_map = build_dummy_weight_map_non_sharded_moe(
            prefix, module_name, router_name, num_layers, expert_keys
        )

    # get the metadata for the a layer
    ckpt_metadata = get_checkpoint_meta_from_sharded_safetensor(
        weight_map,
        prefix + ".0",  # include layer
        module_name,
        router_name,
        expert_name,
    )

    _key = f"{PARAM_NAME_ROUTER_SCATTERMOE}.weight"
    assert _key in ckpt_metadata, "unable to map scattermoe router metadata."

    _n = len(ckpt_metadata[_key])
    assert _n == 1, f"expected only 1 router weights but got {_n}"

    for n in PARAM_NAME_WEIGHT_SCATTERMOE:
        _key = f"{n}.weight"
        assert _key in ckpt_metadata, f"unable top map scattermoe expert weight {n}."

        _n = len(ckpt_metadata[_key])
        if sharded_ckpt:
            assert (
                _n == num_experts
            ), f"missing expert weights, only mapped {_n} weights out of {num_experts}."
        else:
            assert (
                _n == 1
            ), f"missing expert weights, mapped {_n} but expected only 1 for non-sharded."


def test_get_metadata_from_sharded_safetensor_incorrectly():

    weight_map_wrong = {"prefix.moe_name.expert.weight": ST_SHARD}

    # - if passing a prefix, has to map the weight_map
    with pytest.raises(ValueError, match="Could not get safetensor map for"):
        get_checkpoint_meta_from_sharded_safetensor(
            weight_map_wrong, "wrong_prefix", "moe_name", None, "expert_name"
        )

    # - if passing mutiple expert names, cannot violate the number of
    # possible expert gates
    with pytest.raises(
        AssertionError, match="If expert_name has |, expect between 2 and"
    ):
        get_checkpoint_meta_from_sharded_safetensor(
            weight_map_wrong, "prefix", "moe_name", None, "exp1|exp2|exp3|exp4"
        )

    # - if a weight_map key that matches the moe_name, cannot be handled
    with pytest.raises(
        ValueError, match="Unable to handle key 'prefix.moe_name.expert.weight'"
    ):
        get_checkpoint_meta_from_sharded_safetensor(
            weight_map_wrong, "prefix", "moe_name", None, "wrong_expert_name"
        )

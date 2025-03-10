# Standard
from collections import defaultdict

# Third Party
from accelerate.utils import set_module_tensor_to_device
from transformers import PreTrainedModel
import torch

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


def ensure_weights_retied(
    param_init_fn, model: torch.nn.Module, device: torch.cuda.device
):

    _tied_names = model._tied_weights_keys
    if not _tied_names:
        # if no tied names just passthrough
        return param_init_fn

    # get map of parameter instances to params.
    # - needed for replacement later
    _tied_params = {}
    for name in _tied_names:
        name = name.split(".")
        name, param_name = ".".join(name[:-1]), name[-1]
        mod = model.get_submodule(name)
        param = getattr(mod, param_name)

        _tied_params[id(param)] = None  # placeholder for the param first

    # build param_init_fn for the case with tied params
    def param_init_fn_tied_param(module: torch.nn.Module):

        # track which params to tie
        # - usually only 1, but for completeness consider > 1
        params_to_tie = defaultdict(list)
        for n, param in module.named_parameters(recurse=False):
            if id(param) in _tied_params:
                params_to_tie[id(param)].append(n)

        # call the param init fn, which potentially re-allocates the
        # parameters
        module = param_init_fn(module)

        # search the parameters again and tie them up again
        for id_key, _param_names in params_to_tie.items():
            for param_name in _param_names:
                param = _tied_params[id_key]
                if param is None:
                    # everything will be tied to the first time the
                    # param is observed
                    _tied_params[id_key] = getattr(module, param_name)
                else:
                    setattr(module, param_name, param)  # tie

        return module

    return param_init_fn_tied_param


# utility to put tensors on the cpu
def put_selected_meta_tensors_on_cpu(model: PreTrainedModel):

    done = {}
    # - fow now we only put input and output embeddings
    for module in [
        model.get_input_embeddings(),
        model.get_output_embeddings(),
    ]:

        for param_name, param in module.named_parameters(recurse=False):
            param_id = id(param)

            if param.device == torch.device("meta"):
                if param_id not in done:
                    value = torch.empty(*param.size(), dtype=param.dtype)
                    done[param_id] = value  # memoize
                else:
                    # this is a tied weight, get back the previous value
                    value = done[param_id]

                set_module_tensor_to_device(module, param_name, "cpu", value)

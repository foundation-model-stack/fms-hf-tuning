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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import importlib
import sys

# Third Party
from accelerate import Accelerator
from peft import LoraConfig
from transformers import TrainingArguments
import torch


@dataclass
class PluginRegistration:
    plugin: "AccelerationPlugin"
    AND: List[str] = None
    OR: List[str] = None

    # package metadata
    package_name: str = None
    package_version: str = None


PLUGIN_REGISTRATIONS: List[PluginRegistration] = []


def _trace_key_path(configuration: Dict, key: str):
    t = configuration

    try:
        for k in key.split("."):
            t = t[k]
    except KeyError:
        return None  # None will mean not found
    return t


def get_relevant_configuration_sections(configuration: Dict) -> Dict:
    results = []

    # this function updates cfg with content
    # - equivalent to taking a union
    def _update_config_contents(_cfg: Dict, content: Dict, key: str):
        path = key.split(".")
        n = len(path)
        _cfg = relevant_config
        while n > 1:
            p = path.pop(0)
            if p not in _cfg:
                _cfg[p] = {}
            _cfg = _cfg[p]
            n -= 1

        _cfg[path[0]] = content

    # assume the registrations are all done with at least some default key
    for registration in PLUGIN_REGISTRATIONS:
        relevant_config = {}

        _and_keys = registration.AND
        _or_keys = registration.OR
        if _and_keys is None:
            _and_keys = []
        if _or_keys is None:
            _or_keys = []

        # go through AND paths then OR paths
        # - if all AND paths are speciied, then return their union of all content
        # - if any OR path is specified, then return the union of specified content
        reject = False
        for key in _and_keys:
            content = _trace_key_path(configuration, key)
            if content is None:
                # if AND key, then if at least one of them not
                # specified, then reject and do not descend config tree
                reject = True
                break

            # update
            _update_config_contents(relevant_config, content, key)

        # if all the any keys were not satisfied, then reset the config
        if reject:
            relevant_config = {}

        for key in _or_keys:
            content = _trace_key_path(configuration, key)
            if content is not None:
                if reject:
                    # it is an OR key, and if at least one of them specified
                    # then do not reject
                    reject = False

                # update all content that is not None
                _update_config_contents(relevant_config, content, key)

        if reject:
            continue

        if len(relevant_config) > 0:
            results.append((relevant_config, registration.plugin))

    return results


class AccelerationPlugin:
    # will be triggered if the configuration_paths are found in the
    # acceleration framework configuration file (under KEY_PLUGINS)
    @staticmethod
    def register_plugin(
        plugin: "AccelerationPlugin",
        configuration_and_paths: List[str] = None,
        configuration_or_paths: List[str] = None,
        **kwargs,
    ):

        # pylint: disable=trailing-whitespace
        # removed because of src/fms_acceleration/framework_plugin.py:96:8:
        # W0602: Using global for 'PLUGIN_REGISTRATIONS' but no assignment
        # is done (global-variable-not-assigned)
        # global PLUGIN_REGISTRATIONS

        assert (
            configuration_and_paths is not None and len(configuration_and_paths) > 0
        ) or (
            configuration_or_paths is not None and len(configuration_or_paths) > 0
        ), "Specify at least one AND or OR path"

        # get the package metadata
        pkg_name = sys.modules[plugin.__module__].__package__
        try:
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            package_version = None

        PLUGIN_REGISTRATIONS.append(
            PluginRegistration(
                plugin=plugin,
                AND=configuration_and_paths,
                OR=configuration_or_paths,
                package_name=pkg_name,
                package_version=package_version,
            )
        )

    restricted_model_archs: Optional[Set] = None
    require_packages: Optional[Set] = None

    def __init__(self, configurations: Dict[str, Dict]):
        # will pass in a list of dictionaries keyed by "configuration_keys"
        # to be used for initialization
        self.configurations = configurations

    @property
    def requires_custom_loading(self):
        return False

    @property
    def requires_augmentation(self):
        return False

    def model_loader(self, model_name: str, **kwargs):
        raise NotImplementedError

    def augmentation(
        self,
        model: torch.nn.Module,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        raise NotImplementedError

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator: Accelerator = None
    ):
        return []

    def _check_config_and_maybe_check_values(
        self, key: str, values: List[Any] = None, default: Any = None
    ):
        t = _trace_key_path(self.configurations, key)

        if values is not None:  # if there is something to check against
            if isinstance(t, dict):
                # if the tree is a dict
                if len(t.keys()) > 1:
                    raise AccelerationPluginConfigError(
                        f"{self.__class__.__name__}: '{key}' found but amongst multiple "
                        "'{t.keys()}' exist. Ambiguous check in expected set '{values}'."
                    )
                t = list(t.keys())[0]  # otherwise take the first value

            if t not in values:
                if t is not None or default is None:
                    raise AccelerationPluginConfigError(
                        f"{self.__class__.__name__}: Value at '{key}' was '{t}'. "
                        f"Not found in expected set '{values}'."
                    )
                # otherwise if there is a default, then take it
                t = default
        else:
            # if nothing to check against, and no default, we still
            # need to ensure its a valid configuration key
            if t is None:
                if default is None:
                    raise AccelerationPluginConfigError(
                        f"{self.__class__.__name__}: '{key}' was not a valid configuration config"
                    )
                t = default

        return t

    def _check_config_equal(self, key: str, value: Any, **kwargs):
        return self._check_config_and_maybe_check_values(key, [value], **kwargs)


class AccelerationPluginConfigError(Exception):
    pass

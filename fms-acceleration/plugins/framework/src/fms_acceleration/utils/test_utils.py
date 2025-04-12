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
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union

# Third Party
import torch
import yaml

# First Party
from fms_acceleration.framework import KEY_PLUGINS, AccelerationFramework
from fms_acceleration.framework_plugin import PLUGIN_REGISTRATIONS, AccelerationPlugin


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
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_framework_from_json(
    configuration_contents: Dict, require_packages_check: bool = True
):
    "helper function to configure framework given json configuration"
    with NamedTemporaryFile("w") as f:
        yaml.dump({KEY_PLUGINS: configuration_contents}, f)
        return AccelerationFramework(f.name, require_packages_check)


@contextmanager
def build_framework_and_maybe_instantiate(
    plugins_to_be_registered: List[
        Union[
            Tuple[List[str], Type[AccelerationPlugin]],  # and_paths, plugin_class
            Tuple[
                List[str],
                List[str],  # and_or_paths
                Type[AccelerationPlugin],  # plugin_class
            ],
        ]
    ],
    configuration_contents: Dict = None,
    instantiate: bool = True,
    reset_registrations: bool = True,
    require_packages_check: bool = True,
):
    "helper function to register plugins and instantiate an acceleration framework for testing"

    if configuration_contents is None:
        configuration_contents = {}

    # empty out
    if reset_registrations:
        old_registrations = []
        old_registrations.extend(PLUGIN_REGISTRATIONS)
        PLUGIN_REGISTRATIONS.clear()
    old_active_plugins = AccelerationFramework.active_plugins
    old_custom_loading_plugins = AccelerationFramework.plugins_require_custom_loading
    AccelerationFramework.active_plugins = []
    AccelerationFramework.plugins_require_custom_loading = []

    for paths_and_plugins in plugins_to_be_registered:
        try:
            and_paths, plugin = paths_and_plugins
            or_paths = None
        except ValueError:
            and_paths, or_paths, plugin = paths_and_plugins

        AccelerationPlugin.register_plugin(
            plugin,
            configuration_and_paths=and_paths,
            configuration_or_paths=or_paths,
        )

    if instantiate:
        yield configure_framework_from_json(
            configuration_contents, require_packages_check
        )
    else:
        yield

    # put back
    if reset_registrations:
        PLUGIN_REGISTRATIONS.clear()
        PLUGIN_REGISTRATIONS.extend(old_registrations)
    AccelerationFramework.active_plugins = old_active_plugins
    AccelerationFramework.plugins_require_custom_loading = old_custom_loading_plugins


# alias because default instantiate=True
build_framework_and_instantiate = build_framework_and_maybe_instantiate


def instantiate_framework(
    configuration_contents: Dict,
    require_packages_check: bool = True,
):
    """helper function to instantiate an acceleration framework for testing.
    This version does not refresh plugin registration.
    """
    return build_framework_and_instantiate(
        [],
        configuration_contents,
        reset_registrations=False,
        require_packages_check=require_packages_check,
    )


def create_noop_model_with_archs(
    class_name: str = "ModelNoop", archs: List[str] = None
):
    "helper function to create a dummy model with mocked architectures"
    if archs is None:
        archs = []

    config = type("Config", (object,), {"architectures": archs})
    return type(class_name, (torch.nn.Module,), {"config": config})


def create_plugin_cls(
    class_name: str = "PluginNoop",
    restricted_models: Set = None,
    require_pkgs: Set = None,
    requires_custom_loading: bool = False,
    requires_augmentation: bool = False,
    augmentation: Callable = None,
    model_loader: Callable = None,
):
    "helper function to create plugin class"

    if restricted_models is None:
        restricted_models = set()
    if require_pkgs is None:
        require_pkgs = set()

    attributes = {
        "restricted_model_archs": restricted_models,
        "require_packages": require_pkgs,
        "requires_custom_loading": requires_custom_loading,
        "requires_augmentation": requires_augmentation,
    }

    if augmentation is not None:
        attributes["augmentation"] = augmentation

    if model_loader is not None:
        attributes["model_loader"] = model_loader

    return type(class_name, (AccelerationPlugin,), attributes)


def dummy_augmentation(self, model, train_args, modifiable_args):
    "dummy augmentation implementation"
    return model, modifiable_args


def dummy_custom_loader(self, model_name, **kwargs):
    "dummy custom loader returning dummy model"
    return create_noop_model_with_archs(archs=["DummyModel"])  #


@contextmanager
def instantiate_model_patcher():
    # First Party
    from fms_acceleration.model_patcher import (  # pylint: disable=import-outside-toplevel
        ModelPatcher,
    )

    old_registrations = ModelPatcher.rules
    old_history = ModelPatcher.history
    ModelPatcher.rules = {}
    ModelPatcher.history = []
    yield
    ModelPatcher.rules = old_registrations
    ModelPatcher.history = old_history


@contextmanager
def instantiate_accel_patcher():
    # First Party
    from fms_acceleration.accelerator_patcher import (  # pylint: disable=import-outside-toplevel
        AcceleratorPatcher,
    )

    old_registrations = AcceleratorPatcher.replacement_rules
    old_history = AcceleratorPatcher.history
    AcceleratorPatcher.replacement_rules = {}
    AcceleratorPatcher.history = []
    yield
    AcceleratorPatcher.replacement_rules = old_registrations
    AcceleratorPatcher.history = old_history

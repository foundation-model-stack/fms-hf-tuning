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
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Annotated, Dict, List, Type
import warnings

# Third Party
import yaml

# Local
from .fused_ops_and_kernels import FastKernelsConfig, FusedLoraConfig
from .quantized_lora_config import AutoGPTQLoraConfig, BNBQLoraConfig
from tuning.utils.import_utils import is_fms_accelerate_available

if is_fms_accelerate_available():
    # Third Party
    from fms_acceleration import AccelerationFramework  # pylint: disable=import-error
    from fms_acceleration.framework import KEY_PLUGINS  # pylint: disable=import-error

# these are optional annotations that describe different behavior
@dataclass
class ConfigAnnotation:

    # AccelerationFramework configuration path
    path: str

    # if omitted, will take the field name
    key: str = None

    # only one that has single=True may exist under its path
    # - this is used to indicate conflicting configurations
    # - we do not allow two configurations that load the model to be
    #   activated at the same time
    standalone: bool = False

    # set to true to throw a user warning
    experimental: bool = False

    # set to indicate what acceeleration packages are needed
    required_packages: List[str] = None

    def __post_init__(self):
        if self.required_packages is None:
            self.required_packages = []


@dataclass
class AccelerationFrameworkConfig:
    "Dataclass that manages configuration of AccelerationFramework"

    PACKAGE_PREFIX = "fms_acceleration_"

    # each field will a single-level use case dataclass
    auto_gptq: Annotated[
        AutoGPTQLoraConfig,
        ConfigAnnotation(
            path="peft.quantization", standalone=True, required_packages=["peft"]
        ),
    ] = None

    bitsandbytes: Annotated[
        BNBQLoraConfig,
        ConfigAnnotation(
            path="peft.quantization", standalone=True, required_packages=["peft"]
        ),
    ] = None

    fused_lora: Annotated[
        FusedLoraConfig,
        ConfigAnnotation(
            path="peft.quantization",
            key="fused_ops_and_kernels",
            experimental=True,
            required_packages=["foak"],
        ),
    ] = None

    fast_kernels: Annotated[
        FastKernelsConfig,
        ConfigAnnotation(
            path="peft.quantization",
            key="fused_ops_and_kernels",
            experimental=True,
            required_packages=["foak"],
        ),
    ] = None

    @staticmethod
    def from_dataclasses(*dataclasses: Type):
        "Convert one or many FMS config dataclasses to a monolithic AccelerationConfig"

        # Assumption: AccelerationFrameworkConfig only has fields that are
        #             single level dataclasses
        # Assumption: dataclasses is a list of nested dataclasses
        # - each dc in dataclasses is a nested dataclass.
        # - each dc.field in dc is a non-nested dataclass.

        if len(dataclasses) == 0:
            raise ValueError(
                "AccelerationFrameworkConfig construction requires at least one dataclass."
            )

        # first unroll all the dataclases into a single level
        nested_dataclasses = []
        for dc in dataclasses:
            if dc is None:
                continue

            # make sure that it every field is a dataclass
            for fi in fields(dc):
                attr = getattr(dc, fi.name)
                if attr is None:
                    continue  # skip the None attributes

                if not is_dataclass(attr):
                    raise ValueError(
                        f"field '{fi.name}' is specified but not a dataclass"
                    )

                # NOTE: should we also check that these are non-nested
                # dataclasses?
                nested_dataclasses.append(attr)

        config = AccelerationFrameworkConfig()
        rem_fields = {fi.name: fi for fi in fields(config)}  # these need to be parsed

        # process the dataclasses that were nested
        # by assumption these are non-nested dataclasses
        for dc in nested_dataclasses:

            # check the fields that are yet to be populated
            found_field = False
            for fi in rem_fields.values():

                # check if it is an AccelerationFrameworkConfig field
                if isinstance(dc, fi.type.__origin__):
                    found_field = True
                    break

            if not found_field:
                raise ValueError(
                    f"dataclass '{dc}' cannot be placed into AccelerationFrameworkConfig."
                )

            # assign the dataclass
            setattr(config, fi.name, dc)
            del rem_fields[fi.name]  # remove the field

        return config

    def get_framework(self):

        if is_fms_accelerate_available():

            # to be eventually be made to be passed as a dict to Acceleration
            # Framework
            # Standard
            from tempfile import (  # pylint: disable=import-outside-toplevel
                NamedTemporaryFile,
            )

            try:
                with NamedTemporaryFile("w") as f:
                    self.to_yaml(f.name)
                    return AccelerationFramework(f.name)
            except ValueError as e:
                (msg,) = e.args

                # AcceleratorFramework raises ValueError if it
                # fails to configure any plugin
                if self.is_empty() and msg.startswith("No plugins could be configured"):
                    # in the case when the error was thrown when
                    # the acceleration framework config was empty
                    # then this is expected.
                    return None

                raise e
        else:
            if not self.is_empty():
                raise ValueError(
                    "No acceleration framework package found. To use, first "
                    "ensure that 'pip install fms-hf-tuning[fms-accel]' is done first to "
                    "obtain the acceleration framework dependency. Additional "
                    "acceleration plugins make be required depending on the requsted "
                    "acceleration. See README.md for instructions."
                )

    def is_empty(self):
        "check if the configuration is empty"
        for fi in fields(self):
            if getattr(self, fi.name) is not None:
                return False
        return True

    def to_dict(self):
        """convert a valid AccelerationFrameworkConfig dataclass into a schema-less dictionary
        as dictated by the header annotations.
        """

        # populate a dictionary
        configuration_contents = {}

        # helper function to populate
        def _descend_and_set(path: List[str], d: Dict):
            r = configuration_contents
            for p in path[:-1]:
                if p not in r:
                    r[p] = {}  # new branch
                r = r[p]

            p = path[-1]
            r[p] = {**r.get(p, {}), **d}  # merge dict if exists

        # parse each field
        already_set = set()
        for fi in fields(self):
            datacls = getattr(self, fi.name)
            if datacls is not None:
                # this is the documented way to get annotations
                # https://docs.python.org/3/library/typing.html#typing.Annotated
                annotate: ConfigAnnotation
                (annotate,) = fi.type.__metadata__
                prefix_path = tuple(annotate.path.split("."))
                if annotate.standalone and prefix_path in already_set:
                    raise ValueError(
                        f"Configuration path '{'.'.join(prefix_path)}' "
                        "already has one standalone config."
                    )

                if annotate.experimental:
                    warnings.warn(
                        "An experimental acceleration feature is requested by specifying the "
                        f"'--{fi.name}' argument. Please note this feature may not support certain "
                        "edge cases at this juncture. When the feature matures this "
                        "message will be turned off."
                    )

                if not all(
                    is_fms_accelerate_available(x) for x in annotate.required_packages
                ):
                    raise ValueError(
                        "An acceleration feature is requested by specifying the "
                        f"'--{fi.name}' argument, but the this requires acceleration packages "
                        "to be installed. Please do:\n"
                        + "\n".join(
                            [
                                "- python -m fms_acceleration.cli install "
                                f"{AccelerationFrameworkConfig.PACKAGE_PREFIX + x}"
                                for x in annotate.required_packages
                            ]
                        )
                    )

                key = annotate.key if annotate.key is not None else fi.name
                path = prefix_path + (key,)
                already_set.add(prefix_path)
                _descend_and_set(path, asdict(datacls))

        return configuration_contents

    def to_yaml(self, filename: str):
        "convert a valid AccelerationConfig dataclass into a yaml"
        configuration_contents = self.to_dict()
        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump({KEY_PLUGINS: configuration_contents}, f)

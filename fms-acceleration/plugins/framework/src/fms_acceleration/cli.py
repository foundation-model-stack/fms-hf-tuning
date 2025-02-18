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
from typing import List, Union
import os
import subprocess
import sys

# Third Party
from pip._internal.cli.main import main as pipmain
from transformers.utils.import_utils import _is_package_available
import yaml

# Local
from .constants import PLUGIN_PREFIX, PLUGINS

GITHUB_URL = "github.com/foundation-model-stack/fms-acceleration.git"

REPO_CACHE_DIR = ".fms/repository"


# TODO: make a version that fetches the
def install_plugin(
    *args: List[str],
):
    "function to install plugin. Inputs should contain a pkg_name."

    pkg_name = [x for x in args if not x.startswith("-")]
    assert len(pkg_name) == 1, "Please specify exactly one plugin to install"
    pkg_name = pkg_name[0]

    # if toxinidir is specified in path, replace with cwd
    pkg_name = pkg_name.format(toxinidir=os.getcwd())

    # take the flags
    args = [x for x in args if x.startswith("-")]

    if os.path.exists(pkg_name):
        pipmain(["install", *args, pkg_name])
        return

    # otherwise should be an internet install
    response = pipmain(
        [
            "install",
            *args,
            pkg_name,
        ]
    )

    # Reference from https://github.com/pypa/pip/blob/main/src/pip/_internal/cli/status_codes.py
    if response > 0:
        print("PyPi installation failed. Falling back to installation from Github.")

        if pkg_name.startswith(PLUGIN_PREFIX):
            pkg_name = pkg_name.replace(PLUGIN_PREFIX, "")

        pipmain(
            [
                "install",
                *args,
                f"git+https://{GITHUB_URL}#subdirectory=plugins/accelerated-{pkg_name}",
            ]
        )


def list_plugins():
    print(
        "\nChoose from the list of plugin shortnames, and do:\n"
        " * 'python -m fms_acceleration.cli install <pip-install-flags> PLUGIN_NAME'.\n\n"
        "Alternatively if the repository was checked out, pip install it from REPO_PATH:\n"
        " * 'pip install <pip-install-flags> REPO_PATH/plugins/PLUGIN_NAME'.\n\n"
        "List of PLUGIN_NAME [PLUGIN_SHORTNAME]:\n"
    )
    for i, name in enumerate(PLUGINS):
        full_name = f"{PLUGIN_PREFIX}{name}"
        installed = _is_package_available(full_name)

        postfix = ""
        if installed:
            postfix += "(installed)"

        print(f"{i+1}. {full_name} [{name}] {postfix}")


def get_benchmark_artifacts(dest_dir: str):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if not os.path.exists(os.path.join(dest_dir, ".git")):
        command = f"""cd {dest_dir} && git init && \
            git remote add -f origin https://{GITHUB_URL} && \
            git config --global init.defaultBranch main && \
            git config core.sparsecheckout true && \
            echo scripts/benchmarks >> .git/info/sparse-checkout && \
            echo sample-configurations >> .git/info/sparse-checkout && \
        """
    else:
        command = f"cd {dest_dir} && git fetch origin && "
    command += "git pull origin main "

    out = subprocess.run(command, shell=True, capture_output=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(
            f"could not get benchmark artifacts with error code {out.returncode}"
        )
    return out


def list_sample_configs(
    configs_dir: str,
    contents_file: str = "sample-configurations/CONTENTS.yaml",
    get_artifacts: bool = True,
):
    if get_artifacts:
        get_benchmark_artifacts(REPO_CACHE_DIR)
    with open(os.path.join(configs_dir, contents_file), encoding="utf-8") as f:
        for i, entry in enumerate(yaml.safe_load(f)["framework_configs"]):
            shortname = entry["shortname"]
            plugins = entry["plugins"]
            filename = entry["filename"]
            print(f"{i+1}. {shortname} ({filename}) - plugins: {plugins}")


def list_arguments(
    scenario_dir: str,
    config_shortnames: Union[str, List[str]],
    scenario_file: str = "scripts/benchmarks/scenarios.yaml",
    ignored_fields: List = None,
    get_artifacts: bool = True,
):
    if ignored_fields is None:
        ignored_fields = ["model_name_or_path"]

    if get_artifacts:
        get_benchmark_artifacts(REPO_CACHE_DIR)

    if isinstance(config_shortnames, str):
        config_shortnames = [config_shortnames]

    with open(os.path.join(scenario_dir, scenario_file), encoding="utf-8") as f:
        scenarios = yaml.safe_load(f)["scenarios"]
        found = 0
        print(f"Searching for configuration shortnames: {config_shortnames}")
        for scn in scenarios:
            if "framework_config" not in scn:
                continue

            hit_sn = [x for x in config_shortnames if x in scn["framework_config"]]
            if len(hit_sn) > 0:
                found += 1
                name = scn["name"]
                arguments = scn["arguments"]
                hit_sn = ", ".join(hit_sn)
                print(f"{found}. scenario: {name}\n   configs: {hit_sn}\n   arguments:")
                lines = []
                for key, val in arguments.items():
                    if key not in ignored_fields:
                        lines.append(f"      --{key} {val}")

                print(" \\\n".join(lines))
                print("\n")

        if not found:
            print(
                f"ERROR: Could not list arguments for configuration shortname '{config_shortnames}'"
            )


def cli():
    # not using argparse since its so simple
    message = (
        "FMS Acceleration Framework Command Line Tool.\n"
        "Command line tool to help manage the Acceleration Framework packages.\n"
    )
    argv = sys.argv
    if len(argv) == 1:
        print(message)
        return
    if len(argv) > 1:
        command = argv[1]
        if len(argv) > 2:
            variadic = sys.argv[2:]
        else:
            variadic = []

    if command == "install":
        assert len(variadic) >= 1, "Please provide the acceleration plugin name"
        install_plugin(*variadic)
    elif command == "plugins":
        assert len(variadic) == 0, "list does not require arguments"
        list_plugins()
    elif command == "configs":
        assert len(variadic) == 0, "list-config does not require arguments"
        list_sample_configs(REPO_CACHE_DIR)
    elif command == "arguments":
        assert len(variadic) >= 1, "Please provide the config shortname"
        list_arguments(REPO_CACHE_DIR, *variadic)
    else:
        raise NotImplementedError(f"Unknown fms_acceleration.cli command '{command}'")


if __name__ == "__main__":
    cli()

#!/usr/bin/env python3
# Standard
from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess

# Third Party
from tuning_config_recommender.adapters import (  # pylint: disable=import-error
    FMSAdapter,
)
import yaml

ACCEL_NESTED_PREFIXES = {
    "fsdp_": "fsdp_config",
}

DATA_KEYS = {
    "training_data_path",
    "validation_data_path",
    "dataset",
}


def grab_flags(tokens, start, end):
    cfg, i = {}, start
    while i < end:
        t = tokens[i]
        if t.startswith("--"):
            k, v = t[2:], True
            if "=" in t:
                k, v = k.split("=", 1)
                v = v.strip('"')
            elif i + 1 < end and not tokens[i + 1].startswith("--"):
                v = tokens[i + 1].strip('"')
                i += 1
            cfg[k] = v
        i += 1
    return cfg


def load_yaml(path):
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f:
                y = yaml.safe_load(f)
            return y if isinstance(y, dict) else {}
        except (OSError, yaml.YAMLError):
            return {}
    return {}


def nest_accelerate_flags(flat_dist):
    nested = {section: {} for section in ACCEL_NESTED_PREFIXES.values()}
    remaining = {}

    for k, v in flat_dist.items():
        matched = False
        for prefix, section in ACCEL_NESTED_PREFIXES.items():
            if k.startswith(prefix):
                nested[section][k] = v
                matched = True
                break
        if not matched:
            remaining[k] = v

    for sec in list(nested.keys()):
        if not nested[sec]:
            nested.pop(sec)

    return {**remaining, **nested}


def parse(cmd: str):
    tokens = shlex.split(cmd)
    has_m = "-m" in tokens
    is_accel = "accelerate" in tokens and "launch" in tokens
    if is_accel and has_m:
        m = tokens.index("-m")
        dist_flat = grab_flags(tokens, 0, m)
        train = grab_flags(tokens, m + 2, len(tokens))

    elif has_m:
        m = tokens.index("-m")
        dist_flat = {}
        train = grab_flags(tokens, m + 2, len(tokens))
    else:
        dist_flat = {}
        train = grab_flags(tokens, 0, len(tokens))

    yaml_path = train.pop("data_config", None)
    if yaml_path:
        data = load_yaml(yaml_path)
    else:
        data = {}
    accel_yaml_path = dist_flat.pop("config_file", None)
    accel_yaml = load_yaml(accel_yaml_path) if accel_yaml_path else {}
    dist_nested = nest_accelerate_flags(dist_flat)
    dist = {**accel_yaml, **dist_nested}
    train.pop("config_file", None)

    return train, dist, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print parsed configs and exit (no adapter, no execution).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Run adapter and show launch command but DO NOT execute it.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        print("Error: No command provided.")
        return

    cmd = " ".join(args.command)
    train_cfg, dist_cfg, data_cfg = parse(cmd)
    train_cfg.pop("config_file", None)
    dist_cfg.pop("config_file", None)

    if args.debug:
        print("\n[dist_config]\n", json.dumps(dist_cfg, indent=2))
        print("\n[train_config]\n", json.dumps(train_cfg, indent=2))
        print("\n[data_config]\n", json.dumps(data_cfg, indent=2))
        return

    adapter = FMSAdapter(base_dir=Path("fms_recommender_ouput/final"))
    result = adapter.execute(
        train_config=train_cfg,
        dist_config=dist_cfg,
        compute_config={},
        data_config=data_cfg,
        unique_tag="fms-recommender",
        paths={},
    )
    launch_cmd = result["launch_command"]

    if args.preview:
        print("\n[LAUNCH COMMAND — PREVIEW ONLY]\n")
        print(launch_cmd)
        return

    print("\n[EXECUTING launch command]\n")
    print(launch_cmd)
    subprocess.run(launch_cmd, shell=True, check=True)


if __name__ == "__main__":
    main()

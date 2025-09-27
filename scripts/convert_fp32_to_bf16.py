#!/usr/bin/env python3
# Convert FP32 -> BF16 for .pt/.pth, single .safetensors, or a dir of .safetensors.
# Skips optimizer states to preserve FP32.

# Standard
from pathlib import Path
from typing import Any
import argparse
import shutil

# Third Party
import torch

try:
    # Third Party
    from safetensors.torch import safe_open, save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

OPTIM_ROOT_KEYS = {"optimizer", "optim", "opt_state"}


def cast_fp32_to_bf16(x: Any, *, in_optim: bool = False) -> Any:
    if isinstance(x, torch.Tensor):
        return x if in_optim or x.dtype != torch.float32 else x.to(torch.bfloat16)
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            k_lower = k.lower() if isinstance(k, str) else ""
            child_in_optim = in_optim or (k_lower in OPTIM_ROOT_KEYS)
            out[k] = cast_fp32_to_bf16(v, in_optim=child_in_optim)
        return out
    if isinstance(x, (list, tuple)):
        return type(x)(cast_fp32_to_bf16(v, in_optim=in_optim) for v in x)
    return x


def convert_pt_pth(inp: Path, out: Path) -> None:
    data = torch.load(inp, map_location="cpu")
    data = cast_fp32_to_bf16(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    print(f"[pt/pth] wrote: {out}")


def is_optim_tensor_name(name: str) -> bool:
    parts = (name or "").lower().replace("/", ".").split(".")
    return bool(parts) and parts[0] in OPTIM_ROOT_KEYS


def convert_safetensors_file(inp: Path, out: Path) -> None:
    if not HAS_SAFETENSORS:
        raise RuntimeError("safetensors not installed. pip install safetensors")
    tensors = {}
    with safe_open(str(inp), framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if t.dtype == torch.float32 and not is_optim_tensor_name(key):
                t = t.to(torch.bfloat16)
            tensors[key] = t
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata={"converted_to": "bfloat16"})
    print(f"[safetensors] wrote: {out}")


def convert_dir_of_safetensors(src: Path, dst: Path) -> None:
    """Convert all .safetensors in a directory; copy other files as-is."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.suffix == ".safetensors":
            convert_safetensors_file(item, dst / item.name)
        else:
            target = dst / item.name
            if item.is_file():
                shutil.copy2(item, target)
            elif item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
    print(f"[dir] wrote: {dst}")


def main():
    ap = argparse.ArgumentParser(
        description="Convert FP32 tensors to BF16 (skip optimizer states)."
    )
    ap.add_argument(
        "input",
        type=Path,
        help="Input: .pt/.pth, .safetensors, or HF directory with .safetensors",
    )
    ap.add_argument("output", type=Path, help="Output file or directory")
    args = ap.parse_args()

    p = args.input
    if p.is_file():
        sfx = p.suffix.lower()
        if sfx in {".pt", ".pth"}:
            if args.output.is_dir():
                raise SystemExit(
                    "For .pt/.pth input, output must be a file path (not a directory)."
                )
            convert_pt_pth(p, args.output)
        elif sfx == ".safetensors":
            out = (
                args.output
                if args.output.suffix == ".safetensors"
                else (args.output / p.name)
            )
            convert_safetensors_file(p, out)
        else:
            raise SystemExit(f"Unsupported file type: {p}")
    elif p.is_dir():
        if any(x.suffix == ".safetensors" for x in p.iterdir()):
            convert_dir_of_safetensors(p, args.output)
        else:
            raise SystemExit("Directory has no .safetensors files.")
    else:
        raise SystemExit(f"Not found: {p}")

    print("Done.")


if __name__ == "__main__":
    main()

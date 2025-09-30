#!/usr/bin/env python3
# Convert FP32 -> BF16 for .pt/.pth, single .safetensors, or a dir of .safetensors.
# Skips optimizer states to preserve FP32.

# Standard
from pathlib import Path
from typing import Any, Iterable, Set
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
            child_in_optim = in_optim or any(
                k_lower.startswith(root) for root in OPTIM_ROOT_KEYS
            )
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
    first = (name or "").lower().replace("/", ".").split(".")[0]
    return any(first.startswith(root) for root in OPTIM_ROOT_KEYS)


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


def slim_copy_dir_skip_only(src: Path, dst: Path, skip_names: Iterable[str]) -> None:
    """
    Copy everything from src -> dst EXCEPT files whose names are in skip_names.
    Directories are copied entirely unless their name is in skip_names.
    """
    dst.mkdir(parents=True, exist_ok=True)
    skip_set: Set[str] = set(skip_names)

    for item in src.iterdir():
        if item.name in skip_set:
            continue
        target = dst / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
    print(
        f"[slim] wrote: {dst} (skipped: {', '.join(skip_set) if skip_set else 'none'})"
    )


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
        description="Convert FP32 tensors to BF16 (skips optimizer states)."
    )
    ap.add_argument(
        "input",
        type=Path,
        help="Input: .pt/.pth, .safetensors, or HF directory with .safetensors",
    )
    ap.add_argument("output", type=Path, help="Output file or directory")

    ap.add_argument(
        "--slim",
        action="store_true",
        help="For directory inputs: after conversion, copy everything \
        except files listed in --skip (default: optimizer.pt).",
    )
    ap.add_argument(
        "--slim-only",
        action="store_true",
        help="For directory inputs: DO NOT convert; just copy everything \
        except files in --skip.",
    )
    ap.add_argument(
        "--skip",
        default="optimizer.pt",
        help="Comma-separated file names to skip during slimming \
        (applies to --slim or --slim-only). Default: optimizer.pt",
    )

    args = ap.parse_args()

    if args.slim and args.slim_only:
        raise SystemExit("Choose at most one of: --slim or --slim-only.")

    skip_list = (
        [s.strip() for s in args.skip.split(",")]
        if (args.slim or args.slim_only)
        else []
    )

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
        if not any(x.suffix == ".safetensors" for x in p.iterdir()):
            raise SystemExit("Directory has no .safetensors files.")
        if args.slim_only:
            slim_copy_dir_skip_only(p, args.output, skip_list)
        else:
            convert_dir_of_safetensors(p, args.output)
            if args.slim:
                tmp = args.output.parent / (args.output.name + "_tmp_slim")
                if tmp.exists():
                    shutil.rmtree(tmp)
                slim_copy_dir_skip_only(args.output, tmp, skip_list)
                shutil.rmtree(args.output)
                tmp.rename(args.output)
    else:
        raise SystemExit(f"Not found: {p}")

    print("Done.")


if __name__ == "__main__":
    main()

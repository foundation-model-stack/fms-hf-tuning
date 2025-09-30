#!/usr/bin/env python3
# Checkpoint utilities (unified --inplace):
# - Default: copy INPUT -> OUTPUT unchanged
# - --convert-model-to-bf16: convert model FP32 -> BF16 (optimizer tensors remain FP32)
# - --no-optimizer: when writing outputs, drop optimizer files/dirs (defaults + --drop-files)
# - --drop-files: comma-separated extra file/dir (used with --no-optimizer, and with --inplace)
# - --inplace: perform conversion and/or dropping directly in INPUT (destructive)


# Standard
from pathlib import Path
from typing import Any, Iterable, Set
import argparse
import os
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

DEFAULT_OPTIM_DROPS = {"optimizer.pt", "optimizer", "optimizer_0", "optimizer_1"}


def _atomic_replace(tmp: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(tmp), str(dst))  # atomic on POSIX


def cast_fp32_to_bf16(x: Any, *, in_optim: bool = False) -> Any:
    """Recursively cast float32 tensors to bfloat16, skipping optimizer subtrees."""
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


def is_optim_tensor_name(name: str) -> bool:
    first = (name or "").lower().replace("/", ".").split(".")[0]
    return any(first.startswith(root) for root in OPTIM_ROOT_KEYS)


def convert_pt_pth(inp: Path, out: Path) -> None:
    data = torch.load(inp, map_location="cpu")
    data = cast_fp32_to_bf16(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    print(f"[pt/pth] wrote: {out}")


def convert_pt_pth_inplace(inp: Path) -> None:
    tmp = inp.with_suffix(inp.suffix + ".tmp")
    convert_pt_pth(inp, tmp)
    _atomic_replace(tmp, inp)
    print(f"[pt/pth][inplace] updated: {inp}")


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


def convert_safetensors_file_inplace(inp: Path) -> None:
    tmp = inp.with_suffix(inp.suffix + ".tmp")
    convert_safetensors_file(inp, tmp)
    _atomic_replace(tmp, inp)
    print(f"[safetensors][inplace] updated: {inp}")


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


def convert_dir_of_safetensors_inplace(src: Path) -> None:
    """Convert all .safetensors files in-place within `src`."""
    count = 0
    for item in src.iterdir():
        if item.suffix == ".safetensors":
            convert_safetensors_file_inplace(item)
            count += 1
    if count == 0:
        raise SystemExit("Directory has no .safetensors files.")
    print(f"[dir][inplace] converted {count} shard(s) in: {src}")


def _name_matches(name: str, patterns: Set[str]) -> bool:
    """Exact-name match (simple and predictable)."""
    return name in patterns


def copy_dir_drop(src: Path, dst: Path, drop_names: Iterable[str]) -> None:
    """Copy directory but drop certain files/dirs by exact name."""
    dst.mkdir(parents=True, exist_ok=True)
    drop_set: Set[str] = set(drop_names)
    for item in src.iterdir():
        if _name_matches(item.name, drop_set):
            continue
        target = dst / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
    print(
        f"[copy-drop] wrote: {dst} (dropped: {', '.join(sorted(drop_set)) if drop_set else 'none'})"
    )


def prune_dir_inplace(src: Path, drop_names: Iterable[str]) -> None:
    """Delete top-level files/dirs in `src` whose names match `drop_names`. Destructive."""
    drop_set: Set[str] = set(drop_names)
    removed = []
    for item in src.iterdir():
        if _name_matches(item.name, drop_set):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
            removed.append(item.name)
    print(
        f"[inplace-drop] removed: {', '.join(sorted(removed)) if removed else 'nothing'}"
    )


def copy_any(src: Path, dst: Path) -> None:
    """Pure copy (no dtype changes, no dropping)."""
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst if dst.suffix else dst / src.name)
    elif src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dst / item.name
            if item.is_file():
                shutil.copy2(item, target)
            elif item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
    else:
        raise SystemExit(f"Not found: {src}")
    print(f"[copy] wrote: {dst}")


def main():
    ap = argparse.ArgumentParser(
        description="Checkpoint utilities: copy by default; \
        optionally convert FP32->BF16 and/or drop optimizer files. "
        "Use --inplace to modify INPUT directly."
    )
    ap.add_argument("input", type=Path, help="Input file or directory")
    ap.add_argument("output", type=Path, help="Output file or directory")

    ap.add_argument(
        "--convert-model-to-bf16",
        action="store_true",
        help="Convert FP32 -> BF16 for model tensors; optimizer tensors remain FP32.",
    )
    ap.add_argument(
        "--no-optimizer",
        action="store_true",
        help="When writing outputs, drop optimizer files/dirs (defaults + --drop-files).",
    )
    ap.add_argument(
        "--drop-files",
        default="",
        help="Comma-separated extra file/dir names to drop \
        (works with --no-optimizer and/or --inplace).",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Perform operations directly on INPUT (destructive). For files: overwrite in place; "
        "for directories: convert shards in-place and/or delete dropped names.",
    )

    args = ap.parse_args()

    p = args.input

    user_drops = {s.strip() for s in args.drop_files.split(",") if s.strip()}
    if args.no_optimizer:
        drop_set = DEFAULT_OPTIM_DROPS | user_drops
    else:
        drop_set = user_drops

    if args.inplace:
        if not p.exists():
            raise SystemExit(f"Not found: {p}")

        if args.convert_model_to_bf16:
            if p.is_file():
                sfx = p.suffix.lower()
                if sfx in {".pt", ".pth"}:
                    convert_pt_pth_inplace(p)
                elif sfx == ".safetensors":
                    convert_safetensors_file_inplace(p)
                else:
                    raise SystemExit(
                        f"Unsupported file type for inplace conversion: {p}"
                    )
            elif p.is_dir():
                convert_dir_of_safetensors_inplace(p)
            else:
                raise SystemExit(f"Not found: {p}")

        if drop_set:
            if not p.is_dir():
                print(
                    "[inplace] --drop-files applies to directories; skipping for file input."
                )
            else:
                prune_dir_inplace(p, drop_set)

        print("Done.")
        return

    if not args.convert_model_to_bf16 and not args.no_optimizer and not drop_set:
        copy_any(p, args.output)
        print("Done.")
        return

    if p.is_file():
        sfx = p.suffix.lower()
        if args.convert_model_to_bf16:
            if sfx in {".pt", ".pth"}:
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
        else:
            copy_any(p, args.output)
        print("Done.")
        return

    if p.is_dir():
        if args.convert_model_to_bf16:
            if not any(x.suffix == ".safetensors" for x in p.iterdir()):
                raise SystemExit("Directory has no .safetensors files.")
            convert_dir_of_safetensors(p, args.output)
            if args.no_optimizer or drop_set:
                tmp = args.output.parent / (args.output.name + "_tmp_drop")
                if tmp.exists():
                    shutil.rmtree(tmp)
                copy_dir_drop(
                    args.output,
                    tmp,
                    DEFAULT_OPTIM_DROPS | drop_set if args.no_optimizer else drop_set,
                )
                shutil.rmtree(args.output)
                tmp.rename(args.output)
        else:
            if args.no_optimizer or drop_set:
                copy_dir_drop(
                    p,
                    args.output,
                    DEFAULT_OPTIM_DROPS | drop_set if args.no_optimizer else drop_set,
                )
            else:
                copy_any(p, args.output)
        print("Done.")
        return

    raise SystemExit(f"Not found: {p}")


if __name__ == "__main__":
    main()

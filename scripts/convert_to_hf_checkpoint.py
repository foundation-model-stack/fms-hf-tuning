#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from fms_acceleration_moe.utils import recover_safetensors_from_dcp


HF_CACHE = "/workspace/.hf"
os.environ.setdefault("HF_HOME", HF_CACHE)


def has_weights(p: Path) -> bool:
    return (
        (p / "model.safetensors").exists()
        or (p / "model.safetensors.index.json").exists()
        or any(p.glob("model-*.safetensors"))
    )


def get_base_model(model_id_or_path: str, allow_download: bool) -> Path:
    p = Path(model_id_or_path)

    if p.exists():
        if not has_weights(p):
            raise RuntimeError(f"No base weights found in {p}")
        return p.resolve()

    if not allow_download:
        raise RuntimeError("Base model not found locally and downloads disabled")

    local_dir = snapshot_download(
        repo_id=model_id_or_path,
        allow_patterns=[
            "config.json",
            "model*.safetensors",
            "tokenizer*",
            "special_tokens_map.json",
            "generation_config.json",
        ],
    )

    local_dir = Path(local_dir).resolve()
    if not has_weights(local_dir):
        raise RuntimeError(f"Downloaded base model but weights missing in {local_dir}")

    return local_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp_checkpoint_dir", required=True, type=Path)
    ap.add_argument("--pretrained_model_name_or_path", required=True)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--allow_model_download", action="store_true")
    ap.add_argument(
        "--additional_special_tokens",
        nargs="*",
        default=[],
    )
    ap.add_argument("--chat_template", type=str, default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # base model (local snapshot)
    base_model_dir = get_base_model(
        args.pretrained_model_name_or_path,
        args.allow_model_download,
    )

    # dcp to hf compatible
    recover_safetensors_from_dcp(
        str(args.dcp_checkpoint_dir),
        str(base_model_dir),
        str(args.output_dir),
    )

    # tokenizer chat_template plus additional tokens
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    if args.chat_template is not None:
        tokenizer.chat_template = args.chat_template
    if args.additional_special_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": args.additional_special_tokens}
        )
    tokenizer.save_pretrained(args.output_dir)

    config = AutoConfig.from_pretrained(base_model_dir)
    config.vocab_size = len(tokenizer)
    config.save_pretrained(args.output_dir)

    print(f"[OK] HF checkpoint written to {args.output_dir}")
    print(f"[OK] vocab_size = {len(tokenizer)}")


if __name__ == "__main__":
    main()


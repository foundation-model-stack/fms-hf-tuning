import json
import os
import sys
import datasets

PROMPT_DICT = {
    "prompt_input": (
        "Input:\n{input}\n\n### Response:" # HACK
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def format_alpaca_fn(example):
    prompt_input, prompt_no_input = PROMPT_DICT['prompt_input'], PROMPT_DICT['prompt_no_input']
    output = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    output = f"{output} {example['output']}"
    return {"output": output}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python3 alpaca_to_sft_format.py [file_1, file_2, ...]")

    for file_path in sys.argv[1:]:
        if not os.path.isfile(file_path):
            raise ValueError(f"alpaca dataset f{file_path} does not exist!")
        base_path, file_name = os.path.split(file_path)
        export_path = os.path.join(base_path, f"sft_format_{file_name}")
        print(f"Converting alpaca format file: {file_path} to SFT trainer training format")
        ds = datasets.load_dataset("json", data_files=file_path)
        alpaca_ds = ds['train'].map(format_alpaca_fn, remove_columns=['instruction', 'input'])
        alpaca_ds.to_json(export_path)
        print(f"Exported SFT format data to {export_path}")

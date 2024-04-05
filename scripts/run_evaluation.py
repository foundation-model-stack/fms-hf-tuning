"""Runs evaluation on Alpaca formatted data.

Metrics used: Accuracy / Micro & Macro F1.
"""
import os
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score
import argparse
import json
from tqdm import tqdm
import datasets
import evaluate
from run_inference import TunedCausalLM
from shutil import rmtree

def parse_and_validate_args():
    """Parse the arguments and ensure everything is valid."""
    parser = argparse.ArgumentParser(
        description="Runs evaluation on a Alpaca style dataset"
    )
    parser.add_argument(
        "--model", help="Path to tuned model / merged model to be loaded", required=True
    )
    parser.add_argument(
        "--data_path", help="Path to the dataset to be loaded", required=True
    )
    parser.add_argument(
        "--split", help="Split to be used for the data", default="train"
    )
    parser.add_argument(
        "--max_new_tokens", help="Max new tokens to use in generation", type=int,
    )
    parser.add_argument(
        "--output_dir", help="Directory path to export results to", default="eval_results"
    )
    parser.add_argument(
        "--delimiter",
        help="Delimiter to be used for multilabel multiclass evaluation",
        default=None,
    )
    parser.add_argument('--purge_results', action=argparse.BooleanOptionalAction)

    parsed_args = parser.parse_args()

    print(f"Multiclass / multioutput delimiter: {parsed_args.delimiter}")
    # If we have a collision on the outdir, only remove the existing file if we explicitly say to
    if os.path.exists(parsed_args.output_dir):
        if parsed_args.purge_results:
            print(f"Existing output file/directory: [{parsed_args.output_dir}] will be deleted...")
            rmtree(parsed_args.output_dir)
        else:
            raise FileExistsError(
                f"Output dir [{parsed_args.output_dir}] exists; use --purge_results to clobber it"
            )
    return parsed_args


### Alpaca dataset formatting utilities
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def get_formatted_example(example: dict) -> dict:
    """Given a single example, format it based on whether or not we have an input provided.

    Args:
        example: dict
            Dictionary containing the keys for instruction / input / output, i.e., Alpaca formatted
            data.

    Returns:
        dict
            Dictionary containing the following:
                "input" - the formatted text to run the prediction on.
                "output" - the target text we aim to generate.
    """
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    formatted_input = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    return {
        # Text to run the prediction on
        "input": formatted_input,
        # Text to be generated (does not include the input str)
        "output": example["output"]
    }

### Model evaluation
def get_prediction_results(model: TunedCausalLM, data: datasets.arrow_dataset.Dataset, max_new_tokens: int, delimiter: str) -> tuple:
    """Runs the model over the alpaca formatted data to get the predictions / references to be used
    when computing the metrics of interest.

    Args:
        model: TunedCausalLM
            Model to be used for evaliuation.
        data: datasets.arrow_dataset.Dataset
            HF dataset to be processed for evaluation.
        max_new_tokens: int
            Max number of tokens to be used for generation.

    Returns:
        tuple
            Tuple containing:
                predictions [list of strings]
                references [list of strings]
                model_pred_file_info [list of dicts containing formatted data to be dumped later]          
    """
    predictions = []
    references = []
    model_pred_file_info = []
    for datum in tqdm(data):
        # Format the alpaca example
        formatted_datum = get_formatted_example(datum)
        # Run the formatted text through the model, and only save the newly generated text strings
        prediction = model.run(
            formatted_datum["input"],
            max_new_tokens=max_new_tokens,
            ret_gen_text_only=True,
        )
        # Save the raw output / predicted texts
        processed_pred = postprocess_output(prediction, delimiter)
        processed_ref = postprocess_output(formatted_datum["output"], delimiter)
        predictions.append(processed_pred)
        references.append(processed_ref)
        model_pred_file_info.append({
            "formatted input": formatted_datum["input"],
            "predicted target": processed_pred,
            "ref target": processed_ref,
        })
    return predictions, references, model_pred_file_info

def postprocess_output(output_text, delimiter):
    """NOTE: We are returning a list here, since that is what the one hot encoder module expects. """
    if delimiter is not None:
        return [text_substr.strip() for text_substr in output_text.split(delimiter)]
    return [output_text.strip()]
### Metric computation/display & utils for mapping labels to numerics for hf evaluate
def map_predictions_and_references_to_numerics(predictions: list, references: list) -> tuple:
    """Maps string predictions and references to numerics for use in accuracy and
    f1 computations. This process is generally ambiguous and can be done a number of
    ways, but the strategy we use is as follows:

    - Prior to consideration, all predictions and references are stripped of whitespace
    - Map all unique reference values to integers
    - Apply mapping of ref -> int to predictions; anything else is mapped to an unknown label val
      where the unknown label is treated as its own class

    Important caveats:
    - this strategy is case sensitive
    - this cannot be used for multioutput classification problems as is, since the entire
      predicted text is treated as a single label
    - be careful about the value of the max number of tokens for generation, since this
      essentially boils down to a string match problem

    Args:
        predictions: list
            List of strings to be converted to class indices predicted by model.
        references[list]
            List of strings to be converted to class indices for ground truth.

    Returns:
        tuple
            Tuple containing:
                int_predictions [list of ints] class indices for predicted samples
                ref_predictions [list of ints] class indices for ground truth samples
                label_map [dict] dict mapping indices to strings
    """
    le = preprocessing.LabelEncoder()
    le.fit(predictions)
    # Label encoder maps from class indices from [0, n-1], so we use n as our throwaway class
    unk_label = le.classes_.shape[0]
    int_predictions = [get_encoded_label(le, pred, unk_label) for pred in predictions]
    int_references = [get_encoded_label(le, references, unk_label) for pred in predictions]
    # Generate the class mapping + the unk label
    label_map = {
        idx: label for idx, label in enumerate(le.inverse_transform(list(range(unk_label))))
    }
    label_map[unk_label] = "<UNKNOWN LABEL>"
    return int_predictions, int_references, label_map

def get_encoded_label(le: preprocessing.LabelEncoder, gen_text: str, unk_label: int) -> int:
    """Gets the encoded label of a text string.
    Args:
        le: preprocessing.LabelEncode
            Label Encoder object which maps text strings into class indices.
        gen_text: str
            Text that was generated as a label by the model.
        unk_label: int
            Label to be used for unknown / garbage generation, i.e., things unknown to the
            label encoder.

    Returns:
        int
            The integer label index corresponding to the generated text.
    """
    try:
        return le.transform(gen_text)[0]
    except ValueError:
        # Model generated text that is not a valid label, i.e., is not in the label encoder
        return unk_label


def compute_metrics_dict(int_preds: list, int_references: list) -> dict:
    """Calculate the metrics on the (int) lists of preds against ground truth.
    
    Args:
        int_preds: list
            list of class indices for texts generated by the model.
        int_references: list
            list of class indices for ground truth labels.

    Returns:
        dict
            Dictionary containing F1 / accuracy metrics.
    """
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    # Compute the micro & macro f1 scores
    micro_f1 = f1_metric.compute(predictions=int_preds, references=int_references, average="micro")
    macro_f1 = f1_metric.compute(predictions=int_preds, references=int_references, average="macro")
    # Compute the accuracy
    accuracy = accuracy_metric.compute(predictions=int_preds, references=int_references)
    return {
        "f1": {
            "micro": micro_f1,
            "macro": macro_f1,
        },
        "accuracy": accuracy
    }


#### Replaces legacy logic
def map_predictions_and_references_to_one_hot_encoded_vectors(predictions: list, references: list):
    # Currently it's a stub that we can use to validate our metric correctness
    references = [[1,1,0], [1,0,1], [0,0,1], [0,0,0]]
    predictions = [[0,1,1], [1,0,0], [0,0,1], [1,0,1]]
    label_map = {0: "bird", 2: "cat", 3: "dog"}
    # In this scenario, dog basically represents <UNK>
    return references, predictions, label_map

def compute_metrics_dict_multi(enc_preds, enc_refs):
    micro_f1 = f1_score(enc_refs, enc_preds, average="micro")
    macro_f1 = f1_score(enc_refs, enc_preds, average="macro")
    # NOTE: For the multiclass / multilabel scenario, sklearn accuracy does NOT assign partial
    # credit, i.e., instances are only considered correct if they match the ground truth
    # one hot encoded vectors exactly.
    accuracy = accuracy_score(enc_refs, enc_preds)
    return {
        "f1": {
            "micro": micro_f1,
            "macro": macro_f1,
        },
        "accuracy": accuracy
    }

def export_experiment_info(metrics_dict: dict, label_map: dict, model_pred_file_info: dict, experiment_metadata: dict, output_dir: str):
    """Creates an exports all experiments info / metadata.

    Args:
        metrics_dict: dict
            Dictionary containing metrics of interest (i.e., F1 / accuracy).
        label_map: dict
            Mapping of class integers / labels.
        model_pred_file_info: dict
            List of dicts containing formatted data to be processed.
        experiment_metadata: dict
            Other experiment metadata of interest, e.g., model name, max new tokens, etc.
        output_dir: str
            Directory name to be created to hold the experiment files.
    """
    os.mkdir(output_dir)
    with open(os.path.join(output_dir, "eval_metrics.json"), "w") as metrics_fp:
        json.dump(metrics_dict, metrics_fp, indent=4, sort_keys=True)
    # Dump the label map to a file for debugging purposes
    with open(os.path.join(output_dir, "label_map.json"), "w") as map_fp:
        json.dump(label_map, map_fp, indent=4, sort_keys=True)
    # Also, dump the predictions / references info to a file for debugging purposes
    with open(os.path.join(output_dir, "preds_and_references.json"), "w") as preds_fp:
        json.dump(model_pred_file_info, preds_fp, indent=4, sort_keys=True)
    with open(os.path.join(output_dir, "experiment_metadata.json"), "w") as exp_md_fp:
        json.dump(experiment_metadata, exp_md_fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args = parse_and_validate_args()
    model = TunedCausalLM.load(args.model)
    data = datasets.load_dataset("json", data_files=args.data_path, split=args.split)
    predictions, references, model_pred_file_info = get_prediction_results(model, data, args.max_new_tokens, args.delimiter)
    int_preds, int_references, label_map = map_predictions_and_references_to_one_hot_encoded_vectors(predictions, references)
    metrics_dict = compute_metrics_dict_multi(int_preds, int_references)
    experiment_metadata = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "data_path": args.data_path,
    }
    export_experiment_info(metrics_dict, label_map, model_pred_file_info, experiment_metadata, args.output_dir)



"""
python3 run_evaluation.py --model TinyLlama/TinyLlama-1.1B-step-50K-105b  --data_path stanford_alpaca/alpaca_data.json  --max_new_tokens 10

{
    'input': '',
    'instruction': 'Give three tips for staying healthy.',
    'output': '1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.'
}

note: we need scikit learn and evaluate for this script [since f1 is also written on top of sklearn]
"""
"""Runs evaluation on Alpaca formatted data.

Metrics used: Accuracy / Micro & Macro F1.
"""
import os
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score
import argparse
import json
from tqdm import tqdm
import datasets
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


### Metric computation/display & utils for mapping labels to numerics for sklearn
def map_predictions_and_references_to_encoded_vectors(predictions: list, references: list):
    ohe = preprocessing.OneHotEncoder()
    # Extract the unique (potentially delimited labels) to fit the one hot encoder. We need to do
    # this directly in case it's a multiclass/multilabel scenario, because the 2D arr consumed
    # by the OHE expected consistent axis shapes, i.e., columns are treated as different features,
    # and cannot have a variable number of values.
    unk_label = "<UNKNOWN>"
    unique_labels = extract_unique_labels(predictions, references, unk_label)
    ohe.fit(unique_labels)

    # Now get the encoded vectors for our references and our predictions by one hot encoding
    # theunique sublabels and collapsing them into one vector along the row dimension.
    reference_vectors = [get_encoded_vector(ohe, refs, unk_label) for refs in references]
    pred_vectors = [get_encoded_vector(ohe, preds, unk_label) for preds in predictions]

    # For debugging purposes - map the indices in our none hot encoded entries.
    # NOTE: the categories_ attr is a 2D array of features, and we only care about [0]
    # since the uniquely extracted labels are only single dim features when fitting
    # the transform itself.
    label_map = {
        idx: label for idx, label in enumerate(ohe.categories_[0])
    }
    return pred_vectors, reference_vectors, label_map

def get_encoded_vector(ohe, texts, unk_label) -> int:
    # Since our encoded vector is built on collapsing one hot encoded vectors,
    # we need to explicitly handle the empty case since it is not one hot encodable.
    # raise ValueError(np.zeros(len(ohe.categories_[0])).dtype )
    if not texts:
        return np.zeros(len(ohe.categories_[0]))
    # Clean the generated text list; anything that is in the list that is not known to the
    # one hot encoder gets replaced by the unk_label. It is okay if we have multiple unk_labels
    # in the vector, since all of these just map to one positive entry in the encoded vector.
    cleaned_texts = list(set([text if text in ohe.categories_[0] else unk_label for text in texts]))
    # Encode the cleaned text as a 2D feature array of one hot encoded vectors
    vec_stack = ohe.transform([[text] for text in cleaned_texts]).toarray()

    # Then collapse the one hot encoded vectors along the column dimension to get
    # get the encoded binary vector for the multilabel / multiclass prediction.
    return vec_stack.sum(axis=0)

def extract_unique_labels(predictions, references, unk_label):
    """Grab all of the unique labels and return them as a list of single feature lists."""
    unique_ref_labels = set()
    for ref in references:
        for sub_label in ref:
            # This is pretty unlikely to happen (class named "<UNKNOWN>"), but for now, raise
            # if we see it happen, since that will currently mess up the results a little bit.
            if sub_label == unk_label:
                raise ValueError(f"Unk label {unk_label} is being used as a ground truth label!")                
            unique_ref_labels.add(sub_label)

    ref_label_list = [[label] for label in unique_ref_labels]
    # HACK - traverse the predictions and see if any unk predictions were made; if so, make a
    # garbage <UNKNOWN> class, which we will mark as false positives here.
    for pred in predictions:
        for sub_pred in pred:
            # One of our delimited predictions is unknown!
            if sub_pred not in unique_ref_labels:
                # Add the unk label once we know that it isn't a field in our eval data
                print("Adding <unk> label to handle garbage label generation")
                ref_label_list.append([unk_label])
                return ref_label_list
    return ref_label_list

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
    int_preds, int_references, label_map = map_predictions_and_references_to_encoded_vectors(predictions, references)
    metrics_dict = compute_metrics_dict_multi(int_preds, int_references)
    experiment_metadata = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "data_path": args.data_path,
    }
    export_experiment_info(metrics_dict, label_map, model_pred_file_info, experiment_metadata, args.output_dir)

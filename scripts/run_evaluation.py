"""Runs evaluation on Alpaca formatted data.

Metrics used: Accuracy / Micro & Macro F1.
"""
# Standard
from shutil import rmtree
from typing import Any, Optional
import argparse
import json
import os

# Third Party
from run_inference import TunedCausalLM
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import datasets
import numpy as np


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
        "--max_new_tokens",
        help="Max new tokens to use in generation",
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        help="Directory path to export results to",
        default="eval_results",
    )
    parser.add_argument(
        "--delimiter",
        help="Delimiter to be used for multilabel multiclass evaluation",
        default=None,
    )
    parser.add_argument(
        "--eos_token",
        help="EOS token emitted by the model; will recursively remove the token if present",
    )
    parser.add_argument(
        "--use_instruction",
        help="Indicates whether or not the instruction field should be used in formatting",
        action="store_true",
    )
    parser.add_argument("--purge_results", action=argparse.BooleanOptionalAction)
    parsed_args = parser.parse_args()

    print(f"Multiclass / multioutput delimiter: {parsed_args.delimiter}")
    # If we have a collision on the outdir, only remove the existing file if we explicitly say to
    if os.path.exists(parsed_args.output_dir):
        if parsed_args.purge_results:
            print(
                f"Existing output file/directory: [{parsed_args.output_dir}] will be deleted..."
            )
            rmtree(parsed_args.output_dir)
        else:
            raise FileExistsError(
                f"Output dir [{parsed_args.output_dir}] exists; use --purge_results to clobber it"
            )
    return parsed_args


### Alpaca dataset formatting utilities
PROMPT_DICT = {
    "prompt_input": (
        # pylint: disable=line-too-long
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


def get_formatted_example(
    example: dict[str, str], use_instruction: bool
) -> dict[str, str]:
    """Given a single example, format it based on whether or not we have an input provided.

    Args:
        example: dict[str, str]
            Dictionary containing the keys for instruction / input / output, i.e., Alpaca formatted
            data.
        use_instruction: bool
            Indicates whether or not the instruction field will be used.

    Returns:
        dict[str, str]
            Dictionary containing the following:
                "input" - the formatted text to run the prediction on.
                "output" - the target text we aim to generate.
    """
    # NOTE: Currently we ignore the instruction field due to the type of tasks we're tuning against
    if use_instruction:
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        formatted_input = (
            prompt_input.format_map(example)
            if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
        )
    else:
        formatted_input = f"Input: \n{example.get('input')}\n\n### Response:"
    return {
        # Text to run the prediction on
        "input": formatted_input,
        # Text to be generated (does not include the input str)
        "output": example["output"],
    }


### Model evaluation
def get_prediction_results(
    model: TunedCausalLM,
    data: datasets.arrow_dataset.Dataset,
    max_new_tokens: int,
    use_instruction: bool,
    delimiter: Optional[str],
    eos_token: Optional[str],
) -> tuple[list]:
    """Runs the model over the alpaca formatted data to get the predictions / references to be used
    when computing the metrics of interest.

    Args:
        model: TunedCausalLM
            Model to be used for evaliuation.
        data: datasets.arrow_dataset.Dataset
            HF dataset to be processed for evaluation.
        max_new_tokens: int
            Max number of tokens to be used for generation.
        use_instruction: bool
            Indicates whether or not the instruction field should be used.
        delimiter: Optional[str]
            Delimiter to be used for splitting apart multioutput instances.
        eos_token: Optional[str]
            EOS token emitted by the model, which will be recursively removed from predictions.
    Returns:
        tuple[list]
            Tuple containing:
                predictions [list of strings]
                references [list of strings]
                model_pred_info [list of dicts containing formatted data to be dumped later]
    """
    preds = []
    refs = []
    model_pred_info = []
    for datum in tqdm(data):
        # Format the alpaca example
        formatted_datum = get_formatted_example(datum, use_instruction)
        # Run the formatted text through the model, and only save the newly generated text strings
        prediction = model.run(
            formatted_datum["input"],
            max_new_tokens=max_new_tokens,
            ret_gen_text_only=True,
        )
        # Save the raw output / predicted texts
        processed_pred = postprocess_output(prediction, delimiter, eos_token)
        # The reference text should not have an EOS to strip
        processed_ref = postprocess_output(formatted_datum["output"], delimiter, None)
        preds.append(processed_pred)
        refs.append(processed_ref)
        model_pred_info.append(
            {
                "formatted input": formatted_datum["input"],
                "predicted target": processed_pred,
                "ref target": processed_ref,
            }
        )
    return preds, refs, model_pred_info


def postprocess_output(
    output_text: str, delimiter: Optional[str], eos_token: Optional[str]
) -> list[str]:
    """NOTE: We are returning a list here, since that is what the one hot encoder module expects.
    Args:
        output_text: str
            Raw text to be split into one or more (potentially) delimited instances.
        delimiter: Optional[str]
            Delimiter to be used for splitting apart multioutput instances.
        delimiter: Optional[str]
            Delimiter to be used for splitting apart multioutput instances.

    Returns
        list[str]
            List of one or more labels.
    """
    if eos_token is not None:
        while output_text.removesuffix(eos_token) != output_text:
            output_text = output_text.removesuffix(eos_token)
    if delimiter is not None:
        return [text_substr.strip() for text_substr in output_text.split(delimiter)]
    return [output_text.strip()]


### Metric computation/display & utils for mapping labels to numerics for sklearn
def map_predictions_and_references_to_encoded_vectors(
    predictions_lists: list[list[str]], references_lists: list[list[str]]
) -> tuple[Any]:
    """Maps the delimited text lists to lists of encoded vectors.

    Args:
        predictions_lists: list[list[str]]
            Delimited text lists for model predictions to be encoded.
        references_lists: list[list[str]]
            Ground truth delimited text lists to be encoded.

    Returns:
        tuple[Any]
            tuple containing:
                pred_vectors [list of encoded 1D numpy arrays]
                reference_vectors [list of encoded 1D numpy arrays]
                label_map dict[str, str] - maps class indices to labels
    """
    if not predictions_lists or not references_lists:
        raise ValueError("Predictions and/or references should not be empty!")
    ohe = preprocessing.OneHotEncoder()
    # Extract the unique (potentially delimited labels) to fit the one hot encoder. We need to do
    # this directly in case it's a multiclass/multilabel scenario, because the 2D arr consumed
    # by the OHE expected consistent axis shapes, i.e., columns are treated as different features,
    # and cannot have a variable number of values.
    unk_label = "<UNKNOWN>"
    unique_labels = extract_unique_labels(
        predictions_lists, references_lists, unk_label
    )
    ohe.fit(unique_labels)

    # Now get the encoded vectors for our references and our predictions by one hot encoding
    # theunique sublabels and collapsing them into one vector along the row dimension.
    reference_vectors = [
        get_encoded_vector(ohe, refs, unk_label) for refs in references_lists
    ]
    pred_vectors = [
        get_encoded_vector(ohe, preds, unk_label) for preds in predictions_lists
    ]

    # For debugging purposes - map the indices in our none hot encoded entries.
    # NOTE: the categories_ attr is a 2D array of features, and we only care about [0]
    # since the uniquely extracted labels are only single dim features when fitting
    # the transform itself.
    label_map = dict(enumerate(ohe.categories_[0]))
    return pred_vectors, reference_vectors, label_map


def get_encoded_vector(
    ohe: preprocessing.OneHotEncoder, texts: list[str], unk_label: str
) -> np.typing.NDArray:
    """Get the encoded vector representing one or more generated texts by one hot encoding each
    individual text and collapsing the result.

    Args:
        ohe: preprocessing.OneHotEncoder
            Sklearn one hot encoder to be used for one hot encoding all texts
            (including the garbage class if we have one).
        texts: list[str]
            List of texts to be encoded and collapsed into one vector.
        unk_label: str
            Label to be used for garbage generations.

    Returns:
        np.typing.NDArray
            Binary vector encoding the list of texts as labels.
    """
    # Since our encoded vector is built on collapsing one hot encoded vectors,
    # we need to explicitly handle the empty case since it is not one hot encodable.
    # raise ValueError(np.zeros(len(ohe.categories_[0])).dtype )
    if not texts:
        return np.zeros(len(ohe.categories_[0]))
    # Clean the generated text list; anything that is in the list that is not known to the
    # one hot encoder gets replaced by the unk_label. It is okay if we have multiple unk_labels
    # in the vector, since all of these just map to one positive entry in the encoded vector.
    cleaned_texts = list(
        {text if text in ohe.categories_[0] else unk_label for text in texts}
    )

    # Encode the cleaned text as a 2D feature array of one hot encoded vectors
    vec_stack = ohe.transform([[text] for text in cleaned_texts]).toarray()

    # Then collapse the one hot encoded vectors along the column dimension to get
    # get the encoded binary vector for the multilabel / multiclass prediction.
    return vec_stack.sum(axis=0)


def extract_unique_labels(
    preds: list[list[str]], refs: list[list[str]], unk_label: str
) -> list[list[str]]:
    """Grab all of the unique labels and return them as a list of single feature lists.
    Args:
        preds: list[list[str]]
            List of lists, where each sublist contains the stripped delimited substrings of a
            single model prediction.
        refs: list[list[str]]
            List of lists, where each sublist contains the stripped delimited substrings of a
            single ground truth reference.
        unk_label: str
            Label to be used for Unknown - this class is only created in evaluation if the
            generative model predicts something that is not present in the ground truth refs.

    Returns:
        list[list[str]]
            List of single value lists, each of which contains a single label.
    """
    unique_ref_labels = set()
    for ref in refs:
        for sub_label in ref:
            # This is pretty unlikely to happen (class named "<UNKNOWN>"), but for now, raise
            # if we see it happen, since that will currently mess up the results a little bit.
            if sub_label == unk_label:
                raise ValueError(
                    f"Unk label {unk_label} is being used as a ground truth label!"
                )
            unique_ref_labels.add(sub_label)

    ref_label_list = [[label] for label in unique_ref_labels]
    # HACK - traverse the predictions and see if any unk predictions were made; if so, make a
    # garbage <UNKNOWN> class, which we will mark as false positives here.
    for pred in preds:
        for sub_pred in pred:
            # One of our delimited predictions is unknown!
            if sub_pred not in unique_ref_labels:
                # Add the unk label once we know that it isn't a field in our eval data
                print("Adding <unk> label to handle garbage label generation")
                ref_label_list.append([unk_label])
                return ref_label_list
    return ref_label_list


def compute_metrics_dict_multi(
    enc_preds: list[np.typing.NDArray], enc_refs: list[np.typing.NDArray]
) -> dict[str, Any]:
    """Calculate the metrics based on the encoded prediction and reference vector lists.
    Current metrics: precision, recall f1, accuracy

    Args:
        enc_preds: list[np.typing.NDArray]
            List of encoded binary vectors for predictions from the model.
        enc_refs: list[np.typing.NDArray]
            List of encoded binary vectors for ground truth references.

    Returns:
        dict[str, Any]
            Dictionary of metrics.
    """
    micro_f1 = f1_score(enc_refs, enc_preds, average="micro", zero_division=np.nan)
    macro_f1 = f1_score(enc_refs, enc_preds, average="macro", zero_division=np.nan)
    # For recall - the UNK class containing only false positives does NOT affect score.
    micro_recall = recall_score(
        enc_refs, enc_preds, average="micro", zero_division=np.nan
    )
    macro_recall = recall_score(
        enc_refs, enc_preds, average="macro", zero_division=np.nan
    )
    micro_prec = precision_score(
        enc_refs, enc_preds, average="micro", zero_division=np.nan
    )
    macro_prec = precision_score(
        enc_refs, enc_preds, average="macro", zero_division=np.nan
    )
    # NOTE: For the multiclass / multilabel scenario, sklearn accuracy does NOT assign partial
    # credit, i.e., instances are only considered correct if they match the ground truth
    # encoded vectors exactly.
    accuracy = accuracy_score(enc_refs, enc_preds)
    return {
        "f1": {
            "micro": micro_f1,
            "macro": macro_f1,
        },
        "recall": {
            "micro": micro_recall,
            "macro": macro_recall,
        },
        "precision": {
            "micro": micro_prec,
            "macro": macro_prec,
        },
        "accuracy": accuracy,
    }


def export_experiment_info(
    metrics: dict[str, Any],
    label_map: dict[str, str],
    model_pred_info: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_dir: str,
):
    """Creates an exports all experiments info / metadata.

    Args:
        metrics: dict[str, Any],
            Dictionary containing metrics of interest (i.e., F1 / accuracy).
        label_map: dict[str, str]
            Mapping of class integers / labels.
        model_pred_info: list[dict[str, Any]]
            List of serializable dicts containing formatted data to be processed.
        metadata: dict[str, Any]
            Other experiment metadata of interest, e.g., model name, max new tokens, etc.
        output_dir: str
            Directory name to be created to hold the experiment files.
    """
    os.mkdir(output_dir)
    with open(
        os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8"
    ) as metrics_fp:
        json.dump(metrics, metrics_fp, indent=4, sort_keys=True)
    # Dump the label map to a file for debugging purposes
    with open(
        os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8"
    ) as map_fp:
        json.dump(label_map, map_fp, indent=4, sort_keys=True)
    # Also, dump the predictions / references info to a file for debugging purposes
    with open(
        os.path.join(output_dir, "preds_and_references.json"), "w", encoding="utf-8"
    ) as preds_fp:
        json.dump(model_pred_info, preds_fp, indent=4, sort_keys=True)
    with open(
        os.path.join(output_dir, "experiment_metadata.json"), "w", encoding="utf-8"
    ) as exp_md_fp:
        json.dump(metadata, exp_md_fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args = parse_and_validate_args()
    tuned_model = TunedCausalLM.load(args.model)
    eval_data = datasets.load_dataset(
        "json", data_files=args.data_path, split=args.split
    )
    predictions, references, model_pred_file_info = get_prediction_results(
        tuned_model,
        eval_data,
        args.max_new_tokens,
        args.use_instruction,
        args.delimiter,
        args.eos_token,
    )

    (
        pred_vecs,
        ref_vecs,
        eval_label_map,
    ) = map_predictions_and_references_to_encoded_vectors(predictions, references)
    metrics_dict = compute_metrics_dict_multi(pred_vecs, ref_vecs)
    experiment_metadata = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "data_path": args.data_path,
    }
    export_experiment_info(
        metrics_dict,
        eval_label_map,
        model_pred_file_info,
        experiment_metadata,
        args.output_dir,
    )
    print(f"Exported results to: {args.output_dir}")

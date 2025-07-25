# Third Party
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch

# First Party
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_pa(model, tokenizer, languages, num_samples=100):
    def make_prompt(s1, s2, pretext=""):
        return f"""{pretext}Question: Are the following two sentences paraphrases of each other?

    Sentence 1: "{s1}"
    Sentence 2: "{s2}"

    Answer with 'Yes' or 'No' only.

    Answer:"""

    model.eval()
    acc = []
    for language in languages:
        dataset = load_dataset("xtreme", f"PAWS-X.{language}")["test"]
        preds = []
        labels = []
        pretext = ""
        shots = 5
        for item in dataset.select(range(shots)):
            pretext += make_prompt(item["sentence1"], item["sentence2"])
            if int(item["label"]) == 1:
                pretext += " Yes\n\n"
            else:
                pretext += " No\n\n"
        for item in tqdm(
            dataset.shuffle(seed=42).select(range(num_samples)),
            desc=f"Evaluating PA({language})",
        ):  # limit for quick test
            missed = 0
            prompt = make_prompt(item["sentence1"], item["sentence2"], pretext)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            output_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).lower()
            if "yes" in output_text and "no" not in output_text:
                pred = 1
            elif "no" in output_text and "yes" not in output_text:
                pred = 0
            elif "yes" in output_text and "no" in output_text:
                pred = 1 if output_text.index("yes") < output_text.index("no") else 0
            else:
                missed += 1
                continue
            preds.append(pred)
            labels.append(int(item["label"]))
            if missed > 0:
                print(f"Missed {missed} predictions for PA({language})")
        accuracy = accuracy_score(labels, preds)
        acc.append(accuracy)
        # if missed > 0:
        #     print(f"Accuracy for PA({language}): {accuracy:.4f} (missed {missed} predictions)")
        # else:
        #     print(f"Accuracy for PA({language}): {accuracy:.4f}")

    for language, accuracy in zip(languages, acc):
        print(f"PA({language}): {accuracy:.4f}")
    avg_accuracy = sum(acc) / len(acc)
    print(f"Average PA Accuracy: {avg_accuracy:.4f}")
    model.train()
    return avg_accuracy, acc


def evaluate_nli(model, tokenizer, languages, num_samples=100):
    def make_prompt(s1, s2, pretext=""):
        return f"""{pretext}Question: Are the following two sentences neutral, contradiction or entailment?

    Sentence 1: "{s1}"
    Sentence 2: "{s2}"

    Answer with 'neutral', 'contradiction' or 'entailment' only.

    Answer:"""

    model.eval()
    acc = []
    for language in languages:
        dataset = load_dataset("xtreme", f"XNLI")["test"]
        dataset = dataset.filter(lambda item: item["language"] == language)

        preds = []
        labels = []
        pretext = ""
        shots = 5
        for item in dataset.select(range(shots)):
            pretext += make_prompt(item["sentence1"], item["sentence2"])
            pretext += f" {item['gold_label']}\n\n"
        for item in tqdm(
            dataset.shuffle(seed=42).select(range(num_samples)),
            desc=f"Evaluating NLI({language})",
        ):  # limit for quick test
            missed = 0
            prompt = make_prompt(item["sentence1"], item["sentence2"], pretext)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            output_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).lower()
            target_labels = ["neutral", "contradiction", "entailment"]
            positions = [
                (label, output_text.find(label))
                for label in target_labels
                if label in output_text
            ]
            if positions:
                pred = min(positions, key=lambda x: x[1])[
                    0
                ]  # label with earliest position
            else:
                missed += 1
                continue
            preds.append(pred)
            labels.append(item["gold_label"])
            if missed > 0:
                print(f"Missed {missed} predictions for NLI({language})")
        accuracy = accuracy_score(labels, preds)
        acc.append(accuracy)

    for language, accuracy in zip(languages, acc):
        print(f"NLI({language}): {accuracy:.4f}")
    avg_accuracy = sum(acc) / len(acc)
    print(f"Average NLI Accuracy: {avg_accuracy:.4f}")
    model.train()
    return avg_accuracy, acc


def evaluate(model, tokenizer, tasks, languages, num_samples=100):
    for task in tasks:
        if task == "nli":
            evaluate_nli(model, tokenizer, languages, num_samples=num_samples)
        elif task == "pa":
            evaluate_pa(model, tokenizer, languages, num_samples=num_samples)
        else:
            raise ValueError(f"Unknown task {task}")


if __name__ == "__main__":
    # Standard
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "de", "fr", "es", "zh"],
        help="Languages to evaluate",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["pa", "nli"], help="Tasks to evaluate."
    ),
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model ID to use for evaluation",
    )
    args = parser.parse_args()
    num_samples = args.num_samples if args.num_samples > 0 else 10
    languages = args.languages if args.languages else ["en", "de", "fr", "es", "zh"]
    model_id = args.model if args.model else "meta-llama/Llama-3.2-3B"
    tasks = args.tasks if args.tasks else ["pa", "nli"]

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    evaluate(model, tokenizer, tasks, languages, num_samples=num_samples)

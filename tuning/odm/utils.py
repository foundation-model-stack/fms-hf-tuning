# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Third Party
from datasets import load_dataset
import json
# Local
from tuning.odm.dataset import OnlineDataMixing, SimpleTextDataset, UniformDataMixing, SequentialDataMixing


def get_pa_train_dataset(language, n=10):
    with open("config/data_templates.json", "r") as f:
        prompt_template = json.load(f)["pa"]

    def make_prompt(s1, s2, answer):
        return prompt_template.format(
            sentence1=s1,
            sentence2=s2,
            answer=answer
        )

    dataset = load_dataset("xtreme", f"PAWS-X.{language}")["train"]
    # TODO: shuffle dataset
    prompts = [make_prompt(item["sentence1"], item["sentence2"], item['label']) for item in dataset.select(range(n))]
    return prompts


def get_nli_train_dataset(language, n=10):
    choice_dict = {
        'neutral': 0,
        'contradiction': 1,
        'entailment': 2
    }
    with open("config/data_templates.json", "r") as f:
        prompt_template = json.load(f)["nli"]

    def make_prompt(s1, s2, answer):
        return prompt_template.format(
            sentence1=s1,
            sentence2=s2,
            answer=answer
        )

    dataset = load_dataset("xtreme", f"XNLI")["validation"]
    # TODO: shuffle dataset
    dataset = dataset.filter(lambda item: item["language"] == language)
    prompts = [make_prompt(item["sentence1"], item["sentence2"], choice_dict[item['gold_label']]) for item in dataset.select(range(n))]
    return prompts


def get_odm_dataset(
    model,
    tokenizer,
    languages,
    tasks,
    method="UniformDataMixing",
    num_samples=10,
    alpha=1.0,
    sample_interval=10, 
    update_interval=10,
    max_steps=None,
    sequence=None,
):

    texts = []
    tasks = set(tasks)
    for task in tasks:
        if task == "pa":
            for language in languages:
                prompts = get_pa_train_dataset(language, n=num_samples)
                texts.append(prompts)
        elif task == "nli":
            for language in languages:
                prompts = get_nli_train_dataset(language, n=num_samples)
                texts.append(prompts)
        else:
            raise ValueError(f"Unknown task {task}")

    task_lang = [f"{task}_{lang}" for task in tasks for lang in languages]

    all_ds = [SimpleTextDataset(text, tokenizer) for text in texts]
    print("Size of datasets:")
    for tlp, ds in zip(task_lang, all_ds):
        print(f"{tlp}: {len(ds)}")

    dataset = UniformDataMixing(all_ds, model, sample_interval, update_interval)
    if method == "onlinedatamixing":
        dataset = OnlineDataMixing(all_ds, model, sample_interval, update_interval, alpha=alpha)
    elif method == "sequentialdatamixing":
        dataset = SequentialDataMixing(all_ds, model, sample_interval, update_interval, sequence=sequence, max_steps=max_steps)

    return dataset

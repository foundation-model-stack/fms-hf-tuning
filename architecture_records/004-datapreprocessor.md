# Data Pre Processor Design For fms-hf-tuning

**Deciders(s)**: Sukriti Sharma (sukriti.sharma4@ibm.com), Will Johnson (Will.Johnson@ibm.com) , Abhishek Maurya (maurya.abhishek@ibm.com), Yu Chin Fabian Lim (flim@sg.ibm.com), Dushyant Behl (dushyantbehl@in.ibm.com), Ashok Pon Kumar (ashokponkumar@in.ibm.com)

**Date (YYYY-MM-DD)**:  2024-03-06

**Obsoletes ADRs**:  NA

**Modified By ADRs**:  NA

**Relevant Issues**: [1]

- [Summary and Objective](#summary-and-objective)
  - [Motivation](#motivation)
  - [User Benefit](#user-benefit)
- [Decision](#decision)
  - [Alternatives Considered](#alternatives-considered)
- [Consequences](#consequences)
- [Detailed Design](#detailed-design)

## Summary and Objective

The primary objective of the `DataPreProcessor` design for fms-hf-tuning is to provide a unified yet powerful interface for handling diverse data formats and configurations.
This interface should cater to various user expertise levels, enabling basic users to easily load and process data, while allowing advanced users to customize their data handling.

### Key Goals:
1. **Broad Data Format Support**: Allow datasets in formats such as Arrow, Parquet, and CSV.
1. **Compatibility with Multiple Datasets and Files**: Enable multiple files per dataset and interleaving or mixing of datasets.
1. **Support for Different Data Modalities**: Include images, audio, and text data, along with modality-specific preprocessing options.
1. **User-Focused Configurations**: Provide simple data loading for regular users, while enabling advanced configurations for expert users.
1. **Template-Based Preprocessing**: Support jinja template rendering, where necessary, for template-dependent preprocesing requirements.

### Motivation

The main motivation for this ADR stems from the fact that fms-hf-tuning is being used by many teams for a diverse set of use cases which are not currently supported in the library. To be precise, currently in the library for data preprocessing we currently take two primary arguments `training_data_path` and `validataion_data_path` which take in a single file location for a dataset.
A user can currently pass in
1. a pretokenized json(l) dataset via 
   ```
   --training_data_path <pretokenized dataset>
   ```
1. a preprocessed json(l) formats with a single sequence and a specified `response_template` to use for masking on completion.
    ```
    --training_data_path <dataset.json> --dataset_text_field <e.g. 'input'> --response_template <e.g. '\n### Label:'>
    ```
1. a json(l) dataset and a `data_formatter_template` to use the formatting function on fly.
    ```
    --training_data_path <dataset.json> --data_formatter_template <'template'>
    ```
1. a json(l) dataset with `input` and `output` fields, names hardcoded and cannot be changed.
    ```
    --training_data_path <non-pretokenized dataset with input and output fields>.json
    ```

The first motivation for a change is the requirements from users asking for different data formats, current code only supports json while there are teams which are training using Parquet and Arrow format so they require additional data format support.

Also use cases from teams require multiple datasets and even multiple data files in a dataset.

Further requirements from teams is to have a way to interleave datasets at run time by specifying static weights to mix different datasets which is also not supported by the code yet.

Finally other requirements are to have preprocesing support for multiple modalities of data (starting with Image first) and have support for advanced preprocesing like jinja based template rendering of the dataset before consumption.

All these requirements are new and are currently not supported by the library which motivated us to propose a change in the design of data preprocesing in this library to incorporate these and potentially any new changes in one go.

### User Benefit

Users will benefit from the additional argument which allows them to pass a single [`data_config`](#our-considerations-for-the-design) file specifying how to preprocess their dataset.
Our data config file will extend users the capabilities to,
1. Pass multiple data files and multiple datasets.
1. Specify static weights in the configuration to interleave datasets.
1. Define preprocessing routines to apply on the data and in which order

This will make the process of handling custom datasets which might require rendering jinja template or processing image data way much easier.

We do not require users to learn the specification of the additional `data_config` file, as the existing arguments to process dataset which are present in the code [`tuning.config.configs.DataArguments`](https://github.com/foundation-model-stack/fms-hf-tuning/blob/398c2a8fe26d734344240555585d95e05299faa8/tuning/config/configs.py#L67) will not be deprecated in this version and users can keep using the same data arguments for use cases being served by the library.

At the very least a user not well versed with the `data_config` will be able to pass in for e.g. a pre-tokenized pyarrow dataset 

```
--training_data_path <pre-tokenized>.pyarrow
```

And at full length they can specify multi-file, multi dataset configuration which can process dataset according to the config specified like,


```
...
  - name: dataset1
    sampling:
      ratio: 0.3
    data_paths:
      - /data/stackoverflow-kubectl_posts
      - /data/stackoverflow-kubernetes_posts
      - /data/stackoverflow-openshift_posts
    data_handlers:
      - name: render_template
        arguments:
          remove_columns: all
          batched: false
          fn_kwargs:
            jinja_template: "{<jinja-template>}"
```
A small part of the `data-config` spec which is provided in detailed in the section below.

## Decision

Some terminology before we move ahead 

<table>
  <tr>
    <th style="border: 1px solid black; padding: 5px;">User Persona</th>
    <th style="border: 1px solid black; padding: 5px;">Description</th>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 5px;">Simple User</td>
    <td style="border: 1px solid black; padding: 5px;">A user who uses this library to train models using a single dataset, passing it via a single command line argument.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 5px;">Advanced User</td>
    <td style="border: 1px solid black; padding: 5px;">A user with a deep understanding of datasets, who knows how to apply specific preprocessing and mixing techniques during training.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 5px;">Intermediate User</td>
    <td style="border: 1px solid black; padding: 5px;">A user who works with custom datasets but lacks full knowledge of data processing, relying on advanced users for guidance to fulfill their use case.</td>
  </tr>
</table>

Please note that most of the users of product here would fall into the simple user category while advanced and intermediate users are researchers looking to use our library for diverse set of use cases.

### Our considerations for the design

1. Allow advanced users to use full power of the HuggingFace library as much as possible without recreating the same.
1. Allow advanced users to specify custom data preprocessor pipeline in an easy way.
1. Ensure the single design can handle these and many more use cases without major changes.
1. Design for Advanced users while simplify for simple users.

We propose to allow advanced users to specify a full spec which exposes data preprocessing API provided by the HF library directly to them to be able to fully utilize the interface. 

The proposed input spec which user specifies as `data_config` on how to pass information for such preprocessing is

```
datapreprocessor:
    type: default
datasets:
  - name: dataset1
    sampling:
      ratio: 0.3
    data_paths:
      - /data/stackoverflow-kubectl_posts
      - /data/stackoverflow-kubernetes_posts
      - /data/stackoverflow-openshift_posts
    data_handlers:
      - name: tokenize
        arguments:
          remove_columns: all
          batched: false
  - name: dataset2
    sampling:
      ratio: 0.4
    data_paths:
      - /data/stackoverflow-kubectl_posts
      - /data/stackoverflow-kubernetes_posts
      - /data/stackoverflow-openshift_posts
    data_handlers:
      - name: render_template
        arguments:
          remove_columns: all
          batched: false
          fn_kwargs:
            jinja_template: "{<jinja-template>}"
      - name: tokenize
        arguments:
          remove_columns: all
          batched: false
```

To iterate again, here our goal is not to re-implement the functionality provided by HuggingFace but rather have a clean interface using a config where advanced users can use things like Iterable datasets or Interleaving datasets and perform custom preprocessing like applying jinja templates etc in an easy way.

In this spec, at top level we have the `Dataprocessor` config which contains just one field `type` which is set to `default`. This is done to ensure any future top level `dataprocessor` configs will go into this block. Users need not touch or provide this as the `default` is automatically selected.

The second block here is where users will list multiple `datasets` and each dataset will contain information on how to process it. We allow arguments like `sampling` for users to specify sampling ratios while [`interleaving datasets`](https://huggingface.co/docs/datasets/en/process#interleave) to use API like [`interleave_datasets`](https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.interleave_datasets) by HuggingFace.

The most powerful feature of this block is `data_handlers`. Here we allow users to specify a list of routines to apply on the dataset at the time of preprocessing. A `data_handler` is a [`map`](https://huggingface.co/docs/datasets/en/process#map) operation performed on the dataset to which a user can further pass informational arguments. We expose the full set of arguments of HF [`Dataset.map`](https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.Dataset.map) operation here to the user as `kwargs` of a handler.

As example in `dataset2` the data handler is requesting to apply a `render_template` function before tokenization on the dataset which processes the dataset and renders the `jinja template` specified as `fn_kwargs.jinja_template`, rest of the arguments like `remove_column` and `batched` are just HF Map API arguments.

```
- name: dataset2
    sampling:
      ratio: 0.4
    data_paths:
      - /data/stackoverflow-kubectl_posts
      - /data/stackoverflow-kubernetes_posts
      - /data/stackoverflow-openshift_posts
    data_handlers:
      - name: render_template
        arguments:
          remove_columns: all
          batched: false
          fn_kwargs:
            jinja_template: "{<jinja-template>}"
      - name: tokenize
        arguments:
          remove_columns: all
          batched: false
```

By allowing the users to specify data handlers like this we allow them to use full Hugging Face API and at the same time specify preprocessing routines in a fixed order. The handlers list specify a [`DAG`](https://en.wikipedia.org/wiki/Directed_acyclic_graph) of operations to apply on the dataset and will be executed by the code in that order.

Furthermore this design allows flexibility to be extended to any upcoming usecase because any operation to be executed on the dataset can be broken down into function execution implemented as data handlers.

This makes our spec a complete solution for advanced users of the library, who have custom preprocessing needs. Allowing them to specify complete preprocessing operations to be applied to the dataset via a config file.

Finally, with this spec we do not want to break the functionality for the simple users of the library. A simple user which wants to just use the library with a single dataset like today can pass the same dataset via `--training_data_path <file> --validataion_data_path <file>` arguments.

Infact we do not change the behavior currently supported by any of the `tuning.config.configs.DataArguments` arguments hence allowing the simple users of the library to continue using the library as is.

### Performance Considerations

Since this design allows complex preprocessing of the dataset on fly, the design should incorporate performance measures to ensure that the system is not performing too slow or spending too much time while preprocessing the dataset to affect tuning/training time.

The goal that we have here is to not be slower than the HuggingFace library which our whole design is based upon, in this sense we also imagine any performance improvements that we come across to be contributed back to HF library to keep our design simple and not reimplement stuff.

#### Handling Large Dataset

Our main reason for using HF [Map](https://huggingface.co/docs/datasets/en/process#map) heavily for data preprocessing is that for large datasets which are generally loaded as `IterableDatasets` the MAP API automatically performs [`lazy map operations`](https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable#eager-data-processing-and-lazy-data-processing) and hence doesn't produce too much overhead while training.

#### Caching intermediate dataset

Hugging Face caches intermediate map operations which makes replay of our data preprocessor easier if same map parameters and operations are applied. If the file system is an issue we have two considerations,

1. Keep intermediate datasets in memory while preprocessing using [`keep_in_memory=True`](https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.Dataset.map.keep_in_memory), for large datasets and Iterable datasets we assume this to be for mini batches.
1. Disable caching in file system and make it configurable by [`disable_caching()`](https://huggingface.co/docs/datasets/en/cache#enable-or-disable-caching) API from HuggingFace.

These considerations can be made dynamicallly or can be passed via users as we allow any `kwargs` to be passed to the map operations.

## Alternatives Considered

### Letting users process their own data and pass file(s) directly to this library.

A simple alternative to avoid all this is to have the users process their own data, this is also in lines of the fact that most workloads contain preprocessed data which is used by simple users as is for their tuning/training.

The reason to have this design is that many users coming to this library have advanced set of use cases. As stated in the motivation we are getting ever increased demand from researchers looking to use this library are looking for features like `jinja template` rendering, image data processing, mixing and merging datasets. While this can be done at user level most users are not looking to write code to do all this preprocessing but use tools which implement them to perform these tasks. 
Leaving all users to write their own preprocessing logic can also lead to code duplication across many teams which is something we want to avoid.

More importantly as stated in the motivation we are getting ever increased demand from users who want to use this library directly with their dataset and have quick roundtrip for testing. This design allows users to specify simple parameters in the config and test for complex usecases easily.

### Passing all datasets we take to the HuggingFace SFTTrainer API and let it handle them without preprocessing at our end.

Another alternative we have is to take the `dataset` input to this library and pass it directly to the trainer `SFTTrainer` in our case directly and let it handle loading and preprocessing the dataset.

[SFTTrainer](https://huggingface.co/docs/trl/v0.12.1/en/sft_trainer#trl.SFTTrainer) supports specifying the `train_dataset` and `eval_dataset` for both of which it supports Iterable datasets along with normal datasets allowing us to pass a large dataset supported via streaming. 

Please note that even in this case users will need to tell us that the dataset is large and is to be loaded via `streaming=True` because the argument which tells HF to load the dataset in Iterable mode or standard mode is passed to [`load_dataset`](https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/loading_methods#datasets.load_dataset)

```
from datasets import load_dataset
train_ds = load_dataset('imdb', split='train', streaming=True)
```

Additionally, `SFTTrainer` has support for [data formatting function](https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support). Users can pass a `formatting_function` directly to `SFTtrainer` which formats the dataset for them,

```
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
)

trainer.train()
```
Taken from [HuggingFace docs](https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support)

As our library is a wrapper on top of HF we cannot directly allow users to pass a custom formatting function and
our `data_handler` design can also support formatting dataset in a similar way to `formatting function` where users specify just name of the handler and we apply formatting on our end. The design for `data_handler` that we have is a superset of this feature which is more flexible and can support many more use cases.

## Consequences

### Arguments Required
In this design, apart from the `data_config` spec users will also need to pass the `--response_template` argument. This is because the `DataCollator` functionality of this library is not being touched by our design.

Also users who process JSON dataset via our interface need to specify `--dataset_text_field` which is inferred from the `DataArguments` for now and not passed inside the data_config to ensure the simple interface remains same.

We also plan to add a new argument to `tuning.config.configs.DataArguments` which takes in the `data_config` file as input. like,
```
@dataclass
class DataArguments:
...
    data_config_file: str = field(
        default=None,
        metadata={
            "help": "data_config file which specifies the data preprocessing logic to apply.\
                         Supports both JSON and YAML based config files"
        },
    )
```

### Understanding the spec
With this design we have tried to keep our design simple and close to the HF library as much as possible, e.g. exposing the same map `kwargs` that HF has in our `data_handlers`.

Despite this advanced users will need to understand the spec to be able to write it properly.

Advanced users will also need to educate themselves on the data handlers already present in the code. Since the data handlers are selected based on their name we need to ensure the documentation contains complete information on what different data handlers are present and how to use them in the `data_config`.

### Sharing config files
We currently do not propose anything on how advanced users share the `data_config` files created by them with Intermediate and Simple users. This is left outside the scope of our library.

### Simple User Perspective

As mentioned above we are retaining the full functionality supported by `tuning.config.configs.DataArguments` which means simple users can continue using the library by passing a simple dataset via `--training_data_path` and use case specific arguments like `--data_formatter_template` as they please and the code will internally handle how to map these to the `data_config` spec.

### Intermediate User Perspective
Our perspective is that the advanced users will create config files for data preprocessing and the intermediate users can use these existing configs and modify them according to their preference to get the desired result.

## Detailed Design

### The proposed design to implement support for this spec is follows,

Data Pre Processor abstract class

```
class DataPreProcessor(ABC):

    tokenizer = None
    model_name_or_path = None
    block_size = None
    data_config: DataConfig = None
    data_handlers: Dict[str, Callable] = None

    def __init__(self, dataconfig: DataConfig, tokenizer, model_name_or_path, block_size):
        self.data_config = dataconfig
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.block_size = block_size
        self.data_handlers = {}

    def register_data_handler(self, name: str, d: Callable):
        self.data_handlers[name] = d

    @abstractmethod
    def process_data_config(self, data_config: DataConfig):
        pass
```

At the top level we propose to have this `class DataPreProcessor` which is an abstract class and requires functions to process the data config proposed above.

The data pre processor needs to support custom data handlers. In the library for simple use cases we will provide predefined data handlers which need to be registered with the top level class using the 
call `DataPreProcessor.register_data_handler`.

The simple use cases will be handled using these data handlers and which data handler to choose will depend on the use case chosen from data args (same as the current code).

## How are handlers provided and registered - 

Data handlers are python callables which can be called on single/few samples of data and can perform things like tokenising the data, applying tools like jinja template or even things like encoding or decoding multi modal formats like images/audio for processing by the model.

The abstract datapreprocessor class provides a way to register datahandler against a `name` which is a string. The data handler config `DataHandlerConfig` taken by `execute_data_handlers` represents a DAG of data handling routines which are to be executed on the data. 

For standard HF API you can think of these as the HF Processing routines which could be Map/Filter/Select operations. We implement most of the routines as map based operations. The current code also implements functionality like tokenization of data or data formatting via map e.g. 
`tuning/utils/preprocessing_utils.py::get_preprocessed_dataset` such functionality can be retained as predefined data handlers.

The implementation is flexible enough for very advanced users to specify their own implementation of data handling routines by importing fms-hf-tuning and extending the preprocessing by calling `register_data_handler` on the preprocessor. This is left for advanced users of the library and not for simple users.

# Implementation of the default Data Preprocessor.

The default data preprocessor implemented as an instance of the `DataPreProcessor` class uses HF APIs where ever possible
to miminize reimplementation of code.

The HF datapreprocessor processes different type of files via its `load_dataset` factory. If not supported automatically via this, we can look to extend the factory to use an other type of interest via
`Dataset.from_generator(<generator>)` functionality.

This also means that any implementation like `get_json_object` which load `json(l)` and then return a custom json dict
can be implemented as data handlers.

### Interleaving datasets

In case of multiple datasets the user can request how the datasets are to be interleaved.
The probabilities specified by users in the config `sampling.ratio` can be collected from individual datasets and passed to
[`datasets.interleave_datasets`](https://huggingface.co/docs/datasets/v3.0.1/en/package_reference/main_classes#datasets.interleave_datasets).

### Streaming datasets

In HuggingFace the `streaming` argument can be handled by using `IterableDatasets` instead of standard `Datasets`.
HF provides same APIs like `datasets.interleave_datasets` over the `Iterable` datasets as well.

Further important thing to note is in case of HF, if we use hugging face the `map` functionality which we use to implement data handling is handled in a lazy fashion meaning we don't need to handle the data handlers in a different way for streaming data. [More Information on HF Page.](https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable#eager-data-processing-and-lazy-data-processing)

## Handling data collators.

Data collators specifically for TRL use cases like chat based interactions which apply chat templates and proper attention masking on the tokenized data like in the case of `DataCollatorForCompletionOnlyLM` handle a specific functionality on the data.

In this design our approach is to pass data collators from hugging face API directly to SFTTrainer.

In the current code path, collators are collected by `get_data_collator` functionality and passed to `SFTTrainer`. We can retain the same functionality and keep the design simpler.

The job of the data pre processor is to provide a single interface over multiple datasets in the config while keeping a collator like this means we will keep the collator same across all datasets but keeps the design simpler.

## Handling Multi Modal Data.

HF does provide support for handling [image datasets](https://huggingface.co/docs/datasets/en/image_process) and [audio datasets](https://huggingface.co/docs/datasets/en/audio_load) which can be utilized by us in our HF datapreprocessor.

The functionality listed by HF in implementing the use of image and audio datasets is `map` based functions to perform resize, encoding and other such operations on the dataset (see the link above).

This means the image and audio multi modal datasets will be compatible with our data handler routines. Once we implement the data handler routine processing, we will allow users to train with multi modal datasets too.

# Implementing stages.
1. Stage 1: 
    * Refactoring the code in `fms-hf-tuning` into the abstract data class and adding support for preliminary data handling routines.
        This will automatically enable support for multi modal data which is our priority.
    Note at this stage it might be wise to have two side by side implementations, i.e. not deleting the existing implementation.
1. State 2:
    * Implementing `streaming` data or `iterable` dataset support for the HF datapreprocessor implementation.
    * Data handling support for streaming data
1. State 3:
    * Identify and add any other required predefined data handlers.
    * Phase out the old implementation in support of the new one.

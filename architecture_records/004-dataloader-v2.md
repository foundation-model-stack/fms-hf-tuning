# Data Loader Design For FMS-HF-Tuning

**Deciders(s)**: Sukriti Sharma (sukriti.sharma4@ibm.com), Will Johnson (Will.Johnson@ibm.com) , Abhishek Maurya (maurya.abhishek@ibm.com), Dushyant Behl (dushyantbehl@in.ibm.com), Ashok Pon Kumar (ashokponkumar@in.ibm.com)

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

The reason for motivating dataloader design for fms-hf-tuning is to have a unified interface which supports many type of data formats, streaming and non streaming data, weight based data mixing and many others. Full list we focus on is below - 

1. Data Loading -

    1. Different formats of data → Arrow, Parquet etc.

    1. Different modalities of data -> Images, Audio etc.

    1. Streaming data.

    1. Data Replay

    1. Resume with different number of GPUs

    1. Async data loading

1. Data Preprocessing - 

    1. Custom Attn Masking

    1. Tool Usage

1. Data Mixing - 

    1. Static weights based mixing

### Motivation

### User Benefit

## Decision

### Alternatives Considered

## Consequences

### Advantages

### Impact on performance

## Detailed Design

The input spec which user specifies on how to pass information to such data loader is this

```
dataloader:
    type: default(hfdataloader)/stateful(fms-fsdp implementation)
    streaming: true
datasets:
  - name: dataset1
    sampling:
      ratio: 0.3
    splitter_arguments:
      test_size: 0
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
  - name: dataset1
    sampling:
      ratio: 0.4
    data_paths:
      - /data/stackoverflow-kubectl_posts
      - /data/stackoverflow-kubernetes_posts
      - /data/stackoverflow-openshift_posts
    splitter_arguments:
      test_size: 1
    data_handlers:
      - name: render_template
        arguments:
          remove_columns: all
          batched: false
          fn_kwargs:
            jinja_template: "{<jinja-template>}"
  - name: dataset2
    sampling:
      ratio: 0.3
    data_handlers:
      - name: apply_tokenizer_chat_template
        arguments:
          remove_columns: all
          batched: false
    data_files:
      - /data/stackoverflow-kubectl_posts.jsonl
      - /data/stackoverflow-kubernetes_posts.jsonl
  - name: dataset2
    sampling:
      ratio: 0.3
    predefined_handlers:
      name: apply_chat_template # pretraining <tokenize and merge everything>
      fn_kwargs:
        jinja_template: "<>"
    data_files:
      - /data/stackoverflow-kubectl_posts.jsonl
      - /data/stackoverflow-kubernetes_posts.jsonl
```

Please note the various option differences in different type of datasets defined in the code. 
This is just a sample and is presented in YAML but the system is to be designed to take this input spec and parse it from JSON as well.

The config representation in code is specified below.

```

@dataclass
class DataHandlerConfig:
    name: str
    arguments: Optional[Dict]

@dataclass
class DatasetConfig:
    name: str
    sampling: Optional[Dict] = None
    splitter_arguments: Optional[Dict] = None
    data_paths: List[str]
    data_handlers: List[DataHandlerConfig] = None

@dataclass
class DataLoaderConfig:
    streaming: Optional[bool] = None

@dataclass
class DataConfig:
    dataloader: Optional[DataLoaderConfig]
    datasets: List[DatasetConfig]
```


The proposed design to implement support for this spec is follows,

```
class DataLoader(ABC):

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

At the top level we propose to have this `class Dataloader` which is an abstract class 
and requires functions to process the data config proposed above.

We also propose a full length config verification code which preceeds the call to function 
`Dataloader.process_data_config` as the function expects a `DataConfig` object.

The data loader needs to support custom data handlers which are provided by users of the library
or even predefined handlers which need to be registered with the data loader class using the 
call `DataLoader.register_data_handler`.

The reason to have a top level dataloader class is to ensure we have separate classes for
hugging face and stateful data loader (fms-fsdp implementation).
This is also` selected from the argument in the config `dataloader.type`

## How are handlers provided and registered - 

Data handlers are python callables which can be called on single/few samples of data and can perform
things like applying chat template, tokenising the data, applying tools like jinja template or even
things like encoding or decoding multi modal formats like images/audio for processing by the model.

The abstract data loader class provides a way to register datahandler against a `name` which is a string.
The data handler config `DataHandlerConfig` taken by `execute_data_handlers` represents a DAG of data handling
routines which are to be executed on the data. 

For standard HF API you can think of these as the HF Processing routines. Which could be Map/Filter/Select operations
We implement most of the routines as map and because of this even the tokenisation of data which is done today
in fms-hf-tuning via `tuning/utils/preprocessing_utils.py::get_preprocessed_dataset` can be retained as a data 
handler which performs tokenization.

The implementation is flexible enough for users to specify their own implementation of data handling routines
which can be called by the data loader as part of its execution.

To this end, one way to design is we can provide the users and API on like the one shown in the `Dataloader` class
which they can utilise to register custom data handlers, in this case however the user needs to use `fms-hf-tuning` as
a module but not via the implementation of its `main` functionality.

Please note that our implementation needs to support certain predefined built-in handlers like `apply_chat_template`
or `tokenize` which user can request just by a name.

For example see this implementation - https://github.ibm.com/ai4code-wisdom/platform/blob/main/modelops/modelops/train.py#L251

# Implementation of HuggingFace Data Loader.

HF Dataloader or a dataloader which used Hugging Face API is implemented as an instance of the `Dataloader` class.

When the dataloader goes through each `DataSetConfig`

The HF dataloader implements functionality to process different type of files via its `load_dataset` factory.
If not supported automatically via this, we can look to extend the factory to use an other type of interest via
`Dataset.from_generator(<generator>)` functionality.

The reason to make use of HF factory is to keep our code simple and free of duplication with preexisting library code.

This also means that any implementation like `get_json_object` which load `json(l)` and then return a custom json dict
can be implemented as data handlers over basic json dataloader from HF.

### Splitting and Interleaving datasets

Other argument such as `splitter_arguments` can be passed to HF [`datasets.test_train_split`](https://huggingface.co/docs/datasets/v3.0.1/en/package_reference/main_classes#datasets.Dataset.train_test_split) to create a test/train split of
the dataset as requested by the user.

In case of multiple datasets the user can request how the datasets are to be interleaved.
The probabilies specified by users in the config `sampling.ratio` can be collected from individual datasets and passed to
[`datasets.interleave_datasets`](https://huggingface.co/docs/datasets/v3.0.1/en/package_reference/main_classes#datasets.interleave_datasets).

### Streaming datasets

In case of Hugging Face Dataloader the `streaming` argument can be handled by using `IterableDatasets` instead of standard `Datasets`.
HF provides same APIs like `datasets.interleave_datasets` over the `Iterable` datasets as well.

Further important thing to note is in case of HF, if we use hugging face the `map` functionality which we use to implement data handling
is handled in a lazy fashion meaning we don't need to handle the data handlers in a different way for streaming data. [More Information on HF Page.](https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable#eager-data-processing-and-lazy-data-processing)

## Handling data collators.

Data collators specifically for TRL use cases like chat based interactions which apply chat templates and proper attention masking on the
tokenized data like in the case of `DataCollatorForCompletionOnlyLM` handle a specific functionality on the data. In this design we consider two approaces for data collators.

1. Implementing data collators as map.
    We explore this approach as all the processing of data by us is done in data handlers so to implement collators as data handlers 
    makes sense but is a hard problem to solve. Data collation happens at the very last stage of batch creation and is done in an async
    fashion because event the HF data collators like `DataCollatorForCompletionOnlyLM` or [`DataCollatorForSeq2Seq`](https://github.com/huggingface/transformers/blob/3f06f95ebe617b192251ef756518690f5bc7ff76/src/transformers/data/data_collator.py#L543) are passed to torch data loader by the HF APIs and is executed by the torch Dataloader.

    So implementing data collators as map means we need to disable [torch data collation and batching](https://pytorch.org/docs/stable/data.html#disable-automatic-batching) and perform it in our code, which we do not intend to experiment with due to its complexity.

1. Passing collators directly to SFTTrainer.

    In the code collators are collected by `get_data_collator` functionality and passed to `SFTTrainer`. We can retain the same functionality
    and keep the design simpler. The Job of the data loader is to provide a single interface over multiple datasets in the config while keeping a collator like this means we will keep the collator same across all datasets but keeps the design simpler.

## Simplification of code and user configuration

The flexibility provided by this design is that it simplifies the configuration requirement for various use cases.
If chat template and chat style data is requested users can specify chat specific data handlers and not specify all configurations which are
not required.
This can also simplify configuration handling in the code. TODO: give example

## Handling Multi Modal Data.

HF does provide support for handling [image datasets](https://huggingface.co/docs/datasets/en/image_process) and [audio datasets](https://huggingface.co/docs/datasets/en/audio_load) which can be utilised by us in our HF dataloader.

The functionality listed by HF in implementing the use of image and audio datasets is `map` based functions to perform resize, encoding and other such operations on the dataset (see the link above).

This means the image and audio multi modal datasets will be compatible with our data handler routines. Once we have the data handling routines set we will allow users to train with multi modal datasets too.

# Stateful Dataloader implementation.

Stateful dataloader refers to the `fms-fsdp` implementation of the dataloader by our colleagues Davis and Linsong.

Supporting stateful dataloader will mean refactoring the fms-fsdp implementation into our abstract `Dataloader` class.

In brief, things to consider here will be,
1. Data handler support needs to be added to the stateful data loader as we want lazy execution of handlers (as and when data is loaded).
1. Data collation needs to be though through as the dataset currently is not implemented to handle the same. 

TODO: Add more details on how stateful data loader can be integrated.

# Implementing stages.

1. Stage 1: 
    * Refactoring the code in `fms-hf-tuning` into the abstract data class and adding support for preliminery data handling routines.
        This will automatically enable support for multi modal data which is our priority.
    Note at this stage it might be wise to have two side by side implementations, i.e. not deleting the existing implementation.
1. State 2:
    * Implementing `streaming` data or `iterable` data loader support for the HF dataloader implementation.
    * Data handling support for streaming data
1. State 3:
    * Phase out the old implementation in support of the new one.
    * Identify and add required predefined data handlers.
1. State 4:
    * Refactoring stateful dataloader into `fms-hf-tuning` and implementing support for stateful data loading.

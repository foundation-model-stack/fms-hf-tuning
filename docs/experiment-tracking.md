# Experiment Tracker

Experiment tracking is an optional feature of this repo. We have introduced experiment tracking to help in systematically recording hyperparameters, model configurations, and results from each experiment automatically and with the help of third party trackers like [Aimstack](https://aimstack.io).

Tracking can be enabled by passing a config in the [Training Arguments](https://github.com/foundation-model-stack/fms-hf-tuning/blob/a9b8ec8d1d50211873e63fa4641054f704be8712/tuning/config/configs.py#L131)
with the name of the enabled trackers passed as a list.

```
from tuning import sft_trainer

training_args = TrainingArguments(
    ...,
    trackers = ["aim", "file_logger"]
)

sft_trainer.train(train_args=training_args,...)
```

For each of the requested trackers the code expects you to pass a config to the `sft_trainer.train` function which can be specified through `tracker_conifgs` argument [here](https://github.com/foundation-model-stack/fms-hf-tuning/blob/a9b8ec8d1d50211873e63fa4641054f704be8712/tuning/sft_trainer.py#L78) details of which are present below.  




## Tracker Configurations

## File Logging Tracker

[File Logger](../tuning/trackers/filelogging_tracker.py) is an inbuilt tracker which can be used to dump loss at every log interval to a file.  

Currently `File Logger` is enabled by default and will dump loss at every log interval of a training to a default file path specified [here](../tuning/config/tracker_configs.py) inside the output folder passed during training.  

To override the location of file logger please pass an instance of the [FileLoggingTrackerConfig](../tuning/config/tracker_configs.py) to `tracker_configs` argument.  

```
from tuning import sft_trainer
from tuning.config.tracker_configs import FileLoggingTrackerConfig, TrackerConfigFactory

training_args = TrainingArguments(
    ...,
    trackers = ["file_logger"]
)


logs_file = "new_train_logs.jsonl"

tracker_configs = TrackerConfigFactory(
    file_logger_config=FileLoggingTrackerConfig(
        training_logs_filename=logs_file
        )
    )

sft_trainer.train(train_args=training_args, tracker_configs=tracker_configs, ...)
```

Currently File Logging tacker supports only one argument and this file will be placed inside the `train_args.output` folder.

## Aimstack Tracker

To enable [Aim](https://aimstack.io) users need to pass `"aim"` as the requested tracker as part of the [training argument](https://github.com/foundation-model-stack/fms-hf-tuning/blob/a9b8ec8d1d50211873e63fa4641054f704be8712/tuning/config/configs.py#L131).


When using Aimstack, users need to specify additional arguments which specify where the Aimstack database is present and what experiment name to use
for tracking the training.

Aimstack supports either a local (`filesystem path`) based db location or a remote (`aim_server:port`) based database location.  

See Aim [documentation](https://aimstack.readthedocs.io/en/latest/using/remote_tracking.html) for more details.

After [initialising a repo](https://aimstack.readthedocs.io/en/latest/quick_start/setup.html#initializing-aim-repository), users can specify the location of the
repo either local or remote.

For a local aim database where `aim_repo` should point to the path of where the initialized Aimstack repo is present,

```
from tuning import sft_trainer
from tuning.config.tracker_configs import AimConfig, TrackerConfigFactory

training_args = TrainingArguments(
    ...,
    trackers = ["aim"],
)

tracker_configs = TrackerConfigFactory(
    aim_config=AimConfig(
        experiment="experiment-name",
        aim_repo=<path_to_the_repo>
        )
    )

sft_trainer.train(train_args=training_args, tracker_configs=tracker_configs,....)
```

 or, for a remote server where aimstack database is running at `aim://aim_remote_server_ip:aim_remote_server_port`

```
from tuning import sft_trainer
from tuning.config.tracker_configs import AimConfig, TrackerConfigFactory

training_args = TrainingArguments(
    ...,
    trackers = ["aim"],
)

tracker_configs = TrackerConfigFactory(
    aim_config=AimConfig(
        experiment="experiment-name",
        aim_remote_server_ip=<server_url>,
        aim_remote_server_port=<server_port>
        )
    )

sft_trainer.train(train_args=training_args, tracker_configs=tracker_configs,....)
```

The code expects either the `local` or `remote` repo to be specified and will result in a `ValueError` otherwise.
See [AimConfig](https://github.com/foundation-model-stack/fms-hf-tuning/blob/a9b8ec8d1d50211873e63fa4641054f704be8712/tuning/config/tracker_configs.py#L25) for more details.

## MLflow Tracker

To enable [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) users need to pass `"mlflow"` as the requested tracker as part of the [training argument](https://github.com/foundation-model-stack/fms-hf-tuning/blob/a9b8ec8d1d50211873e63fa4641054f704be8712/tuning/config/configs.py#L131).


When using MLflow, users need to specify additional arguments which specify [mlflow tracking uri](https://mlflow.org/docs/latest/tracking.html#common-setups) location where either a [mlflow supported database](https://mlflow.org/docs/latest/tracking/backend-stores.html#supported-store-types) or [mlflow remote tracking server](https://mlflow.org/docs/latest/tracking/server.html) is running.

Example
```
from tuning import sft_trainer
from tuning.config.tracker_configs import MLflowConfig, TrackerConfigFactory

training_args = TrainingArguments(
    ...,
    trackers = ["mlflow"],
)

tracker_configs = TrackerConfigFactory(
    mlflow_config=MLflowConfig(
        mlflow_experiment="experiment-name",
        mlflow_tracking_uri=<tracking uri>
        )
    )

sft_trainer.train(train_args=training_args, tracker_configs=tracker_configs,....)
```

The code expects a valid uri to be specified and will result in a `ValueError` otherwise.

## Running the code via command line `tuning/sft_trainer::main` function

If running the code via main function of [sft_trainer.py](../tuning/sft_trainer.py) the arguments to enable and customise trackers can be passed via commandline.

To enable tracking please pass

```
--tracker <aim/file_logger/mlflow>
```

To further customise tracking you can specify additional arguments needed by the tracker like (example shows aim follow similarly for mlflow)

```
--tracker aim --aim_repo <path-to-aimrepo> --experiment <experiment-name>
```
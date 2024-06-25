# Trainer controller

Trainer controller is a framework for controlling the trainer loop using user-defined rules and metrics.

### Motivation

This frameworks helps user define rules to capture scenarios like criteria for stopping an ongoing training (E.g validation loss reaching a certain target, validation loss increasing with epoch, training loss values for last 100 steps increasing etc).

### Usage
*Note: Evaluation loss and validation loss are the same.*
1. The trainer controller feature can be used and its behavior is controlled by a configuration file (we will illustrate the configuration file below) supplied by the user at the start of the training. Here is a sample of how the user can initiate a trainer controller for a training job, by specifying path to an existing configuration `loss.yaml` in the `./examples/trainercontroller_configs` directory using the flag `--trainer_controller_config_file`:
    ```shell
    python ./tuning/sft_trainer.py  \
    ...
    --trainer_controller_config_file "$EXAMPLE_CONFIGS/epoch-level-eval-loss-below-threshold.yaml" \
    ...
    ...
    ```

1. For this usage illustration, we could use the `epoch-level-eval-loss-below-threshold.yaml` in the `./examples/trainercontroller_configs` directory as shown below:
    ```yaml
    controller_metrics:
    - name: trainer_state
      class: TrainingState
    - name: evalmetric
      class: EvalMetrics
    controllers:
    - name: epoch_level_eval_loss_below_threshold
      triggers:
      - on_epoch_end
      rule: 'evalmetric["eval_loss"] < 2.25 and trainer_state["epoch"] > 2'
      operations:
      - hfcontrols.should_training_stop
    ```
    Here is a brief primer on the above configuration. More details could be found [here](./architecture_records/001-trainer-controller-framework.md).
    - *Description:* The above configuration stops the training when a **evaluation loss** decreases below 2.25 after two epochs.
    - *Metrics:* The configuration uses two metrics listed under `controller-metrics` section. One is named `evalmetric`, which uses an in-built metric class called `EvalMetrics` to expose evaluation loss and the other (`trainer_state`) uses `TrainingState` to expose the current epoch. These are referred to in the `rule` as shown above. There are other metrics also which could be used in place of `evalmetric` and . Here is a list of supported metric classes:
      - `Loss`: Exposes the **training loss** after every `on_log` event. See more on trainer events [here](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/callback#transformers.TrainerCallback).
      - `TrainerState`: This metric exposes the **trainer state** (more on trainer state can be found [here](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/callback#transformers.TrainerState)). [Here](tests/data/trainercontroller/loss_on_threshold_with_trainer_state.yaml) is an example metric which uses both the `TrainerState` and `Loss` metric.
      - `EvalMetrics`: This metric exposes all the evaluation metrics used in the training job (E.g evaluation/validation loss). [Here](tests/data/trainercontroller/exposed_metrics.yaml) is an example metric which uses both the `EvalMetrics`.
      - `HistoryBasedMetric`: This metric exposes a moving **window** of evaluation metrics and training loss. It is useful to create rules on a history of values (i.e. evaluation metrics and training loss). Following are some examples which illustrate how this metric could be used:
        - [epoch-level-eval-loss-patience.yaml](tests/data/trainercontroller/epoch-level-eval-loss-patience.yaml): This configuration performs a threshold test for evaluation loss with a **patience threshold** of 2. I.e suppose the evaluation loss lower threshold is 2, and patience threshold is 3, then the trainer controller will not take an action (E.g. stop training) when the rule becomes true (i.e. evaluation loss is lower than 2) for for three consecutive times.
        - [non-decreasing-training-loss.yaml](tests/data/trainercontroller/non-decreasing-training-loss.yaml): This configuration compares the first and last values of a window of training loss samples and determines if the training loss has increased or not. If there is an increase, the training is stopped.

        Let us assume use the below example to understand the usage:
        ```yaml
        controller_metrics:
        - name: history_window
            class: HistoryBasedMetric
            arguments:
            window_size: 2
        controllers:
        - name: epoch_level_eval_loss_patience
            triggers:
            - on_epoch_end
            rule: len(history_window["metrics"]) > 0 and history_window["metrics"]["eval_loss"][-1] > 2
            patience:
            patience_threshold: 2
            operations:
            - hfcontrols.should_training_stop
        ```
        In the above YAML, the name for `HistoryBasedMetric` used is `history_window`. Here is short primer on defining rules using the `HistoryBasedMetric`:
        1. Treat the `history_window` as a python dictionary. The structure of the data in this dictionary is:
            ```yaml
                {
                    "metrics": {
                                    "global_step": [...],
                                    "epoch": [...],
                                    "eval_loss": [...],
                                    "user_eval_metric_1": [...],
                                    "user_eval_metric_2": [...],
                                    ...
                                },
                    "training_loss": {
                                    "global_step": [...],
                                    "epoch": [...],
                                    "loss": [...],
                                }
                }
            ```
        1. To access the first value in window of evaluation metric `eval_loss`, here is the illustration `history_window["metrics"]["eval_loss"][0]`. In the above YAML, the last element is accessed as follows: `history_window["metrics"]["eval_loss"][-1]`.
        1. Similarly, the `history_window["metrics"]["global_step"][0]` is global_step at the time of generation of this evaluation metric and `history_window["metrics"]["epoch"][0]` is the corresponding epoch.
        1. Similar approach is followed to access training loss (i.e. `history_window["training_loss"]["loss"][0]` givest the first training loss).

    - *Trigger:* There is also a trigger event to decide when the `rule` needs to be evaluated. This event has to be one of the trainer events listed [here](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/callback#transformers.TrainerCallback).
    - *Rule:* The `rule` is a python statement which could use the metric name (e.g. `loss` in the above case) to define conditions which, when satisfied (it is a boolean condition and should evaluate to True to be satisfied) will trigger the operation(s) listed in `operations`.
    - *Operation:* The `operations` section lists the operations that could be performed when the `rule` is satisfied (i.e. condition becomes True). Currently, we support only one type of operation class `HFControls` (In this particular example, the class and corresponding operation name `hfcontrols` are not specified explicitly as they are considered default and can be omitted). The `HFControls` class supports all operations listed below. More on these operations can be found [here](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/callback#transformers.TrainerControl).
        - `hfcontrols.should_training_stop`: Stops the training.
        - `hfcontrols.should_epoch_stop`: Interrupts the current epoch.
        - `hfcontrols.should_save`: Saves the model at the current step.
        - `hfcontrols.should_evaluate`: Should the model be evaluated at current step.
        - `hfcontrols.should_log`: Should logging happen at current step.

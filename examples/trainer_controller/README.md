# Trainer controller

Trainer controller is a framework for controlling the trainer loop using user-defined rules and metrics.

### Motivation

- If there is a need for stopping an ongoing training when some stopping criteria is satisfied (E.g validation loss reaching a certain target, validation loss increasing with epoch, training loss values for last 100 steps increasing etc).
- There is a [EarlyStoppingCallback](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/trainer_callback.py#L543) in HuggingFace, but the granularity of stopping is only on `evaluate` events, and handles only compares instantaneous metric value to a threshold.
- Therefore, there is a need for a mechanism to capture the user-defined custom stopping criteria which could involve multiple metrics.
- In addition to user-defined stopping criteria, there could other types of control operations with respect to training (for instance, should the trainer perform saving, logging or evaluation operations or not, should we scale resources dynamically so that training could run faster and so on). Therefore, there is a need for general need to capture all these use-cases in a single framework. This PR attempts to provide such a framework.

### Usage

1. The trainer controller feature can be used and its behavior is controlled by a configuration file (we will illustrate the configuration file below) supplied by the user at the start of the training. Here is a sample of how the user can initiate a trainer controller for a training job, by specifying path to an existing configuration `loss.yaml` in the `./examples/trainercontroller_configs` directory using the flag `--trainer_controller_config_file`:
    ```shell
    # if you want to use one GPU on multi-gpu machine
    export CUDA_VISIBLE_DEVICES=0

    # MODEL_PATH=meta-llama/Llama-2-7b-hf # Huggingface model id or path to a checkpoint
    # TRAIN_DATA_PATH=twitter_complaints.json # Path to the dataset
                    # contains data in form of [{"input": text , "output": text}]
    # VALIDATION_DATA_PATH=/path/to/validation/dataset
    # OUTPUT_PATH=out # Path to the output folder where the checkpoints are saved

    # EXAMPLE_CONFIGS="./examples/trainercontroller_configs"

    python ./tuning/sft_trainer.py  \
    --model_name_or_path $MODEL_PATH  \
    --tokenizer_name_or_path $MODEL_PATH \
    --training_data_path $TRAIN_DATA_PATH  \
    --output_dir $OUTPUT_PATH  \
    --validation_data_path $VALIDATION_DATA_PATH \
    --num_train_epochs 10  \
    --per_device_train_batch_size 4  \
    --per_device_eval_batch_size 4  \
    --gradient_accumulation_steps 4  \
    --evaluation_strategy "epoch"  \
    --save_strategy "epoch"  \
    --learning_rate 1e-5  \
    --weight_decay 0.  \
    --warmup_ratio 0.03  \
    --lr_scheduler_type "cosine"  \
    --logging_steps 5  \
    --logging_strategy "steps" \
    --include_tokens_per_second  \
    --packing False  \
    --metric_for_best_model "loss" \
    --load_best_model_at_end True \
    --use_flash_attn False \
    --trainer_controller_config_file $EXAMPLE_CONFIGS"/loss.yaml" \
    --response_template "\n### Response:"  \
    --dataset_text_field "output
    ```

1. For this usage illustration, we could use the `loss.yaml` in the `./examples/trainercontroller_configs` directory as shown below:
    ```yaml
    controller-metrics:
    - name: loss
        class: Loss
    controllers:
    - name: loss-controller
        triggers:
        - on_log
        rule: loss < 1.0
        operations:
        - hfcontrols.should_training_stop
    ```
    Here is a brief primer on the above configuration. More details could be found [here](./architecture_records/001-trainer-controller-framework.md).
    - *Description:* The above configuration stops the training when a **training loss** decreases below 1.0.
    - *Metrics:* The configuration uses a metric named `loss` listed under `controller-metrics` section, which uses an in-built metric called `Loss`. This is referred to in the `rule` as shown above. There are other metrics also which could be used in place of `loss`. Here is a list of supported metric classes:
      - `Loss`: Exposes the **training loss** after every `on_log` event. See more on trainer events [here](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/callback#transformers.TrainerCallback).
      - `TrainerState`: This metric exposes the **trainer state** (more on trainer state can be found [here](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/callback#transformers.TrainerState)). [Here](tests/data/trainercontroller/loss_on_threshold_with_trainer_state.yaml) is an example metric which uses both the `TrainerState` and `Loss` metric.
      - `EvalMetrics`: This metric exposes all the evaluation metrics used in the training job (E.g evaluation/validation loss). [Here](tests/data/trainercontroller/exposed_metrics.yaml) is an example metric which uses both the `EvalMetrics`.
    - *Trigger:* There is also a trigger event to decide when the `rule` needs to be evaluated. This event has to be one of the trainer events listed [here](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/callback#transformers.TrainerCallback).
    - *Rule:* The `rule` is a python statement which could use the metric name (e.g. `loss` in the above case) to define conditions which, when satisfied (it is a boolean condition and should evaluate to True to be satisfied) will trigger the operation(s) listed in `operations`.
    - *Operation:* The `operations` section lists the operations that could be performed when the `rule` is satisfied (i.e. condition becomes True). Currently, we support only one type of operation class `HFControls` (In this particular example, the class and corresponding operation name `hfcontrols` are not specified explicitly as they are considered default and can be omitted). The `HFControls` class supports all operations listed below. More on these operations can be found [here](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/callback#transformers.TrainerControl).
        - `hfcontrols.should_training_stop`: Stops the training.
        - `hfcontrols.should_epoch_stop`: Interrupts the current epoch.
        - `hfcontrols.should_save`: Saves the model at the current step.
        - `hfcontrols.should_evaluate`: Should the model be evaluated at current step.
        - `hfcontrols.should_log`: Should logging happen at current step.

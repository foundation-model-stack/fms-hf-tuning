# How-To
To use one of these files with the trainer, execute the `sft_trainer.py` with the following option: 
```
--trainer_controller_config_file "examples/trainer-controller-configs/<file-name>"
```

# Note on trainer controller configuration examples
- `trainercontroller_config_step.yaml`: Defines a trainer controller, which computes loss at every step, and if the loss consistently increases for three steps, then the training is stopped.
- `trainercontroller_config_epoch.yaml`: Defines an epoch-level trainer controller, which computes loss at every epoch. The rule applied here is to compare the current epoch loss with the previous epoch loss, and if the current epoch loss turns out to be more, then the training is stopped.
- `trainercontroller_config_epoch_threshold.yaml`: Defines a trainer controller similar to previous case, but also adds a threshold constraint.
- `trainercontroller_config_evaluate.yaml`: Defines a trainer controller which behaves similar to the `EarlyStoppingCallback` from hugging face which can be found [here](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/trainer_callback.py#L543).
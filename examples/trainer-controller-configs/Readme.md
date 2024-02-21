# How-To
To use one of these files with the trainer, execute the `sft_trainer.py` with the following option: 
```
--training_control_config_file "examples/training-control-configs/<file-name>"
```

# Note on training control configuration examples
- `trainercontroller_config_step_v0.3.yaml`: Defines a training controller which computes loss at every step and loss consistently increases for three steps, then the training is stopped.
- `trainercontroller_config_epoch_v0.3.yaml`: Defines a epoch level training controller which computes loss at every epoch. The rule applied here is to compare a current epoch loss with previous epoch loss and it turns out to be more, then training is stopped.
- `trainercontroller_config_epoch_threshold_v0.3.yaml`: Defines a training controller similar to previous case, but also adds a threshold constraint.
- `trainercontroller_config_evaluate_v0.3.yaml`: Defines a training controller which behaves similar to the `EarlyStoppingCallback` from hugging face which can be found [here](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/trainer_callback.py#L543).
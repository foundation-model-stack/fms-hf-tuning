# Third Party
import pytest
import math

# Local
import tuning.trainercontroller as tc
import tuning.config.configs as config
from transformers import TrainerControl, TrainerState, IntervalStrategy

def test_step_loss():
    test_data = [{'loss': 2.0, 'eval_loss': 2.0, 'epoch': 0.1}, \
                {'loss': 2.1, 'eval_loss': 2.1, 'epoch': 0.25}, \
                {'loss': 2.3, 'eval_loss': 2.3, 'epoch': 0.5}]
    outcomes = [False, False, True]
    training_args = config.TrainingArguments(output_dir='')
    trainer_controller_args = config.TrainerControllerArguments()
    training_args.logging_strategy = IntervalStrategy.STEPS
    training_args.logging_steps = 1
    trainer_controller_args.trainer_controller_config_file = 'examples/trainer-controller-configs/trainercontroller_config_step.yaml'
    tc_callback = tc.TrainerControllerCallback(trainer_controller_args, training_args)
    control = TrainerControl()
    control.should_training_stop = False
    state = TrainerState()
    state.log_history = []
    for i in range(len(test_data)):
        state.log_history.append(test_data[i])
        control = tc_callback.on_step_end(training_args, state, control)
        assert control.should_training_stop == outcomes[i]

def test_epoch_loss():
    test_data = [{'loss': 2.0, 'eval_loss': 2.0, 'epoch': 0.1}, \
                {'loss': 2.1, 'eval_loss': 2.1, 'epoch': 0.25}, \
                {'loss': 2.3, 'eval_loss': 2.3, 'epoch': 0.5}, \
                {'loss': 2.35, 'eval_loss': 2.35, 'epoch': 0.75}, \
                {'loss': 2.4, 'eval_loss': 2.35, 'epoch': 1.0}, \
                {'loss': 2.45, 'eval_loss': 2.4, 'epoch': 1.25}, \
                {'loss': 2.5, 'eval_loss': 2.45, 'epoch': 1.5}, \
                {'loss': 2.55, 'eval_loss': 2.5, 'epoch': 1.75}, \
                {'loss': 2.6, 'eval_loss': 2.55, 'epoch': 2.0}]
    outcomes = [False, False, False, False, False, False, False, False, True]
    training_args = config.TrainingArguments(output_dir='')
    trainer_controller_args = config.TrainerControllerArguments()
    training_args.logging_strategy = IntervalStrategy.STEPS
    training_args.logging_steps = 1
    trainer_controller_args.trainer_controller_config_file = 'examples/trainer-controller-configs/trainercontroller_config_epoch.yaml'
    tc_callback = tc.TrainerControllerCallback(trainer_controller_args, training_args)
    control = TrainerControl()
    control.should_training_stop = False
    state = TrainerState()
    state.log_history = []
    for i in range(len(test_data)):
        state.log_history.append(test_data[i])
        if (math.ceil(test_data[i]['epoch']) - test_data[i]['epoch']) > 0:
            continue
        control = tc_callback.on_epoch_end(training_args, state, control)
        assert control.should_training_stop == outcomes[i]

def test_epoch_threshold_loss():
    test_data = [{'loss': 2.1, 'eval_loss': 2.0, 'epoch': 0.1}, \
                {'loss': 2.1, 'eval_loss': 2.1, 'epoch': 0.25}, \
                {'loss': 2.05, 'eval_loss': 2.3, 'epoch': 0.5}, \
                {'loss': 2.05, 'eval_loss': 2.35, 'epoch': 0.75}, \
                {'loss': 2.02, 'eval_loss': 2.35, 'epoch': 1.0}, \
                {'loss': 2.03, 'eval_loss': 2.4, 'epoch': 1.25}, \
                {'loss': 2.01, 'eval_loss': 2.45, 'epoch': 1.5}, \
                {'loss': 2.0, 'eval_loss': 2.5, 'epoch': 1.75}, \
                {'loss': 2.09, 'eval_loss': 2.55, 'epoch': 2.0}]
    outcomes = [False, False, False, False, False, False, False, False, True]
    training_args = config.TrainingArguments(output_dir='')
    trainer_controller_args = config.TrainerControllerArguments()
    training_args.logging_strategy = IntervalStrategy.STEPS
    training_args.logging_steps = 1
    trainer_controller_args.trainer_controller_config_file = 'examples/trainer-controller-configs/trainercontroller_config_epoch_threshold.yaml'
    tc_callback = tc.TrainerControllerCallback(trainer_controller_args, training_args)
    control = TrainerControl()
    control.should_training_stop = False
    state = TrainerState()
    state.log_history = []
    for i in range(len(test_data)):
        state.log_history.append(test_data[i])
        if (math.ceil(test_data[i]['epoch']) - test_data[i]['epoch']) > 0:
            continue
        control = tc_callback.on_epoch_end(training_args, state, control)
        assert control.should_training_stop == outcomes[i]
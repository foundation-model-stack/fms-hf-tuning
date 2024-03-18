# Copyright The IBM Tuning Team
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

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Local
import tuning.trainercontroller as tc
import tuning.config.configs as config
import tests.data as td

# Third Party
from transformers import TrainerControl, TrainerState, IntervalStrategy

def test_loss_on_threshold():
    """Tests the loss threshold example in `examples/trainer-controller-configs/loss.yaml`
    """
    # Test data to mimic the fields of trainer loop log-lines
    test_data = [{'loss': 2.0, 'epoch': 0.1}, \
                {'loss': 2.1, 'epoch': 0.25}, \
                {'loss': 1.3, 'epoch': 0.5}, \
                {'loss': 0.9, 'epoch': 0.6}]
    # Trainer arguments relevant to the test
    training_args = config.TrainingArguments(
                        output_dir='',
                        logging_strategy=IntervalStrategy.STEPS,
                        logging_steps=1,
                        )
    tc_callback = tc.TrainerControllerCallback(td.TRAINER_CONFIG_TEST_YAML)
    control = TrainerControl(should_training_stop = False)
    state = TrainerState(log_history = [])
    # Trigger on_init_end to perform registration of handlers to events 
    tc_callback.on_init_end(args=training_args, state=state, control=control)
    state.log_history=test_data
    # Trigger rule and test the condition
    tc_callback.on_log(args=training_args, state=state, control=control)
    assert control.should_training_stop == True

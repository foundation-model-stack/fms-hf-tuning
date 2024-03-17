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
from transformers import TrainerControl, TrainerState, IntervalStrategy

def test_step_loss_on_threshold():
    test_data = [{'loss': 2.0, 'epoch': 0.1}, \
                {'loss': 2.1, 'epoch': 0.25}, \
                {'loss': 1.3, 'epoch': 0.5}, \
                {'loss': 0.9, 'epoch': 0.6}]
    training_args = config.TrainingArguments(
                        output_dir='',
                        logging_strategy=IntervalStrategy.STEPS,
                        logging_steps=1,
                        )
    tc_callback = tc.TrainerControllerCallback('examples/trainer-controller-configs/loss.yaml')
    control = TrainerControl(should_training_stop = False)
    state = TrainerState(log_history = [])
    tc_callback.on_init_end(args=training_args)
    state.log_history=test_data
    tc_callback.on_step_end(args=training_args, state=state, control=control)
    assert control.should_training_stop == True

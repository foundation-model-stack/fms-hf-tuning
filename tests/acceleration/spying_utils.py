# Copyright The FMS HF Tuning Authors
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

# utility for spying into a framework plugin
def create_mock_plugin_class(plugin_cls):
    "Create a mocked acceleration framework class that can be used to spy"

    # mocked plugin class
    class MockPlugin(plugin_cls):

        # counters used for spying
        model_loader_calls: int
        augmentation_calls: int
        get_ready_for_train_calls: int

        @classmethod
        def reset_calls(cls):
            # reset the counters
            cls.model_loader_calls = (
                cls.augmentation_calls
            ) = cls.get_ready_for_train_calls = 0

        def model_loader(self, *args, **kwargs):
            MockPlugin.model_loader_calls += 1
            return super().model_loader(*args, **kwargs)

        def augmentation(
            self,
            *args,
            **kwargs,
        ):
            MockPlugin.augmentation_calls += 1
            return super().augmentation(*args, **kwargs)

        def get_callbacks_and_ready_for_train(self, *args, **kwargs):
            MockPlugin.get_ready_for_train_calls += 1
            return super().get_callbacks_and_ready_for_train(*args, **kwargs)

    return MockPlugin

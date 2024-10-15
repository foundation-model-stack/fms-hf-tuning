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


def create_mock_plugin_class_and_spy(class_name, plugin_cls):
    "helper function to create plugin class"

    spy = {
        "model_loader_calls": 0,
        "augmentation_calls": 0,
        "get_ready_for_train_calls": 0,
    }

    def model_loader(self, *args, **kwargs):
        spy["model_loader_calls"] += 1
        return plugin_cls.model_loader(self, *args, **kwargs)

    def augmentation(
        self,
        *args,
        **kwargs,
    ):
        spy["augmentation_calls"] += 1
        return plugin_cls.augmentation(self, *args, **kwargs)

    def get_callbacks_and_ready_for_train(self, *args, **kwargs):
        spy["get_ready_for_train_calls"] += 1
        return plugin_cls.get_callbacks_and_ready_for_train(self, *args, **kwargs)

    attributes = {
        "model_loader": model_loader,
        "augmentation": augmentation,
        "get_callbacks_and_ready_for_train": get_callbacks_and_ready_for_train,
    }

    return type(class_name, (plugin_cls,), attributes), spy

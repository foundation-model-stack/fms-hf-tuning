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

"""Shared pytest fixtures.

The imports inside these fixtures are intentionally function-local: they must not
run at collection time (peft / fms_acceleration pull in heavy optional stacks, and
`fms_acceleration` may not be installed at all), so the module-level pylint check
is disabled for them here rather than at each call site.
"""

# pylint: disable=import-outside-toplevel

# Third Party
import pytest


@pytest.fixture(autouse=True)
def restore_peft_type_mappings():
    """Undo global mutations of peft's PeftType -> class registries.

    `fms_acceleration_peft`'s GPTQ support installs `GPTQLoraConfig` /
    `GPTQLoraModel` into peft's process-wide `PeftType.LORA` slots and never
    puts the originals back (it re-applies them in its `finally` branch). Once
    any test exercises the AutoGPTQ augmentation path, every later LoRA test in
    the same process builds a `GPTQLoraModel`, whose `_create_new_module`
    returns None for an unquantized `torch.nn.Linear` -- surfacing as
    `AttributeError: 'NoneType' object has no attribute 'named_modules'` in a
    test that has nothing to do with GPTQ.

    Snapshot the entries before each test and restore them afterwards so test
    ordering cannot change results.
    """
    # Third Party
    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

    try:
        # peft >= 0.19
        # Third Party
        from peft.peft_model import PEFT_TYPE_TO_TUNER_MAPPING as TUNER_MAPPING
    except ImportError:
        # Third Party
        from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING as TUNER_MAPPING

    before_config = dict(PEFT_TYPE_TO_CONFIG_MAPPING)
    before_tuner = dict(TUNER_MAPPING)

    yield

    for mapping, snapshot in (
        (PEFT_TYPE_TO_CONFIG_MAPPING, before_config),
        (TUNER_MAPPING, before_tuner),
    ):
        mapping.clear()
        mapping.update(snapshot)


@pytest.fixture(autouse=True)
def restore_plugin_registrations():
    """Undo global mutations of fms_acceleration's plugin registry.

    `fms_acceleration.utils.test_utils.build_framework_and_maybe_instantiate`
    is a generator context manager that clears `PLUGIN_REGISTRATIONS`, registers
    the stub plugins a test asks for, and only puts the real ones back in code
    that runs *after* its `yield` -- it has no `try/finally`. So any test whose
    body raises leaves the registry holding just those stubs for the remainder
    of the process, and every later test that relies on a genuinely installed
    plugin (e.g. ODM) then fails with `ValueError: No plugins could be
    configured`, even though it passes in isolation.

    Snapshot the registry before each test and restore it afterwards.
    """
    try:
        # Third Party
        from fms_acceleration.framework_plugin import PLUGIN_REGISTRATIONS
    except ImportError:
        # fms_acceleration is optional; nothing to protect.
        yield
        return

    before = list(PLUGIN_REGISTRATIONS)

    yield

    PLUGIN_REGISTRATIONS.clear()
    PLUGIN_REGISTRATIONS.extend(before)


@pytest.fixture(autouse=True)
def restore_hf_trainer_patches():
    """Undo `fms_acceleration_odm`'s permanent monkey-patching of HF `Trainer`.

    `fms_acceleration_odm.patch.patch_hf_trainer_evaluate()` assigns
    `_evaluate`, `_get_dataloader` and `get_train_dataloader` straight onto the
    `transformers.Trainer` class (and re-points `transformers.trainer.Trainer`),
    with no save/restore. Its `get_train_dataloader` unconditionally reads
    `self.model.resume_from_checkpoint`, an attribute only the ODM plugin's
    `augmentation()` puts on the model.

    So once any ODM test runs, every later test that trains a plain model fails
    with `AttributeError: 'LlamaForCausalLM' object has no attribute
    'resume_from_checkpoint'`. Snapshot the class attributes before each test and
    put them back afterwards.
    """
    # Third Party
    from transformers import Trainer

    patched_names = ("_evaluate", "_get_dataloader", "get_train_dataloader")
    # Record whether each name was defined on Trainer itself (vs inherited/absent),
    # so restoring cannot accidentally create or delete the wrong thing.
    before = {n: Trainer.__dict__.get(n, None) for n in patched_names}

    yield

    for name, original in before.items():
        if original is None:
            # Was not set directly on Trainer before; drop any patch that added it.
            if name in Trainer.__dict__:
                delattr(Trainer, name)
        elif Trainer.__dict__.get(name, None) is not original:
            setattr(Trainer, name, original)

    try:
        # Third Party
        import transformers.trainer as _hf_trainer

        _hf_trainer.Trainer = Trainer
    except ImportError:  # pragma: no cover
        pass

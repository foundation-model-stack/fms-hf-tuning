# FMS Acceleration Framework Library

This contains the library code that implements the acceleration plugin framework, in particular the classes:
- `AccelerationFramework`
- `AccelerationPlugin`

The library is envisioned to:
- Provide single integration point into [Huggingface](https://github.com/huggingface/transformers).
- Manage `AccelerationPlugin` in a flexible manner. 
- Load plugins from single configuration YAML, while enforcing compatiblity rules on how plugins can be combined.

See following resources:
- Instructions for [running acceleration framework with `fms-hf-tuning`](https://github.com/foundation-model-stack/fms-hf-tuning)
- [Sample plugin YAML configurations](../../sample-configurations) for important accelerations.

## Using AccelerationFramework with HF Trainer

Being by instantiating an `AccelerationFramework` object, passing a YAML configuration (say via a `path_to_config`):
```python
from fms_acceleration import AccelerationFramework
framework = AccelerationFramework(path_to_config)
```

Plugins automatically configured based on configuration; for more details on how plugins are configured, [see below](#configuration-of-plugins).

Some plugins may require custom model loaders (in replacement of the typical `AutoModel.from_pretrained`). In this case, call `framework.model_loader`:

```python
model = framework.model_loader(model_name_or_path, ...)
```
E.g., in the GPTQ example, see [sample GPTQ QLoRA configuration](../../sample-configurations/accelerated-peft-autogptq-sample-configuration.yaml), we require `model_name_or_path` to be custom loaded from a quantized checkpoint.

We provide a flag `framework.requires_custom_loading` to check if plugins require custom loading.

Also some plugins will require the model to be augmented, e.g., replacing layers with plugin-compliant PEFT adapters.  In this case:

```python
# will also take in some other configs that may affect augmentation
# some of these args may be modified due to the augmentation
# e.g., peft_config will be consumed in augmentation, and returned as None 
#       to prevent SFTTrainer from doing extraneous PEFT logic
model, (peft_config,) = framework.augmentation(
    model, 
    train_args, modifiable_args=(peft_config,),
)
```

We also provide `framework.requires_augmentation` to check if augumentation is required by the plugins.

Finally pass the model to train:

```python
# e.g. using transformers.Trainer. Pass in model (with training enchancements)
trainer = Trainer(model, ...)

# call train
trainer.train()
```

Thats all! the model will not be reap all acceleration speedups based on the plugins that were installed!

## Configuration of Plugins

Each [package](#packages) in this monorepo:
- can be *independently installed*. Install only the libraries you need:
   ```shell
   pip install fms-acceleration/plugins/accelerated-peft
   pip install fms-acceleration/plugins/fused-ops-and-kernels
   ```
- can be *independently configured*. Each plugin is registed under a particular configuration path. E.g., the [autogptq plugin](libs/peft/src/fms_acceleration_peft/framework_plugin_autogptq.py) is reqistered under the config path `peft.quantization.auto_gptq`.
    ```python
    AccelerationPlugin.register_plugin(
        AutoGPTQAccelerationPlugin,
        configuration_and_paths=["peft.quantization.auto_gptq"], 
    )
    ```

    This means that it will be configured under theat exact stanza:
    ```yaml
    plugins:
        peft:
            quantization:
                auto_gptq:
                    # everything under here will be passed to plugin 
                    # when instantiating
                    ...
    ```

- When instantiating `fms_acceleration.AccelerationFramework`, it internally parses through the configuration stanzas. For plugins that are installed, it will instantiate them; for those that are not, it will simply *passthrough*.
- `AccelerationFramework` will manage plugins transparently for user. User only needs to call `AccelerationFramework.model_loader` and `AccelerationFramework.augmentation`.

## Adding New Plugins

To add new plugins:

1. Create an appropriately `pip`-packaged plugin in `plugins`; the package needs to be named like `fms-acceleration-<postfix>` .
2. For `framework` to properly load and manage plugin, add the package `<postfix>` to [constants.py](./src/fms_acceleration/constants.py):

    ```python
    PLUGINS = [
        "peft",
        "foak",
        "<postfix>",
    ]
    ```
3. Create a sample template YAML file inside the `<PLUGIN_DIR>/configs` to demonstrate how to configure the plugin. As an example, reference the [sample config for accelerated peft](../accelerated-peft/configs/autogptq.yaml).
4. Update [generate_sample_configurations.py](../../scripts/generate_sample_configurations.py) and run `tox -e gen-configs` on the top level directory to generate the sample configurations.

    ```python
    KEY_AUTO_GPTQ = "auto_gptq"
    KEY_BNB_NF4 = "bnb-nf4"
    PLUGIN_A = "<NEW PLUGIN NAME>"

    CONFIGURATIONS = {
        KEY_AUTO_GPTQ: "plugins/accelerated-peft/configs/autogptq.yaml",
        KEY_BNB_NF4: (
            "plugins/accelerated-peft/configs/bnb.yaml",
            [("peft.quantization.bitsandbytes.quant_type", "nf4")],
        ),
        PLUGIN_A: (
            "plugins/<plugin>/configs/plugin_config.yaml",
            [
                (<1st field in plugin_config.yaml>, <value>),
                (<2nd field in plugin_config.yaml>, <value>),
            ]
        )
    }

    # Passing a tuple of configuration keys will combine the templates together
    COMBINATIONS = [
        ("accelerated-peft-autogptq", (KEY_AUTO_GPTQ,)),
        ("accelerated-peft-bnb-nf4", (KEY_BNB_NF4,)),    
        (<"combined name with your plugin">), (KEY_AUTO_GPTQ, PLUGIN_A)
        (<"combined name with your plugin">), (KEY_BNB_NF4, PLUGIN_A)
    ]
    ```
5. After sample configuration is generated by `tox -e gen-configs`, update [CONTENTS.yaml](../../sample-configurations/CONTENTS.yaml) with the shortname and the configuration fullpath.
6. Update [scenarios YAML](../../scripts/benchmarks/scenarios.yaml) to configure benchmark test scenarios that will be triggered when running `tox -e run-benches` on the top level directory.
7. Update the [top-level tox.ini](../../tox.ini) to install the plugin for the `run-benches`.


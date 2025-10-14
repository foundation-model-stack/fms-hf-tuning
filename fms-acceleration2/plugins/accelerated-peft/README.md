# FMS Acceleration for Accelerated PeFT Techniques

Currently only supports LoRA-related techniques, but more are in the pipeline to be added:

## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[autogptq](./src/fms_acceleration_peft/framework_plugin_autogptq.py) | Loads 4bit GPTQ-LoRA with quantized GPTQ as base | AutoGPTQ | ✅ | ✅ | ✅ 
[bnb](./src/fms_acceleration_peft/framework_plugin_bnb.py) | Loads 4bit QLoRA with quantized bitsandbytes Linear4 | Huggingface<br>bitsandbytes | ✅ | ✅ | ✅ 


### Key Points
- fix upcasting (resulting in slowdown) issue for `bnb` plugin, originally discovered by inventors of [Unsloth](https://unsloth.ai/blog/mistral-benchmark). **NOTE**: we recommend using *mixed precision* when using 4bit quant for better performance, as per our benchmarks.
- `bnb` properly configured to work with FSDP following [this guide](https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora). 
- `triton_v2` kernels are not yet properly integrated into huggingface optimum.
- `triton_v2` kernels are [the only 4bit kernels that work for training](https://github.com/AutoGPTQ/AutoGPTQ/issues/633).

## GPTQ-LORA's AutoGPTQ - Current Implementation vs Legacy Implementation

GPTQ-LORA depends on an AutoGPTQ backend to run. There are 2 backend options

1. Current Implementation
    - This is an extracted local subset from [ModelCloud's](https://github.com/ModelCloud/GPTQModel) refactored fork.
    - It removes redundant code to simplify build and installation of the plugin
2. Legacy Implementation
    - This requires building the package from the official AutoGPTQ repository
    - To replicate this implementation, follow the installation below

        - The legacy implementation of GPTQ-LORA uses an external AutoGPTQ package, you must ensure the specific commit is installed
            ```
            pip install git+https://github.com/AutoGPTQ/AutoGPTQ.git@ea829c7bbe83561c2b1de26795b6592992373ef7
            ```
        - To construct the plugin, in the configuration object that is passed to the plugin - set `use_external_lib: True` (otherwise defaults to use the local AutoGPTQ package)
        ```
            peft:
            quantization: 
                auto_gptq:
                kernel: triton_v2
                from_quantized: True
                use_external_lib: True
        ```

## Known Issues

<!--
- Models with sliding windows (e.g., Mistral, Mixtral) will have [memory and throughout issues](https://github.com/huggingface/transformers/issues/30461).
-->
- GPTQ-LORA sometimes observed to have `nan` grad norms in the begining of training, but training proceeds well otherwise.

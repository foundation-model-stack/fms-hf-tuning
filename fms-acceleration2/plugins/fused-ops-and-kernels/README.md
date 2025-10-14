# FMS Acceleration for Fused Operations and Kernels

This library contains fused operations and custom kernels, to be expanded over time. Currently it contains the following:


1. Fused operations and kernels extracted from [unsloth](#extracted-code-from-unsloth). 
    - Low-Rank Adapter Fused Operations
    - Fast RoPE Triton Kernels
    - Fast RMS LayerNorm Triton Kernels
    - Fast Cross Entropy Triton Kernels

## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[fast_kernels](./src/fms_accelerate_foak/framework_plugin_fast_kernels.py) | Enhanced version of `fast_quantized_peft`, also works for full-FT and non-quant peft | Contains extracted code |  | ✅

### Supported DataType Settings
**Compatibility Matrix with Mixed Precision**
torch_dtype | Mixed Precision | Full-FT-FOAK | PEFT-FOAK | QPEFT-FOAK
-- | -- | -- | -- | --
FLOAT16 | - | **Compatible**  | **Compatible** | ✗
FLOAT16 | FP16 | ValueError: <br>Attempting to <br>unscale FP16 gradients. <br>[See here](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/troubleshooting.md) | **Compatible** | **Compatible**
BFLOAT16 | - | **Compatible**  | **Compatible**  | ✗
BFLOAT16 | BF16 | **Compatible** | **Compatible** | [Less Performant](https://github.com/foundation-model-stack/fms-acceleration/issues/84)

NOTE: this chart is also a good reference for supported types, even for the non-FOAK case.

### Code Extracted from Unsloth


Notes on the extraction of code from [unsloth](https://github.com/unslothai/unsloth):
- While unsloth is [released under Apache 2.0](https://github.com/unslothai/unsloth/blob/main/LICENSE), there are comments indicating some exceptions strewn throughout the code base, see [an example here](https://github.com/unslothai/unsloth/blob/ec19e61c854dcf9104386fa63fc6c4f2944d4f35/unsloth/models/llama.py#L1140-L1143).
    ```
    it would require a commercial license if used to run on more than 4 GPUs ...
    ```
- These exceptions appear to be located around the trainer improvements, see [another example here](https://github.com/unslothai/unsloth/blob/ec19e61c854dcf9104386fa63fc6c4f2944d4f35/unsloth/models/llama.py#L1177-L1183).
- These exceptions appear around [Feb 2024 Release](https://github.com/unslothai/unsloth/commit/3e4c5a323c16bbda2c92212b790073c4e99c2a55); any code that appears in any file where such exceptions occur **is not extracted**.
- Instead in its place, we have adopted a different approach; we adopt the approach of model patching, as opposed unsloths' approach to rewrite the model. Our approach is novel and **completely rewritten from scratch**. 
- We have also enabled dropout on the lora fused operations.
- All extracted code appears before the Feb 2024 Release. 
- In the table below we record what was extracted, and the exact commit from which it was taken.

Path | Description | Extracted From  | Modifications | Date
--|--|--|--|--
[fused_ops/unsloth_lora](./src/fms_acceleration_foak/fused_ops/unsloth_lora) | QLoRA fast dequant, activation kernels | `unsloth/main` @ [1ecc0185](https://github.com/unslothai/unsloth/commit/1ecc0185a5759c7a0c95dfc96aceea5023cebdfc) |  | 28 Jan 2024
[fused_ops/unsloth_lora/bnb](./src/fms_acceleration_foak/fused_ops/unsloth_lora/bnb) | BNB fast lora | `unsloth/main` @ [1ecc0185](https://github.com/unslothai/unsloth/commit/1ecc0185a5759c7a0c95dfc96aceea5023cebdfc) | `fast_lora.py` | 28 Jan 2024
[fused_ops/unsloth_lora/gptq](./src/fms_acceleration_foak/fused_ops/unsloth_lora/gptq) | GPTQ fast dequant (triton_v2) | `jeromeku/main` @ [2839d39](https://github.com/jeromeku/unsloth/commit/2839d390ef3bb318904289bfb9a7751a782c4e44) | `fast_lora.py`<br>`triton/layers.py` | 6 Feb 2024
[kernels/unsloth](./src/fms_acceleration_foak/kernels/unsloth) | Fast RMS, RoPE, CrossEnt kernels | `unsloth/main` @ [1ecc0185](https://github.com/unslothai/unsloth/commit/1ecc0185a5759c7a0c95dfc96aceea5023cebdfc) | `cross_entropy_loss.py`<br>`rms_layernorm.py` | 28 Jan 2024

### Supported Models

Model | norm | pos emb | cross-ent | fused_lora 
--|--|--|--|--
`LlamaForCausalLM` | ✅  | ✅ | ✅  | ✅ 
`MistralForCausalLM` | ✅  | ✅ | ✅  | ✅ 
`MixtralForCausalLM` | ✅  | ✅ | ✅  | ✅ 
`GPTBigCodeForCausalLM` | ❌  | ❌ | ✅  | ❌ 
`GraniteForCausalLM` | ✅  | ✅ | ✅  | ✅ 

#### Adding Support For A New Model

It is realtively easy by following an existing template, in what follows we use [GraniteForCausalLM](./src/fms_acceleration_foak/models/granite.py) as an example.
- implement a `get_mp_rules` for the new model, which returns a list of `ModelPatcherRule`.
- logic that needs to be changed is the various classes that the rules are triggered on. Import the various module classes likes so:
    ```python
    from transformers.models.granite.modeling_granite import ( 
        GraniteAttention,
        GraniteMLP,
        GraniteRMSNorm,
    )
    ```
- replace the classes appropriately in various locations in `ModelPatcherRule`. In particular the `ModelPatcherTrigger` portions of it. Name `rule_id` appropriately.
    ```python
    ModelPatcherRule(
        rule_id="granite-rms",
        trigger=ModelPatcherTrigger(check=GraniteRMSNorm),
        forward=fast_rms_layernorm,
    )
    ```

### Running Liger Kernel Benchmarks

Using the [scenarios-liger.yaml](../../scripts/benchmarks/scenarios-liger.yaml), this will run full fine tuning, lora peft, autoGPTQ lora peft, and bits-and-bytes lora peft with the triton kernels (Fast RMS, RoPE, CrossEnt) as a base and then run with the liger kernel for LigerFusedLinearCrossEntropy as well as Fast RMS, RoPE to compare results. It only runs against mistral and llama models.

The benchmarks were ran separately for each `num_gpu` entry; they can be run together in a single command, but this is more efficient.

```sh
tox -e run-benches -- 1 "4 8 16 32" benchmark_outputs_1 scenarios-liger.yaml none
tox -e run-benches -- 2 "8 16 32 64" benchmark_outputs_2 scenarios-liger.yaml none
tox -e run-benches -- 4 "16 32 64 128" benchmark_outputs_3 scenarios-liger.yaml none
```


## Known Issues

- MixedPrecision `--fp16` or `--bf16` should be used with `fast_lora`.
- `fast_lora` has issues with FSDP V1 with the `peft` style of FSDP wrapping. 
    * This is because the adapter's forward functions are bypassed in the fused ops.
    * For AutoGPTQ/QLoRA this is addressed by distributing the adapters using DDP so they will be unsharded in time for the fused ops.
- `fast_rope_embeddings` does not work with `postion_ids`, it seems like HF has depracated passing these ids into the rope embedding methods.
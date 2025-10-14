# FMS Acceleration for Attention And Distributed Packing Plugin

This library contains plugins to accelerate finetuning with the following optimizations:

1. Padding-Free Flash Attention Computation
2. Multipack Distributed Sampling


## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[padding_free](./src/fms_acceleration_aadp/framework_plugin_padding_free.py) | Padding-Free Flash Attention Computation | flash_attn | | ✅ | 
[multipack sampler](./src/fms_acceleration_aadp/framework_plugin_multipack.py) | Multipack Distributed Sampling | numba | | ✅ | 


## Native Transformers Support from v4.44.0
Transformers natively supports padding-free from v4.44.0 [see here](https://github.com/huggingface/transformers/pull/31629). The padding-free plugin will use the transformers library if compatible, 
otherwise if `transformers < v4.44.0` the plugin will use an internal implementation instead.

## Native TRL Support for PaddingFree with DataCollatorForCompletionOnlyLM from v0.10.1
Users will be able to use PaddingFree with untokenized data from TRL >= v0.10.1. The flattening of inputs and addition of `position_ids` to the batch
is carried out inside `DataCollatorForCompletionOnlyLM` when keyword `padding_free` is passed to the collator. The plugin uses the TRL library if compatible, 
otherwise if `trl < v0.10.1` the plugin will use an internal implementation instead.

If a user still passes in a pretokenized dataset, the plugin will still use `DataCollaterForFlattening` in the `collate_fn`.

## Running Benchmarks

To reproduce the benchmarks, simply run the following commands,

Reproduce [Padding Free on A100 80GB](scripts/benchmarks/refs_orca/a100_80gb_pf.csv)
`tox -e run-benches -- "1 2" "4 8" benchmark_outputs scenarios-orca.yaml "none"`

Reproduce [MultiPack on A100 80GB](scripts/benchmarks/refs_orca/a100_80gb_mp.csv)
`tox -e run-benches -- "2 4 8" "16 32 64" benchmark_outputs scenarios-orca.yaml "padding-free"`

## Known Issues

### Currenly Only Supports Multipack with Padding-Free

The multipack plugin currently also requires the padding-free plugin to work.
This may change in the future if there is demand for multipack to work standalone without padding free.


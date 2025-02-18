# FMS Acceleration for Mixture-of-Experts

This library contains plugins to accelerate finetuning with the following optimizations:
1. Expert-Parallel MoE with Triton Kernels from ScatterMoE, and some extracted from [megablocks](https://github.com/databricks/megablocks).
    - Megablocks kernels for `gather` and `scatter` 

## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[scattermoe](./src/fms_acceleration_moe/framework_plugin_scattermoe.py) | MoE Expert Parallel with Triton Kernels from scattermoe (& megablocks) | ScatterMoE / extracted kernels from megablocks | ✅ | |  ✅


## Adding New Models

Our `ScatterMoe` implementation is a module-swap; to add new models we need to update the specifications in [scattermoe_constants.py](./src/fms_acceleration_moe/utils/scattermoe_constants.py).
- See the code documentation within to understand how to add new models.

### Using ScatterMoE Saved Checkpoints

`ScatterMoE` checkpoints are saved using `torch.distributed.checkpoint` (DCP) and which is by default `StateDictType.SHARDED_STATE_DICT`:
- `DTensors` limited support for full state dicts. 
- sharded state dicts are the extremely efficient, and require little comms overhead when saving.

We provide a script to recover back the original checkpoint:
- currently the script is only tested in the case where DCP has saved the model in a single node.

If the checkpoint is stored in `hf/checkpoint-10`, call the following to have the converted checkpoint written into `output_dir`:

```
python -m fms_acceleration_moe.utils.checkpoint_utils \
    hf/checkpoint-10 output_dir \
    mistralai/Mixtral-8x7B-Instruct-v0.1
```

## Code Extracted from Megablocks

Notes on code extraction:
- we have only extracted two `autograd` functions [GatherOp](https://github.com/databricks/megablocks/blob/main/megablocks/ops/gather.py) and [ScatterOp](https://github.com/databricks/megablocks/blob/main/megablocks/ops/scatter.py),
- and the associated triton kernels from [backend/kernels.py](https://github.com/databricks/megablocks/blob/main/megablocks/backend/kernels.py); mostly the `_padded_copy`.

## Running Benchmarks


Run the below in the top-level directory of this repo:
- the `scattermoe` dep is not included by default, so the `-x` switch installs it.
- consider disabling the `torch` memory logging to see improved speeds.

```
tox -e run-benches \
    -x testenv:run-benches.deps+="-r plugins/accelerated-moe/requirements-khd.txt" \
    -x testenv:run-benches.setenv+="MEMORY_LOGGING=nvidia" \
    -- \
    "1 2 4" 128 benchmark_outputs scenarios-moe.yaml accelerated-moe-scatter
```
or run the larger `Mixtral-8x7B` bench:
```
tox ... \
    8 128 benchmark_outputs scenarios-moe.yaml accelerated-moe-scatter-mixtral
```

NOTE: if `FileNotFoundError` is observed on the *triton cache*, similar to issues like these:
- https://github.com/triton-lang/triton/issues/2688

then somehow `tox` is causing problems with triton and multiprocessing (there is some race condition).
But the workaound is to first *activate the tox env* and 
running in `bash`:
```
# if FileNotFoundError in the triton cache is observed
# - then activate the env and run the script manually

source .tox/run-benches/bin/activate
bash scripts/run_benchmarks.sh \
    ....
```


### Triton Kernel Dependencies

Currently we do not copy the `scattermoe` kernels into this respository, to this is an additional manual install:

```
# this will install the kernel-hyperdrive fork with the scattermoe triton kernels
pip install -r requirements-khd.txt
```

### Known Issues

These are currently some known issues not yet resolved:
- should eventually remove the dependency on an external `kernel-hyperdrive` repository.
- now support only loading *sharded* `safetensor` non-GGUF MoE checkpoints. This is a reasonable assumption since MoE checkpoints are typically above the size limit that prevents it being saved into a single checkpoint filed.
- when used together with FSDP, the FSDP's `clip_grad_norm` will not properly compute for `ScatterMoE`, see [issue here](https://github.com/foundation-model-stack/fms-acceleration/issues/109).




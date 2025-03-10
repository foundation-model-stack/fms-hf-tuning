# FMS Benchmark Utility

This utility used to measure throughput and other improvements obtained when using `fms-acceleration` plugins.
- [benchmark.py](./benchmark.py): Main benchmark script.
- [scenarios.yaml](./scenarios.yaml): `sft_trainer.py` arguments organized different *scenarios*.
  * Each `scenario` may apply to one ore more `AccelerationFramework` [sample configuration](../../sample-configurations). These are the *critical* arguments needed for correct operation.
  * See [section on benchmark scenarios](#benchmark-scenarios) for more details.
- [defaults.yaml](./defaults.yaml): `sft_trainer.py` arguments that may be used in addition to [scenarios.yaml](./scenarios.yaml). These are the *non-critical* arguments that will not affect plugin operation.
- [accelerate.yaml](./accelerate.yaml): configurations required by[`accelerate launch`](https://huggingface.co/docs/accelerate/en/package_reference/cli) for multi-gpu benchmarks.


## Benchmark Scenarios

An example of a `scenario` for `accelerated-peft-gptq` given as follows:
```yaml
scenarios:

  # benchmark scenario for accelerated peft using AutoGPTQ triton v2
  - name: accelerated-peft-gptq
    framework_config: 
      # one ore more framework configurations that fall within the scenario group.
      # - each entry points to a shortname in CONTENTS.yaml
      - accelerated-peft-autogptq

    # sft_trainer.py arguments critical for correct plugin operation
    arguments:
      fp16: True
      learning_rate: 2e-4
      torch_dtype: float16
      peft_method: lora
      r: 16
      lora_alpha: 16
      lora_dropout: 0.0
      target_modules: "q_proj k_proj v_proj o_proj"
      model_name_or_path: 
        - 'mistralai/Mistral-7B-v0.1'
        - 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        - 'NousResearch/Llama-2-70b-hf'
```

A `scenario` has the following key components:
- `framework_config`: points to one or more [acceleration configurations](#sample-acceleration-configurations). 
  * list of [sample config `shortname`](../../sample-configurations/CONTENTS.yaml).
  * for each `shortname` is a different bench.
- `arguments`: the *critical* `sft_trainer.py` arguments that need to be passed in alongiside `framework_config` to ensure correct operation.
  * `model_name_or_path` is a list, and the bench will enumerate all of them.
  * **NOTE**: a `plugin` **may not work with arbitrary models**. This depends on the plugin's setting of [`AccelerationPlugin.restricted_model_archs`](../../plugins/framework/src/fms_acceleration/framework_plugin.py).


## Usage

The best way is via `tox` which manages the dependencies, including installing the correct version [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning).

- install the `setup_requirements.txt` to get `tox`:
    ```
    pip install -r setup_requirements.txt
    ```

- run a *small* representative set of benches:
    ```
    tox -e run-benches
    ```
- run the *full* set of benches on for both 1 and 2 GPU cases:
    ```
    tox -e run-benches -- "1 2" 
    ```

Note:
- `tox` command above accepts environment variables `DRY_RUN, NO_DATA_PROCESSING, NO_OVERWRITE`. See `scripts/run_benchmarks.sh`

## Running Benchmarks

The convinience script [`run_benchmarks.sh`](../run_benchmarks.sh) configures and runs `benchmark.py`; the command is:
```
bash run_benchmarks.sh NUM_GPUS_MATRIX RESULT_DIR SCENARIOS_CONFIG SCENARIOS_FILTER
```
where:
- `NUM_GPUS_MATRIX`: list of `num_gpu` settings to bench for, e.g. `"1 2"` will bench for 1 and 2 gpus.
- `EFFECTIVE_BS_MATRIX`: list of effective batch sizes, e.g., `"4 8"` will bench for effective batch sizes 4 and 8.
- `RESULT_DIR`: where the benchmark results will be placed.
- `SCENARIOS_CONFIG`: the `scenarios.yaml` file.
- `SCENARIOS_CONFIG`: specify to run only a specific `scenario` by providing the specific `scenario` name.

The recommended way to run `benchmarks.sh` is using `tox` which handles the dependencies:
```
tox -e run-benches -- NUM_GPUS_MATRIX EFFECTIVE_BS_MATRIX RESULT_DIR SCENARIOS_CONFIG SCENARIOS_FILTER
```

Alternatively run [`benchmark.py`](./benchmark.py) directly. To see the help do:
```
python benchmark.py --help
```

Note:
- in `run_benchmarks.sh` we will clear the `RESULT_DIR` if it exists, to avoid contaimination with old results. To protect against overwrite, then always run with `NO_OVERWRITE=true`.

## Logging GPU Memory

There are 2 ways to benchmark memory in `run_benchmarks.sh`:
- Setting the environment variable `MEMORY_LOGGING=nvidia` will use Nvidia `nvidia-smi`'s API
- Setting the environment variable `MEMORY_LOGGING=huggingface` (default) will use HuggingFace `HFTrainer`'s API 

Both approaches will print out the memory values to the benchmark report.
 - For Nvidia, the result column will be `nvidia_mem_reserved`
 - For Torch/HF, the result column will be `peak_torch_mem_alloc_in_bytes` and `torch_mem_alloc_in_bytes`

### Nvidia-SMI `nvidia-smi`
`nvidia-smi` is a command line utility (CLI) based on the Nvidia Manage Library (NVML)`. A separate process call is used to start, log and finally terminate the CLI for every experiment.  

The keyword `memory.used` is passed to `--query-gpu` argument to log the memory usage at some interval. The list of keywords that can be logged can be referenced from `nvidia-smi --help-query-gpu`

Since it runs on a separate process, it is less likely to affect the training. However, it is a coarser approach than HF as NVML's definition of used memory takes the sum of (memory allocated + memory reserved). Refer to their [documentation](https://docs.nvidia.com/deploy/nvml-api/structnvmlMemory__t.html#structnvmlMemory__t:~:text=Sum%20of%20Reserved%20and%20Allocated%20device%20memory%20(in%20bytes).%20Note%20that%20the%20driver/GPU%20always%20sets%20aside%20a%20small%20amount%20of%20memory%20for%20bookkeeping) here.

After every experiment, 
  - the logged values are calibrated to remove any existing foreign memory values
  - the peak values for each gpu device are taken
  - the values are finally averaged across all devices.

### Torch/HuggingFace `HFTrainer`
HFTrainer has a feature to log memory through the `skip_memory_metrics=False` training argument. In their [documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.skip_memory_metrics), it is mentioned that setting this argument to `False` will affect training speed. In our tests so far (below), we do not see significant difference in throughput (tokens/sec) when using this argument.

The HFTrainer API is more granular than `nvidia-smi` as it uses `torch.cuda` to pinpoint memory usage inside the trainer
  - It reports the allocated memory by calling `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` inside its probes
  - It has memory logging probes at different stages of the Trainer - `init`, `train`, `evaluate`, `predict` 

##### NOTE:
- When in distributed mode, the Trainer will only log the rank 0 memory.
- For stability purposes, it only tracks the outer level of train, evaluate and predict methods. i.e. if eval is called during train, there won't be a nested invocation of the memory probe.
- Any GPU memory incurred outside of the defined Trainer stages won't be tracked.

### Additional Details

#### Calculating Memory from HFTrainer Output Metrics

This is an example of the memory values that HFTrainer will produce in the outputs of `train()`
```
output_metrics = {
    'train_runtime': 191.2491, 
    'train_samples_per_second': 0.209, 
    'train_steps_per_second': 0.052, 
    'train_tokens_per_second': 428.342, 
    'train_loss': 1.0627506256103516, 
    'init_mem_cpu_alloc_delta': 4096, 
    'init_mem_gpu_alloc_delta': 0, 
    'init_mem_cpu_peaked_delta': 0, 
    'init_mem_gpu_peaked_delta': 0, 
    'train_mem_cpu_alloc_delta': 839086080, 
    'train_mem_gpu_alloc_delta': -17491768832, 
    'train_mem_cpu_peaked_delta': 0, 
    'train_mem_gpu_peaked_delta': 26747825664, 
    'before_init_mem_cpu': 5513297920, 
    'before_init_mem_gpu': 36141687296, 
    'epoch': 0.01
}
```

We refer to the keys of the memory metrics in this order 
 - `before_init_mem_X` as stage0 
 - `init_mem_X` as stage1 
 - `train_mem_X` as stage2
 - ... 

We currently compute the memory values in the report by taking the largest of sums. For example:

For allocated memory value
```
max([
  stage0_mem,
  stage0_mem + stage1_allocated_delta, 
  stage0_mem + stage1_allocated_delta + stage2_allocated_delta,
  ...
])
```

For peak memory value
```
max([
  stage0_mem,
  stage0_mem + stage1_allocated_delta + stage1_peaked_delta, 
  stage0_mem + stage1_allocated_delta + stage2_allocated_delta + stage2_peaked_delta,
  ...
])
```


We compare memory values between Nvidia-SMI and Torch in this PR - [Memory Benchmarking](https://github.com/foundation-model-stack/fms-acceleration/pull/14).



#!/usr/bin/env bash

set -x

echo "FMS Acceleration Benchmarking Script"
echo "Please run this script as "
echo "bash scripts/run_benchmarks.sh ... from the root of the repo"

# TODO: this can be improved. For now we assume we always run from 
# root of repo
WORKING_DIR=scripts/benchmarks

# pointing to the configuration directory of the repo
CONFIG_DIR=sample-configurations

# ------------- MAIN CONFIGS -----------------
SCENARIOS_CONFIG=scenarios.yaml
DEFAULTS_CONFIG=defaults.yaml
ACCELERATE_CONFIG=accelerate.yaml

# ------------- SCENARIO CONFIGS -----------------
# this determines which is the default subset
SCNTAG_PEFT_AUTOGPTQ=accelerated-peft-gptq

# ------------- OTHER CONFIGS -----------------

# data will be cached in here
DATA_CACHE=data

# final result placed here
BENCH_RESULT_FILE=benchmarks.csv

# freeze the pip requirements here
PIP_REQUIREMENTS_FILE=requirements.txt

# ------------- DROP COLUMNS FRO RESULTS -----------------
# env inputs
DRY_RUN=${DRY_RUN:-"false"}
NO_DATA_PROCESSING=${NO_DATA_PROCESSING:-"false"}
NO_OVERWRITE=${NO_OVERWRITE:-"false"}
MEMORY_LOGGING=${MEMORY_LOGGING:-"all"}

# inputs
NUM_GPUS_MATRIX=${1:-"1 2"}
EFFECTIVE_BS_MATRIX=${2:-"4 8"}
RESULT_DIR=${3:-"benchmark_outputs"}
SCENARIOS_CONFIG=${4:-$SCENARIOS_CONFIG}
SCENARIOS_FILTER=${5-$SCNTAG_PEFT_AUTOGPTQ}

echo "NUM_GPUS_MATRIX: $NUM_GPUS_MATRIX"
echo "RESULT_DIR: $RESULT_DIR"
echo "SCENARIOS_CONFIG: $SCENARIOS_CONFIG"
echo "SCENARIOS_FILTER: $SCENARIOS_FILTER"
echo "MEMORY_LOGGING: $MEMORY_LOGGING"

if [ -n "$RESULT_DIR" ]; then
    echo "The results directory is not empty. "
    if [ "$NO_OVERWRITE" = "true" ]; then 
        echo "Results dir $RESULT_DIR is not empty, but NO_OVERWRITE=true"
        echo "If intending to overwrite please delete the folder manually"
        echo "or do not set NO_OVERWRITE"
    else
        echo "Deleting $RESULT_DIR"
        rm -rf $RESULT_DIR
    fi
fi

# tag on the directories
SCENARIOS_CONFIG=$WORKING_DIR/$SCENARIOS_CONFIG
DEFAULTS_CONFIG=$WORKING_DIR/$DEFAULTS_CONFIG
ACCELERATE_CONFIG=$WORKING_DIR/$ACCELERATE_CONFIG
DATA_CACHE=$RESULT_DIR/$DATA_CACHE
BENCH_RESULT_FILE=$RESULT_DIR/$BENCH_RESULT_FILE
PIP_REQUIREMENTS_FILE=$RESULT_DIR/$PIP_REQUIREMENTS_FILE

# ------------- EXTRA ARGS -----------------

# preload models by default
EXTRA_ARGS="--preload_models"

if [ "$SCENARIOS_FILTER" != "none" ]; then 
    EXTRA_ARGS="$EXTRA_ARGS --run_only_scenarios $SCENARIOS_FILTER"
fi

if [ "$DRY_RUN" = "true" ]; then 
    EXTRA_ARGS="$EXTRA_ARGS --dry_run"
fi

if [ "$NO_DATA_PROCESSING" = "true" ]; then 
    EXTRA_ARGS="$EXTRA_ARGS --no_data_processing"
fi

if [ "$MEMORY_LOGGING" = "huggingface" ]; then 
    EXTRA_ARGS="$EXTRA_ARGS --log_memory_hf"
elif [ "$MEMORY_LOGGING" = "nvidia" ]; then 
    EXTRA_ARGS="$EXTRA_ARGS --log_nvidia_smi"
elif [ "$MEMORY_LOGGING" = "all" ]; then 
    EXTRA_ARGS="$EXTRA_ARGS --log_nvidia_smi --log_memory_hf"
fi

# dump out the environment
if [ ! "$NO_OVERWRITE" = "true" ]; then 
    echo "Creating $RESULT_DIR"
    mkdir -p $RESULT_DIR
    pip freeze > $PIP_REQUIREMENTS_FILE
fi

# run the bench
PYTHONPATH=. \
python $WORKING_DIR/benchmark.py \
   --num_gpus $NUM_GPUS_MATRIX \
   --effective_batch_size_matrix $EFFECTIVE_BS_MATRIX \
   --scenarios_config_path $SCENARIOS_CONFIG \
   --accelerate_config $ACCELERATE_CONFIG \
   --defaults_config_path $DEFAULTS_CONFIG \
   --dataset_save_path $DATA_CACHE \
   --results_output_path $RESULT_DIR $EXTRA_ARGS

# produce the final CSV for checkin
# need to set PYTHONPATH because there is an import inside
# this will write to the BENCH_RESULT_FILE
# Remove the columns with values already represented by other metrics in the summary report
PYTHONPATH=. \
    python $WORKING_DIR/display_bench_results.py $RESULT_DIR \
    --result_file $BENCH_RESULT_FILE \
    --keep_columns \
        'torch_dtype' \
	    "framework_config" \
		"peft_method" \
     	"model_name_or_path" \
		"num_gpus" \
		"per_device_train_batch_size" \
		"mem_nvidia_mem_reserved" \
	    "mem_peak_torch_mem_alloc_in_bytes" \
		"mem_torch_mem_alloc_in_bytes" \
		"train_tokens_per_second" \
    --remove_columns \
        'before_init_mem_cpu' \
        'before_init_mem_gpu' \
        'init_mem_cpu_alloc_delta' \
        'init_mem_cpu_peaked_delta' \
        'init_mem_gpu_alloc_delta' \
        'init_mem_gpu_peaked_delta' \
        'train_mem_cpu_alloc_delta' \
        'train_mem_cpu_peaked_delta' \
        'train_mem_gpu_alloc_delta' \
        'train_mem_gpu_peaked_delta' \
        'training_data_path' \
        'error_messages' \
        'acceleration_framework_config_file'


# For every new benchmark run, it is good practice to perform a regression check
# against a previous known set of benchmark results. This repo provides a convenient comparison
# tool that analyses the differences of metrics like loss and throughput between an old and new set 
# of benchmark results.
# To use this tool simply run the following python command
# PYTHONPATH=. \
#     python $WORKING_DIR/compare_with_reference.py
# The following arguments can be used to further configure the analysis, otherwise it uses default values
#   arguments:
#   --result_dir <Output directory to save comparison artifacts>
#   --reference_benchmark_filepath <filepath of the old benchmark results to compare againts>
#   --threshold_ratio <to define an acceptable difference between old and new results>
#   --indices <defines the set of column names used as unique identifier to merge the 2 sets of results>
#   --plot_columns <specifies the metric name to be compared and vizualized>
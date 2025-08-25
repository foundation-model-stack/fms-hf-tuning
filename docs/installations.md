# Table of Contents

Table of contents:
 - [Basic Installations](#basic-installation)
 - [Installing FlashAttention](#using-flashattention)
 - [Installing Fms Acceleration](#using-fms-acceleration)
 - [Installing Mamba Model Support](#training-mamba-models)

## Basic Installation

```
pip install fms-hf-tuning
```

## Using FlashAttention

> Note: After installing, if you wish to use [FlashAttention](https://github.com/Dao-AILab/flash-attention), then you need to install these requirements:
```sh
pip install fms-hf-tuning[dev]
pip install fms-hf-tuning[flash-attn]
```
[FlashAttention](https://github.com/Dao-AILab/flash-attention) requires the [CUDA Toolit](https://developer.nvidia.com/cuda-toolkit) to be pre-installed.

*Debug recommendation:* While training, if you encounter flash-attn errors such as `undefined symbol`, you can follow the below steps for clean installation of flash binaries. This may occur when having multiple environments sharing the pip cache directory or torch version is updated.

```sh
pip uninstall flash-attn
pip cache purge
pip install fms-hf-tuning[flash-attn]
```

## Using FMS-Acceleration

`fms-acceleration` is a collection of plugins that packages that accelerate fine-tuning / training of large models, as part of the `fms-hf-tuning` suite. For more details see [this document](./docs/tuning-techniques.md#fms-acceleration).

If you wish to use [fms-acceleration](https://github.com/foundation-model-stack/fms-acceleration), you need to install it. 
```
pip install fms-hf-tuning[fms-accel]
```

## Using Experiment Trackers

To use experiment tracking with popular tools like [Aim](https://github.com/aimhubio/aim), note that some trackers are considered optional dependencies and can be installed with the following command:
```
pip install fms-hf-tuning[aim]
```
For more details on how to enable and use the trackers, Please see, [the experiment tracking section below](#experiment-tracking).

## Training Mamba Models

To train Mamba models one needs to have `mamba-ssm` package installed which is compatible with fms-hf-tuning to ensure the optimal training. Not using this package while training Mamba models can result in higher resource usage and suboptimal performance.

Install this as 
```
pip install fms-hf-tuning[mamba]
```


```
pip install fms-hf-tuning
```
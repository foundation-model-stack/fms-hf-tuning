# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

# Standard
import functools

# Third Party
import torch

# Local
from .kernels import gather as _kernels_gather
from .kernels import scatter as _kernels_scatter
from .kernels import scatter_wgrad as _kernels_scatter_wgrad


# ------------------------ HELPERS -----------------------------
def _is_eligible(x):
    return x.is_floating_point() and x.is_cuda and (x.dtype is not torch.float64)


def _cast(x, dtype):
    if isinstance(x, torch.Tensor) and _is_eligible(x):
        return x.to(dtype)
    elif isinstance(x, map):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in x.items()}
    elif isinstance(x, list) or isinstance(x, tuple):
        return type(x)(map(lambda y: _cast(y, dtype), x))
    return x


def custom_fwd(fwd):
    """Wrap a custom autograd function that always uses autocast dtype."""

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        if torch.is_autocast_enabled():
            with torch.autocast(device_type="cuda", enabled=False):
                dtype = torch.get_autocast_gpu_dtype()
                return fwd(*_cast(args, dtype), **_cast(kwargs, dtype))
        return fwd(*args, **kwargs)

    return decorate_fwd


def custom_bwd(bwd):
    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with torch.autocast(device_type="cuda", enabled=False):
            return bwd(*args, **kwargs)

    return decorate_bwd


# ------------------------ AUTOGRAD -----------------------------


class AllToAllOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, output_split_sizes, input_split_sizes, group, async_op):
        out = torch.empty(
            (sum(output_split_sizes),) + x.shape[1:], device=x.device, dtype=x.dtype
        )

        ctx.input_shape = x.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        handle = torch.distributed.all_to_all_single(
            out,
            x,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )
        return out, handle

    @staticmethod
    def backward(ctx, grad, _):
        if ctx.needs_input_grad[0]:
            out = torch.empty(
                ctx.input_shape,
                device=grad.device,
                dtype=grad.dtype,
            )
            torch.distributed.all_to_all_single(
                out,
                grad,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            return out, None, None, None, None
        return None, None, None, None, None


def all_to_all(x, output_split_sizes, input_split_sizes, group, async_op=False):
    return AllToAllOp.apply(
        x,
        output_split_sizes,
        input_split_sizes,
        group,
        async_op,
    )


# Autograd wrapper for scatter kernel.
class ScatterOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, indices, bin_ids, weights, bins, top_k):
        maybe_x = [x] if ctx.needs_input_grad[3] else []
        ctx.save_for_backward(indices, bin_ids, weights, bins, *maybe_x)
        ctx.top_k = top_k
        ctx.x_shape = x.shape
        return _kernels_scatter(x, indices, bin_ids, weights, bins, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()
        saved_tensors = ctx.saved_tensors

        indices, bin_ids, weights, bins = saved_tensors[:4]
        dgrad = None
        if ctx.needs_input_grad[0]:
            dgrad = _kernels_gather(
                grad,
                indices,
                bin_ids,
                weights,
                bins,
                ctx.top_k,
            )

        wgrad = None
        if ctx.needs_input_grad[3]:  # need wgrad
            x = saved_tensors[-1]
            wgrad = _kernels_scatter_wgrad(
                x,
                grad,
                indices,
                bin_ids,
                bins,
                ctx.top_k,
            )
        return dgrad, None, None, wgrad, None, None, None


def scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
):
    return ScatterOp.apply(x, indices, bin_ids, weights, bins, top_k)


# Autograd wrapper for gather kernel.
class GatherOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, indices, bin_ids, bins, top_k):
        ctx.save_for_backward(indices, bin_ids, bins)
        ctx.top_k = top_k
        return _kernels_gather(x, indices, bin_ids, None, bins, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()

        indices, bin_ids, bins = ctx.saved_tensors
        out = _kernels_scatter(grad, indices, bin_ids, None, bins, ctx.top_k)
        return out, None, None, None, None, None


gather = GatherOp.apply

# taken from 
# https://github.com/jeromeku/unsloth/commit/
# 2839d390ef3bb318904289bfb9a7751a782c4e44

# with modifications from The IBM Tuning Team

from dataclasses import dataclass
from logging import getLogger
from typing import Optional

import torch

from .triton.kernels import dequant248
from ..swiglu import swiglu_DWf_DW_dfg_kernel, swiglu_fg_kernel

logger = getLogger(__name__)


@dataclass
class GPTQuantState:
    """
    Stores params for GPTQ linear layer quantization
    """

    infeatures: int
    outfeatures: int

    bits: int
    group_size: int
    maxq: int
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor
    g_idx: torch.Tensor

    # cuda_kernel params (not used currently)
    kernel_switch_threshold: int
    autogptq_cuda_available: bool = False
    autogptq_cuda: bool = False

    wf: Optional[torch.Tensor] = None
    use_cuda_fp16: bool = False

    bias: Optional[torch.Tensor] = None
    trainable: bool = True


def unpack_gptqstate(qstate):
    qweight, scales, qzeros, g_idx, bits = (
        qstate.qweight,
        qstate.scales,
        qstate.qzeros,
        qstate.g_idx,
        qstate.bits,
    )
    return qweight, scales, qzeros, g_idx, bits


def extract_gptq_state(qmodule):
    if hasattr(qmodule, "base_layer"):
        qmodule = qmodule.base_layer

    def check_bias(qmodule):
        if hasattr(qmodule, "bias") and qmodule.bias is not None:
            if qmodule.bias.count_nonzero() > 0:
                return qmodule.bias
        return None

    return GPTQuantState(
        infeatures=qmodule.infeatures,
        outfeatures=qmodule.outfeatures,
        bits=qmodule.bits,
        group_size=qmodule.group_size,
        maxq=qmodule.maxq,
        qweight=qmodule.qweight.cuda(),
        qzeros=qmodule.qzeros.cuda(),
        scales=qmodule.scales.cuda(),
        g_idx=qmodule.g_idx.cuda(),
        bias=check_bias(qmodule),
        wf=qmodule.wf.cuda() if hasattr(qmodule, "wf") else None,
        kernel_switch_threshold=(
            qmodule.kernel_switch_threshold
            if hasattr(qmodule, "kernel_switch_threshold")
            else None
        ),
        autogptq_cuda_available=( # fixed by @aaron.chew1@sg.ibm.com
            qmodule.autogptq_cuda_available 
            if hasattr(qmodule, "autogptq_cuda_available") else False
        ),
        # use_cuda_fp16=qmodule.use_cuda_fp16,
    )


def get_lora_parameters(proj):
    # For DPO or disabled adapters
    base_layer = proj.base_layer if hasattr(proj, "base_layer") else proj
    qstate = extract_gptq_state(base_layer)
    bias = base_layer.bias if hasattr(base_layer, 'bias') else None

    if base_layer.__module__.startswith("auto_gptq"):
        setattr(qstate.qzeros, "offset", 1)

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return qstate, None, None, None, None, None

    active_adapter = (
        proj.active_adapters[0]
        if hasattr(proj, "active_adapters")
        else proj.active_adapter
    )
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]
    dropout = proj.lora_dropout[active_adapter] if hasattr(proj, "lora_dropout") else None
    dropout.X = None
    return qstate, bias, A, B, s, dropout

# modified by aaron.chew1@ibm.com
def matmul_lora_canonicalized(X, W, A, B, s, dropout=None):
    """
    X: rank-2 tensor (batch, seq_len) x (din)
    W: rank-2 tensor (din, dout)
    out: rank-2 tensor (batch, seq_len) x (dout)
    din = X.shape[1]
    dout = W.shape[1]
    """

    out = torch.matmul(X, W)
    if dropout is not None:
        if isinstance(dropout, torch.Tensor):
            X *= dropout
        elif isinstance(dropout, torch.nn.Module):
            X = dropout(X)        
            dropout.X = X
        else:
            raise NotImplementedError("dropout must be a tensor or module.")
    A, B = A.t(), B.t()
    out += (X @ A) @ (s * B)

    return out

# modified by aaron.chew1@ibm.com
def matmul_lora(X, W, A, B, s, out=None, dropout=None):
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    out = torch.matmul(X, W, out=out)

    if A is not None:
        # LoRA is enabled
        if dropout is not None:
            if isinstance(dropout, torch.Tensor):
                X *= dropout
            elif isinstance(dropout, torch.nn.Module):
                # save post-dropout X for backward computation
                X = dropout(X)
                dropout.X = X
            else:
                raise NotImplementedError("dropout must be a tensor or module.")
        A, B = A.t(), B.t()
        out += (X @ A.to(dtype)) @ (s * B.to(dtype))

    return out.view(batch, seq_len, -1) if reshape else out


# modified by flim@sg.ibm.com
# modified by aaron.chew1@ibm.com
class LoRA_MLP(torch.autograd.Function):
    """
    ### LoRA weights
    G = G + Ag @ Bg
    U = U + Au @ Bu
    W = W + Aw @ Bw

    ### SwiGLU(X)
    e = X @ G
    f = e * sigmoid(e)
    g = X @ U
    h = f * g
    i = h @ W

    ### Backpropagation chain rule
    See our blog post for more details

    df = sigmoid(e) * (1 - f) + f
    dC/dW = h.T @ dY
    dC/dU = X.T @ (D @ W.T * f)
    dC/dG = X.T @ (D @ W.T * df * g)

    ### Down projection LoRA weights
    dC/dAw = dC/dW @ B.T
    dC/dBw = A.T @ dC/dW
    dC/dAw =       h.T @ dY @ B.T
    dC/dBw = A.T @ h.T @ dY

    ### Up projection LoRA weights
    dC/dAu =       X.T @ (D @ W.T * f) @ B.T
    dC/dBu = A.T @ X.T @ (D @ W.T * f)

    ### Gate projection LoRA weights
    dC/dAg =       X.T @ (D @ W.T * df * g) @ B.T
    dC/dBg = A.T @ X.T @ (D @ W.T * df * g)

    Don't forget to see our blog post for more details!
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(
        ctx,
        X: torch.Tensor,
        gate_qweight,
        gate_scales,
        gate_qzeros,
        gate_g_idx,
        gate_bits,
        gate_bias,
        gateA,
        gateB,
        gateS,
        up_qweight,
        up_scales,
        up_qzeros,
        up_g_idx,
        up_bits,
        up_bias,
        upA,
        upB,
        upS,
        down_qweight,
        down_scales,
        down_qzeros,
        down_g_idx,
        down_bits,
        down_bias,
        downA,
        downB,
        downS,
        dropout_gate=None,
        dropout_up=None,
        dropout_down=None,
    ):
        dtype = X.dtype

        # Separate dequant248 from matmul
        gateW = dequant248(
            gate_qweight, gate_scales, gate_qzeros, gate_g_idx, gate_bits
        )
        e = matmul_lora(X, gateW, gateA, gateB, gateS, dropout=dropout_gate)
        upW = dequant248(up_qweight, up_scales, up_qzeros, up_g_idx, up_bits)
        g = matmul_lora(X, upW, upA, upB, upS, dropout=dropout_up)
        if gate_bias is not None: e += gate_bias
        if up_bias is not None: g += up_bias
        # f = torch.nn.functional.silu(e)
        # h = f * g
        h = swiglu_fg_kernel(e, g)

        downW = dequant248(
            down_qweight, down_scales, down_qzeros, down_g_idx, down_bits
        )
        i = matmul_lora(h, downW, downA, downB, downS, dropout=dropout_down)
        if down_bias is not None: i += down_bias

        ctx.custom_saved_tensors = (
            gate_qweight,
            gate_scales,
            gate_qzeros,
            gate_g_idx,
            gate_bits,
            gateS,
            up_qweight,
            up_scales,
            up_qzeros,
            up_g_idx,
            up_bits,
            upS,
            down_qweight,
            down_scales,
            down_qzeros,
            down_g_idx,
            down_bits,
            downS,
        )

        # Extract post-dropout X for use in backward computation
        _dropped_X = []
        for _dropout in [
            dropout_gate, dropout_up, dropout_down
        ]:
            if _dropout is not None:
                # then matmul_lora must have attached the X
                _dropped_X.append(_dropout.X)
                del _dropout.X # remove it
            else:
                # otherwise will use X
                _dropped_X.append(X)

        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g,
            *_dropped_X
        )
        return i

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dY: torch.Tensor):
        (
            gate_qweight,
            gate_scales,
            gate_qzeros,
            gate_g_idx,
            gate_bits,
            gateS,
            up_qweight,
            up_scales,
            up_qzeros,
            up_g_idx,
            up_bits,
            upS,
            down_qweight,
            down_scales,
            down_qzeros,
            down_g_idx,
            down_bits,
            downS,
        ) = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, \
            X, e, g, gateX, upX, downX = ctx.saved_tensors

        gateA, gateB, upA, upB, downA, downB = (
            gateA.t(),
            gateB.t(),
            upA.t(),
            upB.t(),
            downA.t(),
            downB.t(),
        )

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X = X.view(-1, X.shape[-1])
        e = e.view(-1, e.shape[-1])
        g = g.view(-1, g.shape[-1])
        dtype = X.dtype

        downW = dequant248(
            down_qweight, down_scales, down_qzeros, down_g_idx, down_bits
        )
        DW = matmul_lora(
            dY, downW.t(), downB, downA, downS,
            dropout=(downX !=0)
        )
        # e = e.float()
        # se = 1.0 / (1.0 + torch.exp(-e))
        # f = (se * e).to(dtype)
        # h = f * g
        # df = DW * f
        # dg = DW * g
        # de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
        DW, e, g = swiglu_DWf_DW_dfg_kernel(DW, e, g)
        h, df, de = DW, e, g

        # Down projection LoRA weights
        d_downA = h.t() @ (dY @ downB.t())
        d_downB = (downA.t() @ h.t()) @ dY
        d_downA *= downS
        d_downB *= downS

        # Up projection LoRA weights
        d_upA = upX.t() @ (df @ upB.t())
        d_upB = (upA.t() @ upX.t()) @ df
        d_upA *= upS
        d_upB *= upS

        # Gate projection LoRA weights
        d_gateA = gateX.t() @ (de @ gateB.t())
        d_gateB = (gateA.t() @ gateX.t()) @ de
        d_gateA *= gateS
        d_gateB *= gateS

        # dX  = matmul_lora(df, upW.t(), upW_quant, upB, upA, upS)
        # dX += matmul_lora(de, gateW.t(), gateW_quant, gateB, gateA, gateS)
        upW = dequant248(up_qweight, up_scales, up_qzeros, up_g_idx, up_bits)
        dX = torch.matmul(df, upW.t())  # , out=X)
        del upW
        dX += (upX != 0) * (df @ upB.to(dtype).t() @ (upS * upA.to(dtype).t()))

        gateW = dequant248(
            gate_qweight, gate_scales, gate_qzeros, gate_g_idx, gate_bits
        )
        dX += de @ gateW.t()
        del gateW
        dX += (gateX != 0) * (de @ gateB.to(dtype).t() @ (gateS * gateA.to(dtype).t()))

        # qweight, scales, qzeros, g_idx, bits
        #  upW,    upW_quant,   upA,   upB,   upS,
        # downW, downW_quant, downA, downB, downS,
        return (
            dX.view(batch, seq_len, hd),
            None,  # qweight
            None,  # scales
            None,  # qzeros
            None,  # g_idx
            None,  # bits
            None,  # gate_bias
            d_gateA.t(),
            d_gateB.t(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # up_bias 
            d_upA.t(),
            d_upB.t(),
            None,  # dS
            None,
            None,
            None,
            None,
            None,
            None,  # down_bias 
            d_downA.t(),
            d_downB.t(),
            None,
            None,
            None,
            None,
        )


def apply_lora_mlp(self, X):
    gateQstate, gate_bias, gateA, gateB, gateS, dropout_gate = get_lora_parameters(self.gate_proj)
    upQState,   up_bias, upA, upB, upS, dropout_up = get_lora_parameters(self.up_proj)
    downQState, down_bias, downA, downB, downS, dropout_down = get_lora_parameters(self.down_proj)
    out = LoRA_MLP.apply(
        X,
        *unpack_gptqstate(gateQstate),
        gate_bias,
        gateA,
        gateB,
        gateS,
        *unpack_gptqstate(upQState),
        up_bias,
        upA,
        upB,
        upS,
        *unpack_gptqstate(downQState),
        down_bias,
        downA,
        downB,
        downS,
        dropout_gate,
        dropout_up,
        dropout_down,
    )
    return out


class LoRA_QKV(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    See our blogpost for more details.

    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)
    We then sum them all find dC/dX

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(
        ctx,
        X: torch.Tensor,
        Q_qweight,
        Q_scales,
        Q_qzeros,
        Q_g_idx,
        Q_bits,
        Q_bias,
        QA,
        QB,
        QS,
        K_qweight,
        K_scales,
        K_qzeros,
        K_g_idx,
        K_bits,
        K_bias,
        KA,
        KB,
        KS,
        V_qweight,
        V_scales,
        V_qzeros,
        V_g_idx,
        V_bits,
        V_bias,
        VA,
        VB,
        VS,
        dropout_Q=None,
        dropout_K=None,
        dropout_V=None,

    ):
        dtype = X.dtype

        QW = dequant248(Q_qweight, Q_scales, Q_qzeros, Q_g_idx, Q_bits)
        KW = dequant248(K_qweight, K_scales, K_qzeros, K_g_idx, K_bits)
        VW = dequant248(V_qweight, V_scales, V_qzeros, V_g_idx, V_bits)
        Q = matmul_lora(X, QW, QA, QB, QS, dropout=dropout_Q)
        K = matmul_lora(X, KW, KA, KB, KS, dropout=dropout_K)
        V = matmul_lora(X, VW, VA, VB, VS, dropout=dropout_V)

        if Q_bias is not None: Q += Q_bias
        if K_bias is not None: K += K_bias
        if V_bias is not None: V += V_bias

        ctx.custom_saved_tensors = (
            Q_qweight,
            Q_scales,
            Q_qzeros,
            Q_g_idx,
            Q_bits,
            QS,
            K_qweight,
            K_scales,
            K_qzeros,
            K_g_idx,
            K_bits,
            KS,
            V_qweight,
            V_scales,
            V_qzeros,
            V_g_idx,
            V_bits,
            VS,
        )
        # Extract post-dropout X for use in backward computation
        _dropped_X = []
        for _dropout in [
            dropout_Q, dropout_K, dropout_V
        ]:
            if _dropout is not None:
                # then matmul_lora must have attached the X
                _dropped_X.append(_dropout.X)
                del _dropout.X # remove it
            else:
                # otherwise will use X
                _dropped_X.append(X)
        ctx.save_for_backward(
            X,
            QA,
            QB,
            KA,
            KB,
            VA,
            VB,
            *_dropped_X
        )
        return Q, K, V

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dQ, dK, dV):
        (
            Q_qweight,
            Q_scales,
            Q_qzeros,
            Q_g_idx,
            Q_bits,
            QS,
            K_qweight,
            K_scales,
            K_qzeros,
            K_g_idx,
            K_bits,
            KS,
            V_qweight,
            V_scales,
            V_qzeros,
            V_g_idx,
            V_bits,
            VS,
        ) = ctx.custom_saved_tensors
        (
            X,
            QA,
            QB,
            KA,
            KB,
            VA,
            VB,
            QX, 
            KX, 
            VX,
        ) = ctx.saved_tensors

        QA, QB, KA, KB, VA, VB = QA.t(), QB.t(), KA.t(), KB.t(), VA.t(), VB.t()

        batch, seq_len, hd = X.shape
        dQ = dQ.view(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1])  # view doesn't work on K.T
        dV = dV.view(-1, dV.shape[-1])
        X = X.view(-1, X.shape[-1])
        dtype = X.dtype

        ### Weight projection LoRA weights
        # See our blogpost for more details.

        # Q Projection
        d_QA = QX.t() @ (dQ @ QB.t())
        d_QB = (QA.t() @ QX.t()) @ dQ
        d_QA *= QS
        d_QB *= QS

        # K Projection
        d_KA = KX.t() @ (dK @ KB.t())
        d_KB = (KA.t() @ KX.t()) @ dK
        d_KA *= KS
        d_KB *= KS

        # V Projection
        d_VA = VX.t() @ (dV @ VB.t())
        d_VB = (VA.t() @ VX.t()) @ dV
        d_VA *= VS
        d_VB *= VS

        # Combine derivatives to find dX
        # dQ
        QW = dequant248(Q_qweight, Q_scales, Q_qzeros, Q_g_idx, Q_bits)
        dX = torch.matmul(dQ, QW.t())  # , out=X)
        del QW
        dX += (QX != 0) * (dQ @ QB.to(dtype).t() @ (QS * QA.to(dtype).t()))

        # dK
        KW = dequant248(K_qweight, K_scales, K_qzeros, K_g_idx, K_bits)
        dX += dK @ KW.t()
        del KW
        dX += (KX != 0) * (dK @ KB.to(dtype).t() @ (KS * KA.to(dtype).t()))

        # dV
        VW = dequant248(V_qweight, V_scales, V_qzeros, V_g_idx, V_bits)
        dX += dV @ VW.t()
        del VW
        dX += (VX != 0) * (dV @ VB.to(dtype).t() @ (VS * VA.to(dtype).t()))

        # Q_qweight, Q_scales, Q_qzeros, Q_wf, Q_g_idx, Q_bits, Q_bias, QA, QB, QS,
        # K_qweight, K_scales, K_qzeros, K_wf, K_g_idx, K_bits, K_bias, KA, KB, KS,
        # V_qweight, V_scales, V_qzeros, V_wf, V_g_idx, V_bits, V_bias, VA, VB, VS,
        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            None,
            None,
            None,
            None,
            d_QA.t(),
            d_QB.t(),
            None,  # d_QS.t(),
            None,
            None,
            None,
            None,
            None,
            None,
            d_KA.t(),
            d_KB.t(),
            None,  # d_KS.t(),
            None,
            None,
            None,
            None,
            None,
            None,
            d_VA.t(),
            d_VB.t(),
            None,
            None,
            None,
            None,
        )


def apply_lora_qkv(self, X):
    Qqstate, Q_bias, QA, QB, QS, Qdropout = get_lora_parameters(self.q_proj)
    Kqstate, K_bias, KA, KB, KS, Kdropout = get_lora_parameters(self.k_proj)
    Vqstate, V_bias, VA, VB, VS, Vdropout = get_lora_parameters(self.v_proj)
    Q, K, V = LoRA_QKV.apply(
        X,
        *unpack_gptqstate(Qqstate),
        Q_bias,
        QA,
        QB,
        QS,
        *unpack_gptqstate(Kqstate),
        K_bias,
        KA,
        KB,
        KS,
        *unpack_gptqstate(Vqstate),
        V_bias,
        VA,
        VB,
        VS,
        Qdropout,
        Kdropout,
        Vdropout,
    )
    return Q, K, V


class LoRA_W(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(
        ctx,
        X: torch.Tensor,
        O_qweight,
        O_scales,
        O_qzeros,
        O_g_idx,
        O_bits,
        O_bias,
        A,
        B,
        S,
        dropout_O=None,
    ):
        W = dequant248(O_qweight, O_scales, O_qzeros, O_g_idx, O_bits)
        XW = matmul_lora(X, W, A, B, S, dropout=dropout_O)
        if O_bias is not None: XW += O_bias
        del W
        ctx.custom_saved_tensors = (
            O_qweight,
            O_scales,
            O_qzeros,
            O_g_idx,
            O_bits,
            S,
        )
        # Extract post-dropout X for use in backward computation
        if dropout_O is not None:
            _dropped_X = dropout_O.X
            del dropout_O.X
        else:
            _dropped_X = X
        ctx.save_for_backward(A, B, X, _dropped_X)
        return XW

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dY: torch.Tensor):
        O_qweight, O_scales, O_qzeros, O_g_idx, O_bits, S = ctx.custom_saved_tensors
        A, B, X, OX = ctx.saved_tensors

        A, B = A.t(), B.t()

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1])  # Must be reshape
        X = X.reshape(-1, X.shape[-1])  # Must be reshape
        dtype = X.dtype

        ### Weight projection LoRA weights
        # Weight projection
        d_A = OX.t() @ (dY @ B.t())
        d_B = (A.t() @ OX.t()) @ dY
        d_A *= S
        d_B *= S

        # Get derivative for dX
        W = dequant248(O_qweight, O_scales, O_qzeros, O_g_idx, O_bits)
        dX = dY @ W.t()
        del W
        dX += (OX !=0) * (dY @ B.to(dtype).t() @ (S * A.to(dtype).t()))

        # O_qweight, O_scales, O_qzeros, O_wf, O_g_idx, O_bits, O_bias, A, B, S
        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            None,
            None,
            None,
            None, 
            d_A.t(),
            d_B.t(),
            None,
            None,
        )


def apply_lora_o(self, X):
    Oqstate, O_bias, OA, OB, OS, dropout = get_lora_parameters(self.o_proj)
    O = LoRA_W.apply(X, *unpack_gptqstate(Oqstate), O_bias, OA, OB, OS, dropout)
    return O

# added by flim@sg.ibm.com
# this version can be directly patched on the output linear
def apply_lora_o_v2(self, X):
    Oqstate, O_bias, OA, OB, OS, dropout = get_lora_parameters(self)
    O = LoRA_W.apply(X, *unpack_gptqstate(Oqstate), O_bias, OA, OB, OS, dropout)
    return O

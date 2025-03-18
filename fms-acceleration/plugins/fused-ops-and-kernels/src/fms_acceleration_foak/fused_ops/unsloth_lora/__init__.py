# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Local
from .geglu import (
    geglu_approx_backward_kernel,
    geglu_approx_forward_kernel,
    geglu_exact_backward_kernel,
    geglu_exact_forward_kernel,
)
from .swiglu import swiglu_DWf_DW_dfg_kernel, swiglu_fg_kernel
from .utils import (
    QUANT_STATE,
    fast_dequantize,
    fast_gemv,
    fast_linear_forward,
    matmul_lora,
)

import pytest
import torch

from mixing.models.mixllama import MixLlamaConfig, MixLlamaForCausalLM

dtype = torch.float16
simple_arch = dict(
    hidden_size=128,
    intermediate_size=1024,
    num_hidden_layers=4,
    num_attention_heads=8,
    torch_dtype=dtype
)

test_cases = [
    # (name, cpu_ok, config)
    ("Basic LLaMa (no mixture)", True, 
        MixLlamaConfig(
            moe_mlp=False,
            output_router_logits=False,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MixLLaMa with MLP MOE", True,
        MixLlamaConfig(
            num_local_experts=4,
            output_router_logits=True,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MixLLaMa with MLP MOE always-on expert", True,
        MixLlamaConfig(
            num_experts_per_tok=2,
            num_local_experts=4,
            always_on_idx=0,
            output_router_logits=True,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MixLLaMa with QUERY MOE", True,
        MixLlamaConfig(
            num_experts_per_tok=2,
            num_local_experts=4,
            moe_query=True,
            output_router_logits=True,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MixLLaMa with ALL MOE", True,
        MixLlamaConfig(
            num_experts_per_tok=2,
            num_local_experts=4,
            moe_query=True,
            moe_key=True,
            moe_value=True,
            output_router_logits=True,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MixLLaMa with ALL MOE with flash attn", False,
        MixLlamaConfig(
            num_experts_per_tok=2,
            num_local_experts=4,
            moe_query=True,
            moe_key=True,
            moe_value=True,
            output_router_logits=True,
            pretraining_tp=False,
            _attn_implementation="flash_attention_2",
            **simple_arch
        )),
]

@pytest.fixture(ids=[x[0] for x in test_cases], params=test_cases)
def config(request):
    return request.param

def test_mix(config):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dummy = torch.ones((1,8)).int().to(device)

    name, cpu_ok, model_config = config
    if not torch.cuda.is_available(): assert cpu_ok, "test requires gpu"
    m = MixLlamaForCausalLM(model_config).to(dtype).to(device)
    o = m(dummy)
    o = m.generate(dummy)
    del m

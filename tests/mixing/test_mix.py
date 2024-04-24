import gc, shutil
import pytest
import torch

from mixing.mix import mix

from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM
)

dtype = torch.float16
TMP_DIR = "./m1"
simple_arch = dict(
    hidden_size=128,
    intermediate_size=1024,
    num_hidden_layers=4,
    num_attention_heads=8,
    torch_dtype=dtype
)

test_cases=  [
    # (name, cpu_ok, modules_to_mix, model_class, config)
    ("mistral mlp mix", True, ["mlp"],
        MistralForCausalLM, MistralConfig(
            **simple_arch
        )),
    ("Llama mlp mix", True, ["mlp"],
        LlamaForCausalLM, LlamaConfig(
            **simple_arch
        )),
    ("Llama mlp query mix", True, ["mlp", "q_proj"],
        LlamaForCausalLM, LlamaConfig(
            **simple_arch
        )),
    ("Llama mlp all mix", True, ["mlp", "q_proj", "k_proj", "v_proj"],
        LlamaForCausalLM, LlamaConfig(
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
    
    name, cpu_ok, modules_to_mix, cls, model_config = config
    if not torch.cuda.is_available(): assert cpu_ok, "test requires gpu"
    m = cls(model_config)
    m.save_pretrained("m1")
    del m
    gc.collect()
    m = mix(TMP_DIR, [TMP_DIR]*4, modules_to_mix)
    del m
    m = AutoModelForCausalLM.from_pretrained(TMP_DIR, trust_remote_code=True)
    m.to(device)(dummy)
    del m
    shutil.rmtree(TMP_DIR)
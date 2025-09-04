# Supported models list

- Legend:

  ✅ Ready and available 

  ✔️ Ready and available - compatible architecture (*see first bullet point above)

  🚫 Not supported

  ? May be supported, but not tested

Model Name & Size  | Model Architecture | Full Finetuning | Low Rank Adaptation (i.e. LoRA) | qLoRA(quantized LoRA) | 
-------------------- | ---------------- | --------------- | ------------------------------- | --------------------- |
[Granite 4.0 Tiny Preview](https://huggingface.co/ibm-granite/granite-4.0-tiny-preview) | GraniteMoeHybridForCausalLM | ✅ | ✅ | ? |
[Granite PowerLM 3B](https://huggingface.co/ibm-research/PowerLM-3b) | GraniteForCausalLM | ✅ | ✅ | ✅ |
[Granite 3.1 1B](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-base)       | GraniteForCausalLM | ✔️ | ✔️ | ✔️ |
[Granite 3.1 2B](https://huggingface.co/ibm-granite/granite-3.1-2b-base)             | GraniteForCausalLM | ✔️ | ✔️ | ✔️ |
[Granite 3.1 8B](https://huggingface.co/ibm-granite/granite-3.1-8b-base)       | GraniteForCausalLM | ✔️ | ✔️ | ✔️ |
[Granite 3.0 2B](https://huggingface.co/ibm-granite/granite-3.0-2b-base)       | GraniteForCausalLM | ✔️ | ✔️ | ✔️ |
[Granite 3.0 8B](https://huggingface.co/ibm-granite/granite-3.0-8b-base)       | GraniteForCausalLM | ✅ | ✅ | ✔️ |
[GraniteMoE 1B](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-base)        | GraniteMoeForCausalLM  | ✅ | ✅* | ? |
[GraniteMoE 3B](https://huggingface.co/ibm-granite/granite-3.0-3b-a800m-base)        | GraniteMoeForCausalLM  | ✅ | ✅* | ? |
[Granite 3B Code](https://huggingface.co/ibm-granite/granite-3b-code-base-2k)           | LlamaForCausalLM      | ✅ | ✔️  | ✔️ | 
[Granite 8B Code](https://huggingface.co/ibm-granite/granite-8b-code-base-4k)           | LlamaForCausalLM      | ✅ | ✅ | ✅ |
Granite 13B          | GPTBigCodeForCausalLM  | ✅ | ✅ | ✔️  | 
Granite 20B          | GPTBigCodeForCausalLM  | ✅ | ✔️  | ✔️  | 
[Granite 34B Code](https://huggingface.co/ibm-granite/granite-34b-code-instruct-8k)            | GPTBigCodeForCausalLM  | 🚫 | ✅ | ✅ | 
[Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)          | LlamaForCausalLM               | ✅** | ✔️ | ✔️ |  
[Llama3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)(same architecture as llama3) | LlamaForCausalLM   | 🚫 - same as Llama3-70B | ✔️  | ✔️ | 
[Llama3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)                            | LlamaForCausalLM   | 🚫 | 🚫 | ✅ | 
[Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)                               | LlamaForCausalLM   | ✅ | ✅ | ✔️ |  
[Llama3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)                             | LlamaForCausalLM   | 🚫 | ✅ | ✅ |
aLLaM-13b                                 | LlamaForCausalLM |  ✅ | ✅ | ✅ |
[Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)                              | MixtralForCausalLM   | ✅ | ✅ | ✅ |
[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)                                  | MistralForCausalLM   | ✅ | ✅ | ✅ |  
Mistral large                             | MistralForCausalLM   | 🚫 | 🚫 | 🚫 | 
[GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)                                  | GptOssForCausalLM   | ✅ | ✅ | ? |  
[GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b)                                  | GptOssForCausalLM   | ✅ | ✅ | ? |  

(*) - Supported for q,k,v,o layers . `all-linear` target modules does not infer on vLLM yet.

(**) - Supported from platform up to 8k context length - same architecture as llama3-8b.

### Supported vision model

We also support full fine-tuning and LoRA tuning for vision language models - `Granite 3.2 Vision`, `Llama 3.2 Vision`, and `LLaVa-Next` from `v2.8.1` onwards.
For information on supported dataset formats and how to tune a vision-language model, please see [this document](./vision-language-model-tuning.md).

Model Name & Size  | Model Architecture | LoRA Tuning | Full Finetuning |
-------------------- | ---------------- | --------------- | --------------- |
Llama 3.2-11B Vision  | MllamaForConditionalGeneration | ✅ | ✅ |
Llama 3.2-90B Vision  | MllamaForConditionalGeneration | ✔️ | ✔️ |
Granite 3.2-2B Vision  | LlavaNextForConditionalGeneration | ✅ | ✅ |
Llava Mistral 1.6-7B  | LlavaNextForConditionalGeneration | ✅ | ✅ |
Llava 1.6-34B  | LlavaNextForConditionalGeneration | ✔️ | ✔️ |
Llava 1.5-7B  | LlavaForConditionalGeneration | ✅ | ✅ |
Llava 1.5-13B  | LlavaForConditionalGeneration | ✔️ | ✔️ |

**Note**:
* vLLM currently does not support inference with LoRA-tuned vision models. To use a tuned LoRA adapter of vision model, please merge it with the base model before running vLLM inference.
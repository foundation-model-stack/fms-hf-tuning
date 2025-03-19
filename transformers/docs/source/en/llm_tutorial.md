<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Text generation

[[open-in-colab]]

Text generation is the most popular application for large language models (LLMs). A LLM is trained to generate the next word (token) given some initial text (prompt) along with its own generated outputs up to a predefined length or when it reaches an end-of-sequence (`EOS`) token.

In Transformers, the [`~GenerationMixin.generate`] API handles text generation, and it is available for all models with generative capabilities.

This guide will show you the basics of text generation with [`~GenerationMixin.generate`] and some common pitfalls to avoid.

## Default generate

Before you begin, it's helpful to install [bitsandbytes](https://hf.co/docs/bitsandbytes/index) to quantize really large models to reduce their memory usage.

```bash
!pip install -U transformers bitsandbytes
```
Bitsandbytes supports multiple backends in addition to CUDA-based GPUs. Refer to the multi-backend installation [guide](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend) to learn more.

Load a LLM with [`~PreTrainedModel.from_pretrained`] and add the following two parameters to reduce the memory requirements.

- `device_map="auto"` enables Accelerates' [Big Model Inference](./models#big-model-inference) feature for automatically initiating the model skeleton and loading and dispatching the model weights across all available devices, starting with the fastest device (GPU).
- `quantization_config` is a configuration object that defines the quantization settings. This examples uses bitsandbytes as the quantization backend (see the [Quantization](./quantization/overview) section for more available backends) and it loads the model in [4-bits](./quantization/bitsandbytes).

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", quantization_config=quantization_config)
```

Tokenize your input, and set the [`~PreTrainedTokenizer.padding_side`] parameter to `"left"` because a LLM is not trained to continue generation from padding tokens. The tokenizer returns the input ids and attention mask.

> [!TIP]
> Process more than one prompt at a time by passing a list of strings to the tokenizer. Batch the inputs to improve throughput at a small cost to latency and memory.

```py
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
```

Pass the inputs to [`~GenerationMixin.generate`] to generate tokens, and [`~PreTrainedTokenizer.batch_decode`] the generated tokens back to text.

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
"A list of colors: red, blue, green, yellow, orange, purple, pink,"
```

## Generation configuration

All generation settings are contained in [`GenerationConfig`]. In the example above, the generation settings are derived from the `generation_config.json` file of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1). A default decoding strategy is used when no configuration is saved with a model.

Inspect the configuration through the `generation_config` attribute. It only shows values that are different from the default configuration, in this case, the `bos_token_id` and `eos_token_id`.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
model.generation_config
GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2
}
```

You can customize [`~GenerationMixin.generate`] by overriding the parameters and values in [`GenerationConfig`]. Some of the most commonly adjusted parameters are [max_new_tokens](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.max_new_tokens), [num_beams](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.num_beams), [do_sample](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.do_sample), and [num_return_sequences](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.num_return_sequences).

```py
# enable beam search sampling strategy
model.generate(**inputs, num_beams=4, do_sample=True)
```

[`~GenerationMixin.generate`] can also be extended with external libraries or custom code. The `logits_processor` parameter accepts custom [`LogitsProcessor`] instances for manipulating the next token probability distribution. `stopping_criteria` supports custom [`StoppingCriteria`] to stop text generation. Check out the [logits-processor-zoo](https://github.com/NVIDIA/logits-processor-zoo) for more examples of external [`~GenerationMixin.generate`]-compatible extensions.

Refer to the [Generation strategies](./generation_strategies) guide to learn more about search, sampling, and decoding strategies.

### Saving

Create an instance of [`GenerationConfig`] and specify the decoding parameters you want.

```py
from transformers import AutoModelForCausalLM, GenerationConfig

model = AutoModelForCausalLM.from_pretrained("my_account/my_model")
generation_config = GenerationConfig(
    max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
)
```

Use [`~GenerationConfig.save_pretrained`] to save a specific generation configuration and set the `push_to_hub` parameter to `True` to upload it to the Hub.

```py
generation_config.save_pretrained("my_account/my_model", push_to_hub=True)
```

Leave the `config_file_name` parameter empty. This parameter should be used when storing multiple generation configurations in a single directory. It gives you a way to specify which generation configuration to load. You can create different configurations for different generative tasks (creative text generation with sampling, summarization with beam search) for use with a single model.

```py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

translation_generation_config = GenerationConfig(
    num_beams=4,
    early_stopping=True,
    decoder_start_token_id=0,
    eos_token_id=model.config.eos_token_id,
    pad_token=model.config.pad_token_id,
)

translation_generation_config.save_pretrained("/tmp", config_file_name="translation_generation_config.json", push_to_hub=True)

generation_config = GenerationConfig.from_pretrained("/tmp", config_file_name="translation_generation_config.json")
inputs = tokenizer("translate English to French: Configuration files are easy to use!", return_tensors="pt")
outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

## Pitfalls

The section below covers some common issues you may encounter during text generation and how to solve them.

### Output length

[`~GenerationMixin.generate`] returns up to 20 tokens by default unless otherwise specified in a models [`GenerationConfig`]. It is highly recommended to manually set the number of generated tokens with the [`max_new_tokens`] parameter to control the output length. [Decoder-only](https://hf.co/learn/nlp-course/chapter1/6?fw=pt) models returns the initial prompt along with the generated tokens.

```py
model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")
```

<hfoptions id="output-length">
<hfoption id="default length">

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'
```

</hfoption>
<hfoption id="max_new_tokens">

```py
generated_ids = model.generate(**model_inputs, max_new_tokens=50)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

</hfoption>
</hfoptions>

### Decoding strategy

The default decoding strategy in [`~GenerationMixin.generate`] is *greedy search*, which selects the next most likely token, unless otherwise specified in a models [`GenerationConfig`]. While this decoding strategy works well for input-grounded tasks (transcription, translation), it is not optimal for more creative use cases (story writing, chat applications).

For example, enable a [multinomial sampling](./generation_strategies#multinomial-sampling) strategy to generate more diverse outputs. Refer to the [Generation strategy](./generation_strategies) guide for more decoding strategies.

```py
model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to("cuda")
```

<hfoptions id="decoding">
<hfoption id="greedy search">

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

</hfoption>
<hfoption id="multinomial sampling">

```py
generated_ids = model.generate(**model_inputs, do_sample=True)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

</hfoption>
</hfoptions>

### Padding side

Inputs need to be padded if they don't have the same length. But LLMs aren't trained to continue generation from padding tokens, which means the [`~PreTrainedTokenizer.padding_side`] parameter needs to be set to the left of the input.

<hfoptions id="padding">
<hfoption id="right pad">

```py
model_inputs = tokenizer(
    ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
).to("cuda")
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 33333333333'
```

</hfoption>
<hfoption id="left pad">

```py
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model_inputs = tokenizer(
    ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
).to("cuda")
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 3, 4, 5, 6,'
```

</hfoption>
</hfoptions>

### Prompt format

Some models and tasks expect a certain input prompt format, and if the format is incorrect, the model returns a suboptimal output. You can learn more about prompting in the [prompt engineering](./tasks/prompting) guide.

For example, a chat model expects the input as a [chat template](./chat_templating). Your prompt should include a `role` and `content` to indicate who is participating in the conversation. If you try to pass your prompt as a single string, the model doesn't always return the expected output.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-alpha", device_map="auto", load_in_4bit=True
)
```

<hfoptions id="format">
<hfoption id="no format">

```py
prompt = """How many cats does it take to change a light bulb? Reply as a pirate."""
model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
input_length = model_inputs.input_ids.shape[1]
generated_ids = model.generate(**model_inputs, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
"Aye, matey! 'Tis a simple task for a cat with a keen eye and nimble paws. First, the cat will climb up the ladder, carefully avoiding the rickety rungs. Then, with"
```

</hfoption>
<hfoption id="chat template">

```py
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many cats does it take to change a light bulb?"},
]
model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
input_length = model_inputs.shape[1]
generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
"Arr, matey! According to me beliefs, 'twas always one cat to hold the ladder and another to climb up it an’ change the light bulb, but if yer looking to save some catnip, maybe yer can
```

</hfoption>
</hfoptions>

## Resources

Take a look below for some more specific and specialized text generation libraries.

- [Optimum](https://github.com/huggingface/optimum): an extension of Transformers focused on optimizing training and inference on specific hardware devices
- [Outlines](https://github.com/dottxt-ai/outlines): a library for constrained text generation (generate JSON files for example).
- [SynCode](https://github.com/uiuc-focal-lab/syncode): a library for context-free grammar guided generation (JSON, SQL, Python).
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference): a production-ready server for LLMs.
- [Text generation web UI](https://github.com/oobabooga/text-generation-webui): a Gradio web UI for text generation.
- [logits-processor-zoo](https://github.com/NVIDIA/logits-processor-zoo): additional logits processors for controlling text generation.
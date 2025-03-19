<!--Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Qwen2.5-VL

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The [Qwen2.5-VL](https://qwenlm.github.io/blog/qwen2_5-vl/) model is an update to [Qwen2-VL](https://arxiv.org/abs/2409.12191) from Qwen team, Alibaba Group. 

The abstract from this update is the following:

*Qwen2.5-VL marks a major step forward from Qwen2-VL, built upon the latest Qwen2.5 LLM. We've accelerated training and testing through the strategic implementation of window attention within the ViT. The ViT architecture itself has been refined with SwiGLU and RMSNorm, aligning it more closely with the LLM's structure. A key innovation is the expansion of native dynamic resolution to encompass the temporal dimension, in addition to spatial aspects. Furthermore, we've upgraded MRoPE, incorporating absolute time alignment on the time axis to allow the model to effectively capture temporal dynamics, regardless of frame rate, leading to superior video understanding.*

## Usage example

### Single Media inference

The model can accept both images and videos as input. Here's an example code for inference.

```python

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Load the model in half-precision on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


conversation = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)


# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)

# Video
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "text", "text": "What happened in the video?"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    video_fps=1,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

### Batch Mixed Media Inference

The model can batch inputs composed of mixed samples of various types such as images, videos, and text. Here is an example.

```python
# Conversation for the first image
conversation1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image1.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# Conversation with two images
conversation2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image2.jpg"},
            {"type": "image", "path": "/path/to/image3.jpg"},
            {"type": "text", "text": "What is written in the pictures?"}
        ]
    }
]

# Conversation with pure text
conversation3 = [
    {
        "role": "user",
        "content": "who are you?"
    }
]


# Conversation with mixed midia
conversation4 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image3.jpg"},
            {"type": "image", "path": "/path/to/image4.jpg"},
            {"type": "video", "path": "/path/to/video.jpg"},
            {"type": "text", "text": "What are the common elements in these medias?"},
        ],
    }
]

conversations = [conversation1, conversation2, conversation3, conversation4]
# Preparation for batch inference
ipnuts = processor.apply_chat_template(
    conversations,
    video_fps=1,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)


# Batch Inference
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

### Usage Tips

#### Image Resolution trade-off

The model supports a wide range of resolution inputs. By default, it uses the native resolution for input, but higher resolutions can enhance performance at the cost of more computation. Users can set the minimum and maximum number of pixels to achieve an optimal configuration for their needs.

```python
min_pixels = 224*224
max_pixels = 2048*2048
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
```

In case of limited GPU RAM, one can reduce the resolution as follows:

```python
min_pixels = 256*28*28
max_pixels = 1024*28*28 
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
```
This ensures each image gets encoded using a number between 256-1024 tokens. The 28 comes from the fact that the model uses a patch size of 14 and a temporal patch size of 2 (14 x 2 = 28).

#### Multiple Image Inputs

By default, images and video content are directly included in the conversation. When handling multiple images, it's helpful to add labels to the images and videos for better reference. Users can control this behavior with the following settings:

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"}, 
            {"type": "text", "text": "Hello, how are you?"}
        ]
    },
    {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking. How can I assist you today?"
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Can you describe these images and video?"}, 
            {"type": "image"}, 
            {"type": "image"}, 
            {"type": "video"}, 
            {"type": "text", "text": "These are from my vacation."}
        ]
    },
    {
        "role": "assistant",
        "content": "I'd be happy to describe the images and video for you. Could you please provide more context about your vacation?"
    },
    {
        "role": "user",
        "content": "It was a trip to the mountains. Can you see the details in the images and video?"
    }
]

# default:
prompt_without_id = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'


# add ids
prompt_with_id = processor.apply_chat_template(conversation, add_generation_prompt=True, add_vision_id=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?Picture 2: <|vision_start|><|image_pad|><|vision_end|>Picture 3: <|vision_start|><|image_pad|><|vision_end|>Video 1: <|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'

```

#### Flash-Attention 2 to speed up generation

First, make sure to install the latest version of Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

To load and run a model using FlashAttention-2, add `attn_implementation="flash_attention_2"` when loading the model:

```python
from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
)
```



## Qwen2_5_VLConfig

[[autodoc]] Qwen2_5_VLConfig

## Qwen2_5_VLProcessor

[[autodoc]] Qwen2_5_VLProcessor

## Qwen2_5_VLModel

[[autodoc]] Qwen2_5_VLModel
    - forward

## Qwen2_5_VLForConditionalGeneration

[[autodoc]] Qwen2_5_VLForConditionalGeneration
    - forward

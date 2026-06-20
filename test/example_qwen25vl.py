import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

device = "cuda:0"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/data/lyc/models/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device).eval()

processor = AutoProcessor.from_pretrained(
    "/data/lyc/models/Qwen2.5-VL-7B-Instruct",
    use_fast=False,  # 建议固定
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "./test/video/28s.mp4",
                "fps": 1.0,
                "max_pixels": 360 * 420,
            },
            {
                "type": "text",
                "text": "Please describe this video in detail."
            }
        ]
    }
]

# Qwen2.5 仍需要
image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages,
    return_video_kwargs=True,
)

print("video_kwargs =", video_kwargs)

# 与 Qwen3 风格尽量一致
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt",
    padding=True,
    **video_kwargs,
)

inputs = inputs.to(device)

print("input_ids:", inputs.input_ids.shape)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=False,
)

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print(output_text)
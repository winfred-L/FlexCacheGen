import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "/data/lyc/models/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
# )

device = "cuda:0"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/data/lyc/models/Qwen3-VL-8B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device).eval()

processor = AutoProcessor.from_pretrained("/data/lyc/models/Qwen3-VL-8B-Instruct")


video_fps = 1.0
# video_path = "/data/lyc/datasets/Video-MME/video/ZHWZf1Z4B5k.mp4" #28s
# video_path = "/data/lyc/datasets/Video-MME/video/zNxi2s36tS0.mp4" #43s
video_path = "/data1/lyc/datasets/MLVU/MLVU/video/9_summary/217.mp4.mp4" #66s
question = 'Please describe this video in detail.'
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/data1/lyc/datasets/MLVU/MLVU/video/9_summary/217.mp4",
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

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(output_text)
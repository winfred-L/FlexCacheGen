import os
from pathlib import Path
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def _load_dotenv():
    """Load .env from project root (nearest parent directory containing .env)."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        env_file = parent / '.env'
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key not in os.environ:
                        os.environ[key] = value
            return

_load_dotenv()

MODEL_ROOT = os.environ.get('MODEL_ROOT', '/data/lyc/models')
DATASET_ROOT = os.environ.get('DATASET_ROOT', '/data/lyc/datasets')
MODEL_PATH = os.path.join(MODEL_ROOT, 'Qwen3-VL-8B-Instruct')

# default: Load the model on the available device(s)
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     MODEL_PATH, dtype="auto", device_map="auto"
# )

device = "cuda:0"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device).eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)


video_fps = 1.0
video_path = os.path.join(DATASET_ROOT, 'Video-MME', 'video', 'ZHWZf1Z4B5k.mp4')  #28s
# video_path = os.path.join(DATASET_ROOT, 'Video-MME', 'video', 'zNxi2s36tS0.mp4')  #43s
# video_path = os.path.join(DATASET_ROOT, 'Video-MME', 'video', 'Z-rHofd6g2Q.mp4')  #66s
question = 'Please describe this video in detail.'
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420,
                "fps": video_fps,
            },
            {"type": "text", "text": question},
        ],
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
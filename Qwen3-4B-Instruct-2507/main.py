from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Path to your local model folder
# model_path = "D:/ai-models/numind--NuMarkdown-8B-Thinking"
model_path = "D:/ai/models/qwen--Qwen3-4B-Instruct-2507"

# Load tokenizer (local only, no internet)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

# Load model (use new `dtype`)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,      # Use float16 for GPU
    device_map="auto",        # Auto GPU/CPU placement
    local_files_only=True
)

# Chat messages
messages = [
    {"role": "user", "content": "ffmpeg command convert all videos in a specific folder,codec divx, resolution 640x480,saving the result on the same folder, using gpu if possible."},
]

# Prepare input
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# Generate output
outputs = model.generate(
    **inputs,
    max_new_tokens=4000,
    do_sample=True,
    temperature=0.7
)

# Decode result
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print("Model reply:")
print(response)

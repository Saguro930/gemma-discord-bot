from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "google/gemma-3-270m"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,   # CPU想定
    device_map="auto"
)

def generate_text(prompt: str, max_new_tokens: int = 120) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

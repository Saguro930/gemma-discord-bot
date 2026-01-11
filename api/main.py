from fastapi import FastAPI
from pydantic import BaseModel
from api.model import tokenizer, model

app = FastAPI()

class Prompt(BaseModel):
    text: str

@app.post("/generate")
def generate(prompt: Prompt):
    inputs = tokenizer(prompt.text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=120
    )
    return {
        "response": tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
    }

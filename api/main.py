import os
import threading
import requests
import discord
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import generate_text

# ===== FastAPI =====
app = FastAPI()

class Prompt(BaseModel):
    text: str

@app.post("/generate")
def generate(prompt: Prompt):
    return {
        "response": generate_text(prompt.text)
    }

# ===== Discord Bot =====
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
API_URL = "http://127.0.0.1:10000/generate"

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.content.startswith("!ai"):
        prompt = message.content[4:]
        res = requests.post(API_URL, json={"text": prompt}).json()
        await message.channel.send(res["response"][:1900])

# ===== 並列起動 =====
def run_api():
    uvicorn.run(app, host="0.0.0.0", port=10000)

def run_bot():
    client.run(DISCORD_TOKEN)

if __name__ == "__main__":
    threading.Thread(target=run_api, daemon=True).start()
    run_bot()

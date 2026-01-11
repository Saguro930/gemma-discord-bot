import discord
import requests
import os

TOKEN = os.environ["DISCORD_TOKEN"]
API_URL = os.environ["GEMMA_API_URL"]

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.content.startswith("!ai"):
        prompt = message.content[4:]
        res = requests.post(
            API_URL,
            json={"text": prompt}
        ).json()

        await message.channel.send(res["response"][:1900])

client.run(TOKEN)

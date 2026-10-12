import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")


async def main():
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    chat_completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "你好，请简单回复一句话。"}],
        model="gpt-5.6-sol",
    )
    print(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
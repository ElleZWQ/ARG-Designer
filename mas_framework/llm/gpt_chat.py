import os
import json
import aiohttp
import asyncio
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv

from mas_framework.llm.format import Message
from mas_framework.llm.price import cost_count
from mas_framework.llm.llm import LLM
from mas_framework.llm.llm_registry import LLMRegistry


load_dotenv()
MINE_BASE_URL = (os.getenv("BASE_URL") or "").rstrip("/")
MINE_API_KEY = os.getenv("API_KEY")


# Direct HTTP call (opens hard against the OpenAI-compatible /v1/chat/completions
# endpoint). We avoid the OpenAI SDK because some internal gateways reject the
# extra fields/headers the SDK injects (502), while a plain request succeeds.
async def _request_chat(
        model: str,
        messages: List[Dict],
        max_tokens: Optional[int],
        temperature: Optional[float],
):
    url = f"{MINE_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINE_API_KEY}",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    # GPT-5.6-Sol-joybuilder only accepts the default temperature; omit it
    # unless explicitly requested, otherwise the gateway rejects with 400.
    if temperature is not None and temperature != 0.2:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"LLM request failed ({resp.status}): {body[:500]}")
            data = await resp.json()
    return data["choices"][0]["message"]["content"]


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
        model: str,
        msg: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 0.2,
        num_comps: Optional[int] = 1,
):
    return await _request_chat(model, msg, max_tokens, temperature)


def chat(
        model: str,
        msg: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 0.2,
        num_comps: Optional[int] = 1,
):
    return asyncio.run(_request_chat(model, msg, max_tokens, temperature))


@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
            self,
            messages: List[Message],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        # Convert Message dataclasses to plain dicts if needed.
        msgs = [m if isinstance(m, dict) else {"role": m.role, "content": m.content} for m in messages]
        return await achat(self.model_name, msgs, max_tokens=max_tokens, temperature=temperature)

    def gen(
            self,
            messages: List[Message],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        msgs = [m if isinstance(m, dict) else {"role": m.role, "content": m.content} for m in messages]
        return chat(self.model_name, msgs, max_tokens=max_tokens, temperature=temperature)
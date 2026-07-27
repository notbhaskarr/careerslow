import asyncio
import logging
import re
from typing import AsyncIterator, List

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


async def ainvoke_with_retry(llm, messages: List[BaseMessage], max_retries: int = 4):
    """Invoke LLM with backoff on 429 rate-limit errors."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return await llm.ainvoke(messages)
        except Exception as e:
            last_error = e
            err = str(e)
            if "429" not in err and "ResourceExhausted" not in err and "quota" not in err.lower():
                raise
            if attempt >= max_retries - 1:
                raise
            delay = extract_retry_seconds(err)
            logger.warning(f"Gemini rate limit, retry {attempt + 1}/{max_retries} in {delay}s")
            await asyncio.sleep(delay)
    if last_error:
        raise last_error


async def astream_with_retry(llm, messages: List[BaseMessage], max_retries: int = 4) -> AsyncIterator:
    """Stream from Gemini with backoff on 429 rate-limit errors."""
    last_error = None
    for attempt in range(max_retries):
        try:
            async for chunk in llm.astream(messages):
                yield chunk
            return
        except Exception as e:
            last_error = e
            err = str(e)
            if "429" not in err and "ResourceExhausted" not in err and "quota" not in err.lower():
                raise
            if attempt >= max_retries - 1:
                raise
            delay = extract_retry_seconds(err)
            logger.warning(f"Gemini rate limit, retry {attempt + 1}/{max_retries} in {delay}s")
            await asyncio.sleep(delay)
    if last_error:
        raise last_error


def extract_retry_seconds(error_text: str) -> int:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_text, re.I)
    if match:
        return int(float(match.group(1))) + 2
    return 20

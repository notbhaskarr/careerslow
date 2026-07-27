import base64
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class SarvamTTSClient:
    URL = "https://api.sarvam.ai/text-to-speech"

    def __init__(self, api_key: str, language_code: str = "en-IN"):
        self.api_key = api_key
        self.language_code = language_code

    async def synthesize(self, text: str, client: httpx.AsyncClient) -> Optional[bytes]:
        payload = {
            "inputs": [text],
            "target_language_code": self.language_code,
            "speaker": "neha",
            "pace": 1.2,
            "speech_sample_rate": 16000,
            "enable_preprocessing": True,
            "model": "bulbul:v3",
        }
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }
        try:
            resp = await client.post(self.URL, json=payload, headers=headers, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
            if data.get("audios"):
                return base64.b64decode(data["audios"][0])
        except Exception as e:
            logger.error(f"TTS error: {e}")
        return None

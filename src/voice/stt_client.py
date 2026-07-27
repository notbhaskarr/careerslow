import asyncio
import base64
import json
import logging
import struct
from typing import Awaitable, Callable, Optional
from urllib.parse import urlencode

import websockets

logger = logging.getLogger(__name__)

TranscriptHandler = Callable[[str, bool], Awaitable[None]]
VadHandler = Callable[[str], Awaitable[None]]


def pcm_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw s16le mono PCM in a minimal WAV container for Sarvam."""
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + pcm


def parse_stt_message(resp: dict) -> tuple[Optional[str], Optional[str], bool]:
    """
    Parse Sarvam saaras:v3 WebSocket response.
    Returns (event_kind, text, is_final).
    event_kind: 'transcript', 'vad_start', 'vad_end', 'error', or None
    """
    msg_type = resp.get("type", "")
    data = resp.get("data") if isinstance(resp.get("data"), dict) else {}

    if msg_type == "error":
        err = (
            data.get("message")
            or data.get("error")
            or resp.get("message")
            or resp.get("error")
            or json.dumps(resp)
        )
        logger.error(f"Sarvam STT error: {err}")
        return "error", None, False

    if msg_type == "events":
        signal = data.get("signal_type", "")
        if signal == "START_SPEECH":
            return "vad_start", None, False
        if signal == "END_SPEECH":
            return "vad_end", None, False

    if msg_type == "data":
        text = data.get("transcript") or data.get("text")
        if text and text.strip():
            return "transcript", text.strip(), True

    # Legacy / alternate shapes
    if msg_type in ("transcript", "final_transcript"):
        text = data.get("transcript") or data.get("text") or resp.get("transcript")
        if text and text.strip():
            return "transcript", text.strip(), True

    text = data.get("transcript") or data.get("text") or resp.get("transcript") or resp.get("text")
    if text and text.strip():
        is_partial = data.get("is_partial") is True or resp.get("is_partial") is True
        if not is_partial:
            return "transcript", text.strip(), True

    return None, None, False


class SarvamSTTClient:
    BASE_URI = "wss://api.sarvam.ai/speech-to-text/ws"

    def __init__(self, api_key: str, language_code: str = "en-IN"):
        self.api_key = api_key
        self.language_code = language_code

    def _build_uri(self) -> str:
        params = urlencode({
            "language-code": self.language_code,
            "model": "saaras:v3",
            "mode": "transcribe",
            "sample_rate": "16000",
            "high_vad_sensitivity": "true",
            "vad_signals": "true",
            "flush_signal": "true",
            "input_audio_codec": "pcm_s16le",
        })
        return f"{self.BASE_URI}?{params}"

    async def stream(
        self,
        audio_queue,
        on_transcript: TranscriptHandler,
        on_vad: Optional[VadHandler] = None,
        stop_event: Optional[asyncio.Event] = None,
    ):
        pending_partial = ""
        uri = self._build_uri()
        headers = {"api-subscription-key": self.api_key}

        async with websockets.connect(
            uri,
            extra_headers=headers,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            logger.info("Sarvam STT connected (v3)")

            async def send_audio():
                while True:
                    if stop_event and stop_event.is_set():
                        break
                    data = await audio_queue.get()
                    wav = pcm_to_wav(data)
                    b64 = base64.b64encode(wav).decode("utf-8")
                    await ws.send(json.dumps({
                        "audio": {
                            "data": b64,
                            "sample_rate": 16000,
                            "encoding": "audio/wav",
                        }
                    }))

            async def receive():
                nonlocal pending_partial
                while True:
                    msg = await ws.recv()
                    try:
                        resp = json.loads(msg)
                    except json.JSONDecodeError:
                        logger.warning(f"Sarvam STT non-JSON: {msg[:200]}")
                        continue

                    kind, text, is_final = parse_stt_message(resp)

                    if kind == "vad_start":
                        if on_vad:
                            await on_vad("start")
                        continue

                    if kind == "vad_end":
                        if on_vad:
                            await on_vad("end")
                        try:
                            await ws.send(json.dumps({"type": "flush"}))
                        except Exception:
                            pass
                        if pending_partial.strip():
                            await on_transcript(pending_partial.strip(), True)
                            pending_partial = ""
                        continue

                    if kind == "transcript" and text:
                        logger.info(f"Sarvam STT transcript: {text}")
                        await on_transcript(text, is_final)
                        pending_partial = ""
                        continue

                    if kind == "error":
                        raise RuntimeError(f"Sarvam STT protocol error: {resp}")

            await asyncio.gather(send_audio(), receive())


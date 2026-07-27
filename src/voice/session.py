import asyncio
import logging
import os
import re
import time
import uuid
from typing import List, Optional, Tuple, Union

import httpx
from fastapi import WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.voice.llm_utils import astream_with_retry
from src.voice.protocol import STOP_PLAYBACK
from src.voice.stt_client import SarvamSTTClient
from src.voice.tts_client import SarvamTTSClient
from src.voice.turn_intents import (
    detect_meta_intent,
    is_filler_only,
)
from src.voice.turn_manager import TurnManager, TurnPhase
from src.voice.utterance_buffer import UtteranceBuffer

logger = logging.getLogger(__name__)

STT_LANGUAGE = os.getenv("INTERVIEW_STT_LANGUAGE", "en-IN")
TTS_LANGUAGE = os.getenv("INTERVIEW_TTS_LANGUAGE", "en-IN")
LLM_MODEL = os.getenv("INTERVIEW_LLM_MODEL", "gemini-3.1-flash-lite")
SILENCE_NUDGE_SECONDS = float(os.getenv("INTERVIEW_SILENCE_SECONDS", "12"))
UTTERANCE_DEBOUNCE_SECONDS = float(os.getenv("INTERVIEW_UTTERANCE_DEBOUNCE_MS", "3500")) / 1000.0
MIN_COMMIT_WORDS = int(os.getenv("INTERVIEW_MIN_COMMIT_WORDS", "5"))
SHORT_ANSWER_FALLBACK_SECONDS = float(os.getenv("INTERVIEW_SHORT_ANSWER_SECONDS", "4"))

LISTENING_DIRECTIVE = (
    "[INTERVIEWER DIRECTIVE — LISTENING CHECK]\n"
    "The candidate thinks you did not hear them. Apologize briefly, confirm you are listening, "
    "then repeat your last full question exactly. Max 30 words."
)

SKIP_DIRECTIVE_SUFFIX = (
    "[INTERVIEWER DIRECTIVE — SKIP REQUEST]\n"
    "The candidate asked to move to the next question. Acknowledge briefly in one phrase, "
    "then ask the next planned question. Max 25 words.\n"
)

LlmQueueItem = Union[str, Tuple[str, str]]


class InterviewSession:
    def __init__(
        self,
        websocket: WebSocket,
        pair_id: str,
        session_id: str,
        cache,
        system_prompt: str,
        plan: dict,
    ):
        self.websocket = websocket
        self.pair_id = pair_id
        self.session_id = session_id
        self.resume_id = pair_id  # legacy alias for logs
        self.cache = cache
        self.turns = TurnManager(plan)

        self.stt_in_queue: asyncio.Queue = asyncio.Queue()
        self.llm_in_queue: asyncio.Queue = asyncio.Queue()
        self.tts_in_queue: asyncio.Queue = asyncio.Queue()
        self.out_audio_queue: asyncio.Queue = asyncio.Queue()

        self.interrupted = asyncio.Event()
        self.stt_stop = asyncio.Event()
        self.pending_tts = 0
        self.last_activity = time.monotonic()
        self.nudge_sent = False
        self._answer_in_progress = False
        self._segment_answer_sent = False
        self._utterance = UtteranceBuffer()
        self._during_ai_buffer = UtteranceBuffer()
        self._commit_task: Optional[asyncio.Task] = None
        self._fallback_task: Optional[asyncio.Task] = None

        self.tasks: List[asyncio.Task] = []
        self.history = [SystemMessage(content=system_prompt)]
        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.4)

        sarvam_key = os.getenv("SARVAM_API_KEY", "")
        self.stt = SarvamSTTClient(sarvam_key, STT_LANGUAGE)
        self.tts = SarvamTTSClient(sarvam_key, TTS_LANGUAGE)

    async def run(self):
        self.tasks = [
            asyncio.create_task(self._browser_listener(), name="browser_listener"),
            asyncio.create_task(self._stt_worker(), name="stt_worker"),
            asyncio.create_task(self._llm_worker(), name="llm_worker"),
            asyncio.create_task(self._tts_worker(), name="tts_worker"),
            asyncio.create_task(self._audio_sender(), name="audio_sender"),
            asyncio.create_task(self._silence_watchdog(), name="silence_watchdog"),
        ]
        try:
            await self.tasks[0]
        finally:
            self._cleanup()

    def _format_transcript(self) -> str:
        lines = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                lines.append(f"Candidate: {msg.content}")
            elif isinstance(msg, AIMessage) and msg.content:
                lines.append(f"Interviewer: {msg.content}")
        return "\n".join(lines)

    def _persist_session_state(self):
        """Write transcript + meta to Redis so debrief can run before WS cleanup finishes."""
        if not self.cache:
            return
        meta = self.turns.get_session_meta()
        meta["session_id"] = self.session_id
        meta["pair_id"] = self.pair_id
        self.cache.set_transcript(self.session_id, self._format_transcript())
        self.cache.set_session_meta(self.session_id, meta)
        self.cache.link_session_pair(self.session_id, self.pair_id)

    def _cleanup(self):
        self.stt_stop.set()
        self._cancel_pending_commit()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        self._persist_session_state()

    def _last_ai_message(self) -> Optional[str]:
        for msg in reversed(self.history):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content.strip()
        return None

    def _planned_question_label(self, directive: str) -> str:
        seg = self.turns.current_segment()
        if "SILENCE NUDGE" in directive:
            return "(silence nudge)"
        if "CLOSE" in directive or "CLOSE]" in directive:
            return "(close interview)"
        if "REPEAT QUESTION" in directive:
            return seg.get("question", "") if seg else "(repeat last question)"
        if seg:
            return seg.get("question", "")
        return "(no planned question)"

    async def _interrupt_ai_playback(self):
        """Stop in-flight TTS/audio and tell the browser to halt playback."""
        self.interrupted.set()
        while True:
            try:
                self.tts_in_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while True:
            try:
                self.out_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.pending_tts = 0
        try:
            await self.websocket.send_json(STOP_PLAYBACK)
        except Exception as exc:
            logger.debug(f"stop_playback send failed: {exc}")

    async def _begin_barge_in(self):
        """User started speaking over the interviewer — cut AI audio and listen."""
        await self._interrupt_ai_playback()
        if not self._answer_in_progress:
            self._answer_in_progress = True
            self._drain_pending_answers()
        if self._during_ai_buffer.has_content():
            held = self._during_ai_buffer.joined().strip()
            self._during_ai_buffer.clear()
            if held:
                self._utterance.append(held)
        self.turns.on_user_started_speaking()
        logger.info("Barge-in: user speaking over AI")

    async def _browser_listener(self):
        try:
            while True:
                message = await self.websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    await self.stt_in_queue.put(message["bytes"])
        except WebSocketDisconnect:
            logger.info(f"Browser disconnected: {self.pair_id} session={self.session_id}")

    async def _stt_worker(self):
        if not os.getenv("SARVAM_API_KEY"):
            logger.error("No SARVAM_API_KEY")
            return

        async def on_transcript(text: str, is_final: bool):
            if is_final and text.strip():
                await self._on_user_transcript(text)

        async def on_vad(signal: str):
            if signal == "start":
                await self._on_vad_start()
            elif signal == "end":
                await self._on_vad_end()

        backoff = 1
        while not self.stt_stop.is_set():
            try:
                await self.stt.stream(
                    self.stt_in_queue,
                    on_transcript,
                    on_vad=on_vad,
                    stop_event=self.stt_stop,
                )
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.stt_stop.is_set():
                    break
                err = str(e)
                if "1000 (OK)" in err or "1001" in err:
                    break
                logger.error(f"STT worker error: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _on_vad_start(self):
        self._utterance.user_speaking = True
        self.last_activity = time.monotonic()
        self.nudge_sent = False
        self._cancel_pending_commit()
        if self.turns.phase in (TurnPhase.AWAITING_USER, TurnPhase.USER_SPEAKING):
            self.turns.on_user_started_speaking()
        elif self.turns.phase == TurnPhase.AI_SPEAKING:
            await self._begin_barge_in()

    async def _on_vad_end(self):
        self._utterance.user_speaking = False

        if self._during_ai_buffer.has_content():
            await self._process_during_ai_buffer()
            if self._answer_in_progress:
                return

        if self.turns.phase == TurnPhase.AI_SPEAKING and not self._answer_in_progress:
            return

        if not self._utterance.has_content():
            return
        self._schedule_commit()

    async def _on_user_transcript(self, text: str):
        if self.turns.phase == TurnPhase.CLOSED:
            return

        self.last_activity = time.monotonic()
        self.nudge_sent = False

        if self.turns.phase == TurnPhase.AI_SPEAKING:
            intent = detect_meta_intent(text)
            if intent:
                await self._interrupt_ai_playback()
                await self._handle_meta_intent(intent, text)
                return
            if not self._answer_in_progress:
                if is_filler_only(text):
                    self._during_ai_buffer.append(text)
                    logger.debug(f"User (held during AI): {text}")
                    return
                await self._begin_barge_in()
            self._utterance.append(text)
            self.turns.on_user_started_speaking()
            logger.debug(f"User (continue): {text}")
            return

        if self.turns.phase not in (TurnPhase.AWAITING_USER, TurnPhase.USER_SPEAKING):
            return

        if not self._answer_in_progress:
            self._answer_in_progress = True
            self._drain_pending_answers()

        self._utterance.append(text)
        self.turns.on_user_started_speaking()
        logger.debug(f"User (partial): {text}")

    async def _process_during_ai_buffer(self):
        if not self._during_ai_buffer.has_content():
            return

        text = self._during_ai_buffer.joined().strip()
        self._during_ai_buffer.clear()
        if not text:
            return

        intent = detect_meta_intent(text)
        if intent:
            await self._handle_meta_intent(intent, text)
            return

        if is_filler_only(text):
            logger.debug(f"Ignoring filler during AI: {text!r}")
            return

        await self._begin_barge_in()
        self._utterance.append(text)
        logger.info(f"User (answer started during AI): {text}")

    async def _handle_meta_intent(self, intent: str, text: str):
        logger.info(f"User (meta/{intent}): {text}")
        self._utterance.clear()
        self._during_ai_buffer.clear()
        self._answer_in_progress = False
        self._cancel_pending_commit()
        self._drain_pending_answers()
        self.turns.on_user_finished_speaking()

        if intent == "end":
            self.history.append(HumanMessage(content=text))
            self._persist_session_state()
            await self.llm_in_queue.put(
                "[INTERVIEWER DIRECTIVE — CLOSE]\n"
                "The candidate wants to end early. Thank them warmly and close in under 25 words."
            )
            return

        if intent in ("repeat", "listening"):
            self.history.append(HumanMessage(content=text))
            self._persist_session_state()
            last_ai = self._last_ai_message()
            if last_ai:
                await self._replay_interviewer(last_ai)
                return
            await self.llm_in_queue.put(("repeat", text))
            return

        if intent == "skip":
            self.history.append(HumanMessage(content=text))
            self._persist_session_state()
            self.turns.advance_segment()
            await self.llm_in_queue.put(("skip", text))
            return

    def _cancel_pending_commit(self):
        if self._commit_task and not self._commit_task.done():
            self._commit_task.cancel()
            self._commit_task = None
        if self._fallback_task and not self._fallback_task.done():
            self._fallback_task.cancel()
            self._fallback_task = None

    def _schedule_commit(self):
        self._cancel_pending_commit()
        self._commit_task = asyncio.create_task(self._debounced_commit())

    def _schedule_short_answer_fallback(self):
        if self._fallback_task and not self._fallback_task.done():
            return
        self._fallback_task = asyncio.create_task(self._short_answer_fallback())

    async def _debounced_commit(self):
        try:
            await asyncio.sleep(UTTERANCE_DEBOUNCE_SECONDS)
            if self._utterance.user_speaking:
                return
            if not self._utterance.has_content():
                return
            text = self._utterance.joined().strip()
            intent = detect_meta_intent(text)
            if intent:
                await self._handle_meta_intent(intent, text)
                return
            if self._utterance.word_count() < MIN_COMMIT_WORDS:
                logger.info(f"Deferring commit ({self._utterance.word_count()} words): {text!r}")
                self._schedule_short_answer_fallback()
                return
            await self._commit_user_turn()
        except asyncio.CancelledError:
            pass

    async def _short_answer_fallback(self):
        try:
            await asyncio.sleep(SHORT_ANSWER_FALLBACK_SECONDS)
            if self._utterance.user_speaking:
                return
            if not self._utterance.has_content():
                return
            text = self._utterance.joined().strip()
            intent = detect_meta_intent(text)
            if intent:
                await self._handle_meta_intent(intent, text)
                return
            if self._utterance.word_count() < MIN_COMMIT_WORDS:
                logger.info(f"Still too short to commit ({self._utterance.word_count()} words)")
                return
            await self._commit_user_turn()
        except asyncio.CancelledError:
            pass

    async def _commit_user_turn(self):
        text = self._utterance.joined().strip()
        if not text:
            return

        intent = detect_meta_intent(text)
        if intent:
            await self._handle_meta_intent(intent, text)
            return

        if self.turns.phase not in (
            TurnPhase.AWAITING_USER,
            TurnPhase.USER_SPEAKING,
            TurnPhase.AI_SPEAKING,
        ):
            return

        if self._segment_answer_sent:
            logger.info(f"User (continued, merged): {text}")
            if self.history and isinstance(self.history[-1], HumanMessage):
                merged = f"{self.history[-1].content} {text}".strip()
                self.history[-1] = HumanMessage(content=merged)
            else:
                self.history.append(HumanMessage(content=text))
            self._persist_session_state()
            self._utterance.clear()
            self._answer_in_progress = False
            return

        logger.info(f"User (commit): {text}")
        self._utterance.clear()
        self._answer_in_progress = False
        self._segment_answer_sent = True
        self.turns.on_user_finished_speaking()
        self._drain_pending_answers()
        self.turns.advance_segment()
        await self.llm_in_queue.put(("answer", text))

    def _drain_pending_answers(self):
        kept: List[LlmQueueItem] = []
        while not self.llm_in_queue.empty():
            try:
                item = self.llm_in_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, str) and item.startswith("["):
                kept.append(item)
        for item in kept:
            self.llm_in_queue.put_nowait(item)

    async def _replay_interviewer(self, text: str):
        """Re-speak the last interviewer line without calling the LLM."""
        self.interrupted.clear()
        self.turns.on_ai_started()
        logger.info(f"Replaying last interviewer line ({len(text)} chars)")
        logger.info("AI said (replay): %s", text)
        await self._enqueue_tts(text)

    async def _llm_worker(self):
        while True:
            item: LlmQueueItem = await self.llm_in_queue.get()
            directive: str
            user_message: Optional[str] = None
            skip_segment_mark = False

            kind = ""
            if isinstance(item, tuple):
                kind, text = item
                user_message = text
                if kind == "repeat":
                    directive = self.turns.build_directive(repeat_question=True)
                    skip_segment_mark = True
                elif kind == "skip":
                    directive = SKIP_DIRECTIVE_SUFFIX + self.turns.build_directive()
                    skip_segment_mark = False
                elif kind == "answer":
                    directive = self.turns.build_directive()
                else:
                    continue
            elif isinstance(item, str) and item.startswith("["):
                directive = item
            else:
                continue

            if self._answer_in_progress:
                logger.info("Deferring LLM — user answer in progress")
                await self.llm_in_queue.put(item)
                await asyncio.sleep(0.3)
                continue

            planned = self._planned_question_label(directive)
            logger.info(
                "AI turn [segment=%s]: planned=%r",
                self.turns.segment_index,
                planned,
            )

            if user_message:
                self.history.append(HumanMessage(content=user_message))
                self._persist_session_state()

            self.interrupted.clear()
            self.turns.on_ai_started()
            is_planned_question = (
                not directive.startswith("[INTERVIEWER DIRECTIVE — SILENCE NUDGE]")
                and not directive.startswith("[INTERVIEWER DIRECTIVE — REPEAT QUESTION]")
                and "SKIP REQUEST" not in directive
            )
            if is_planned_question and not skip_segment_mark:
                self.turns.mark_current_segment_asked()

            if user_message:
                messages_for_llm = list(self.history) + [SystemMessage(content=directive)]
            else:
                messages_for_llm = list(self.history) + [HumanMessage(content=directive)]

            buffer = ""
            full_response = ""

            try:
                async for chunk in astream_with_retry(self.llm, messages_for_llm):
                    if self.interrupted.is_set():
                        break
                    if self._answer_in_progress:
                        logger.info("Stopping LLM stream — user answer in progress")
                        break
                    if not chunk.content:
                        continue
                    buffer += chunk.content
                    full_response += chunk.content
                    match = re.search(r"(?<=[.!?])\s+", buffer)
                    if match:
                        sentence = buffer[: match.end()].strip()
                        if sentence:
                            await self._enqueue_tts(sentence)
                        buffer = buffer[match.end() :]

                if buffer.strip() and not self.interrupted.is_set() and not self._answer_in_progress:
                    await self._enqueue_tts(buffer.strip())

                if full_response and not self.interrupted.is_set() and not self._answer_in_progress:
                    self.history.append(AIMessage(content=full_response))
                    logger.info("AI said: %s", full_response)
                    self._persist_session_state()
                    if self.pending_tts == 0:
                        self._maybe_finish_ai_playback()

            except Exception as e:
                logger.error(f"LLM worker error: {e}")

    def _maybe_finish_ai_playback(self):
        if self._answer_in_progress or self._utterance.has_content():
            return
        if self._during_ai_buffer.has_content():
            asyncio.create_task(self._process_during_ai_buffer())
        self.turns.on_playback_finished()
        if self.turns.phase == TurnPhase.AWAITING_USER:
            self._segment_answer_sent = False
        self.last_activity = time.monotonic()
        self.nudge_sent = False

    async def _enqueue_tts(self, text: str):
        if self._answer_in_progress:
            return
        self.pending_tts += 1
        await self.tts_in_queue.put(text)

    def _decrement_pending_tts(self):
        self.pending_tts = max(0, self.pending_tts - 1)

    async def _tts_worker(self):
        async with httpx.AsyncClient() as client:
            while True:
                text = await self.tts_in_queue.get()
                if self.interrupted.is_set():
                    self._decrement_pending_tts()
                    continue
                audio = await self.tts.synthesize(text, client)
                if self.interrupted.is_set():
                    self._decrement_pending_tts()
                    continue
                if audio:
                    await self.out_audio_queue.put(audio)
                else:
                    self._decrement_pending_tts()

    async def _audio_sender(self):
        try:
            while True:
                data = await self.out_audio_queue.get()
                if self.interrupted.is_set():
                    self._decrement_pending_tts()
                    continue
                await self.websocket.send_bytes(data)
                self._decrement_pending_tts()
                if self.pending_tts == 0 and self.out_audio_queue.empty():
                    self._maybe_finish_ai_playback()
        except Exception as e:
            logger.error(f"Audio sender error: {e}")

    async def _silence_watchdog(self):
        try:
            while True:
                await asyncio.sleep(2)
                if not self.turns.should_nudge():
                    continue
                if (
                    self._answer_in_progress
                    or self._utterance.user_speaking
                    or self._utterance.has_content()
                    or self._during_ai_buffer.has_content()
                ):
                    continue
                if self.pending_tts > 0 or not self.out_audio_queue.empty():
                    continue
                if time.monotonic() - self.last_activity < SILENCE_NUDGE_SECONDS:
                    continue
                if self.nudge_sent:
                    continue
                self.nudge_sent = True
                logger.info("Silence nudge")
                await self.llm_in_queue.put(self.turns.build_directive(silence_nudge=True))
        except asyncio.CancelledError:
            pass

    async def start(self):
        self._persist_session_state()
        await self.websocket.send_json(
            {"type": "session_started", "session_id": self.session_id, "pair_id": self.pair_id}
        )
        await self.llm_in_queue.put(self.turns.build_directive())

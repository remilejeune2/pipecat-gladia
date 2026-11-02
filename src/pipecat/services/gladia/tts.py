#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gladia text-to-speech service implementation.

Gladia's realtime text to speech runs on the regional Enterprise API (eu1, us1):
``POST /v2/tts/live`` opens a session and returns its WebSocket, the text is streamed in
as the LLM writes it, and the audio comes back as binary PCM frames while it is
synthesized. See https://eu1.docs.gladia.io/text-to-speech/quickstart/.
"""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.gladia.stt import language_to_gladia_language
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven

# Gladia ends a session that sends nothing and receives no audio for 60 s.
KEEPALIVE_INTERVAL_S = 20.0


@dataclass
class GladiaTTSSettings(TTSSettings):
    """Settings for GladiaTTSService.

    Parameters:
        speed: The speaking rate, from 0.7 to 1.2. ``None`` keeps the voice's own (1.0).
        style: How to say it, in words (``"warm, reassuring"``). ``seldonia-1`` follows
            it, ``seldonia-1-small`` ignores it.
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    style: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class GladiaTTSService(WebsocketTTSService):
    """Gladia WebSocket text-to-speech service, on the regional Enterprise API.

    One Gladia session serves the whole conversation: each reply's text is sent as it
    comes, ``flush`` says the rest when the reply ends, and an interruption sends
    ``cancel``, which drops what was not said yet and keeps the session open.

    Gladia's audio carries no request ID, so replies play one after the other. A reply
    is over once Gladia has said all the text sent for it: the ``characters`` of its
    ``segment_end`` messages add up to the text, less its leading and trailing
    whitespace and the one character at each cut between two segments.
    """

    Settings = GladiaTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "https://eu1.api.gladia.io/v2/tts/live",
        sample_rate: int | None = None,
        max_chars: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Gladia TTS service.

        Args:
            api_key: Gladia Enterprise API key, with the ``tts`` scope.
            url: The session endpoint. Defaults to eu1
                (``"https://eu1.api.gladia.io/v2/tts/live"``); us1 is
                ``"https://us1.api.gladia.io/v2/tts/live"``.
            sample_rate: Audio sample rate in Hz: 8000, 16000, 22050, 24000, 44100 or
                48000. If None, uses the pipeline's.
            max_chars: The longest segment Gladia cuts the text into, from 20 to 1000
                characters. Defaults to Gladia's (250).
            settings: Runtime-updatable settings: ``model`` (``"seldonia-1"``,
                ``"seldonia-1-small"``, ``"seldonia-multi-1"``), ``voice`` (``"en-1"``,
                ``"en-2"``...; ``"default"`` is the model's first), ``language``,
                ``speed`` and ``style``. A change opens a new session.
            **kwargs: Additional arguments passed to parent WebsocketTTSService class.
        """
        default_settings = self.Settings(
            model="seldonia-1",
            voice="default",
            language=None,
            speed=None,
            style=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            pause_frame_processing=True,
            push_stop_frames=False,
            push_start_frame=True,
            append_trailing_space=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._max_chars = max_chars

        self._receive_task = None
        # the reply being said: its text, the characters and segments said, and
        # whether the LLM finished it
        self._reply_text = ""
        self._said_chars = 0
        self._said_segments = 0
        self._flushed = False
        # after a cancel, until Gladia's "cancelled": audio of the interrupted reply
        self._cancelling = False

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, as Gladia TTS service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Gladia's language code.

        Args:
            language: The language to convert.

        Returns:
            The Gladia language code.
        """
        return language_to_gladia_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta: the voice is set per session, so open a new one.

        Args:
            delta: A :class:`TTSSettings` (or ``GladiaTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed:
            await self._disconnect()
            await self._connect()
        return changed

    def _session_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "model": self._settings.model,
            "voice": self._settings.voice,
            "sample_rate": self.sample_rate,
            "encoding": "pcm_s16le",
            "output": "binary",
        }
        voice_settings: dict[str, Any] = {}
        if self._settings.speed is not None:
            voice_settings["speed"] = self._settings.speed
        if self._settings.style:
            voice_settings["style"] = self._settings.style
        if voice_settings:
            config["voice_settings"] = voice_settings
        if self._settings.language:
            config["language"] = self._settings.language
        if self._max_chars is not None:
            config["segmentation"] = {"max_chars": self._max_chars}
        return config

    async def _create_session(self) -> str:
        """Create a Gladia TTS session and return its WebSocket URL."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url,
                headers={"x-gladia-key": self._api_key},
                json=self._session_config(),
            ) as response:
                if response.status != 201:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to create a Gladia TTS session: {response.status} - {error_text}"
                    )
                created = await response.json()
                logger.debug(f"{self}: Gladia TTS session {created.get('id')} created")
                return created["url"]

    async def _connect(self):
        """Open a Gladia session and start the receive and keepalive tasks."""
        await super()._connect()

        # A session that ended closed its socket: its receive task is over too.
        if self._websocket is None or self._websocket.state is not State.OPEN:
            if self._receive_task:
                await self.cancel_task(self._receive_task)
                self._receive_task = None

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))
        self._create_keepalive_task(KEEPALIVE_INTERVAL_S)

    async def _disconnect(self):
        """Close the Gladia session and clean up tasks."""
        await super()._disconnect()

        await self._cancel_keepalive_task()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Create a Gladia session and connect to its WebSocket."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._reset_reply()
            self._cancelling = False
            url = await self._create_session()
            self._websocket = await self._websocket_connect(url)
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error_frame(ErrorFrame(error=f"{self} error: {e}", exception=e))
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket, which ends the Gladia session."""
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug(f"{self}: Disconnecting from Gladia TTS")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error_frame(ErrorFrame(error=f"{self} error: {e}"))
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get active websocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _send_keepalive(self):
        """Keep the session open while the agent listens."""
        await self._get_websocket().send(json.dumps({"type": "ping"}))

    def _reset_reply(self):
        self._reply_text = ""
        self._said_chars = 0
        self._said_segments = 0
        self._flushed = False

    def _reply_said(self) -> bool:
        if not self._flushed:
            return False
        # a segment keeps its inner whitespace; each cut drops the character it was
        # made at (a space or a newline)
        expected = len(self._reply_text.strip())
        return self._said_chars + max(self._said_segments - 1, 0) >= expected

    async def _end_reply_if_said(self):
        """End the reply's audio context once Gladia has said all its text."""
        if not self._reply_said():
            return
        ctx_id = self.get_active_audio_context_id()
        if ctx_id and self.audio_context_available(ctx_id):
            await self.append_to_audio_context(ctx_id, TTSStoppedFrame(context_id=ctx_id))
            await self.remove_audio_context(ctx_id)
        self._reset_reply()

    async def flush_audio(self, context_id: str | None = None):
        """Say the rest of the reply now: the LLM finished it.

        Args:
            context_id: Unused: Gladia says replies one after the other.
        """
        if not self._websocket:
            return
        try:
            await self._websocket.send(json.dumps({"type": "flush"}))
        except Exception as e:
            logger.error(f"{self} error sending flush message: {e}")
            return
        self._flushed = True
        await self._end_reply_if_said()

    async def on_audio_context_interrupted(self, context_id: str):
        """Drop what Gladia has not said yet of the interrupted reply.

        The session stays open: the next reply's text starts a new segment.

        Args:
            context_id: The ID of the audio context that was interrupted.
        """
        await self.stop_all_metrics()
        self._reset_reply()
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(json.dumps({"type": "cancel"}))
                self._cancelling = True
            except Exception as e:
                logger.error(f"{self} error sending cancel message: {e}")
        await super().on_audio_context_interrupted(context_id)

    async def _receive_messages(self):
        """Receive the audio and the session's messages from Gladia."""
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                if self._cancelling:
                    continue
                ctx_id = self.get_active_audio_context_id()
                frame = TTSAudioRawFrame(message, self.sample_rate, 1, context_id=ctx_id)
                await self.append_to_audio_context(ctx_id, frame)
                continue

            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.error(f"{self} invalid JSON message: {message}")
                continue

            msg_type = msg.get("type")
            if msg_type == "segment_end":
                if not self._cancelling:
                    self._said_chars += msg.get("characters", 0)
                    self._said_segments += 1
                    await self._end_reply_if_said()
            elif msg_type == "cancelled":
                self._cancelling = False
            elif msg_type == "segment_error":
                logger.warning(f"{self} Gladia could not say a segment: {msg}")
            elif msg_type == "error":
                if msg.get("code") == "invalid_message":
                    logger.warning(f"{self} Gladia ignored a message: {msg}")
                else:
                    await self.push_error(error_msg=f"{self} Gladia error: {msg}")
            elif msg_type == "end_session":
                logger.debug(f"{self} Gladia TTS session ended: {msg.get('status')}")
            elif msg_type == "enterprise.handoff":
                logger.debug(f"{self} Gladia TTS session moved to another worker: {msg}")
            else:
                logger.trace(f"{self} Gladia message: {msg}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send text to the Gladia session; its audio arrives in the receive task.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: None, the audio comes from the receive task; an ErrorFrame on failure.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            await self._get_websocket().send(json.dumps({"type": "text", "text": text}))
            self._reply_text += text
            await self.start_tts_usage_metrics(text)
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")

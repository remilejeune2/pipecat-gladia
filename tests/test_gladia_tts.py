#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for GladiaTTSService."""

import json
import re

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    ErrorFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.gladia.tts import GladiaTTSService
from pipecat.tests.utils import SleepFrame, run_test

CHUNK = b"\x01\x00" * 480


class FakeGladia:
    """Gladia's /v2/tts/live, cutting the text the way eu1 does.

    A segment ends after a sentence end and a space, or on a newline; the cut drops
    that whitespace and a segment keeps the rest of its text. With ``hold``, nothing
    is said until a ``cancel``.
    """

    def __init__(self, hold: bool = False) -> None:
        self.hold = hold
        self.configs: list[dict] = []
        self.messages: list[dict] = []
        app = web.Application()
        app.router.add_post("/v2/tts/live", self.create)
        app.router.add_get("/ws", self.connect)
        self.server = TestServer(app)

    @property
    def url(self) -> str:
        return str(self.server.make_url("/v2/tts/live"))

    async def create(self, request: web.Request) -> web.Response:
        assert request.headers["x-gladia-key"] == "test-key"
        self.configs.append(await request.json())
        url = self.server.make_url("/ws").with_scheme("ws")
        return web.json_response({"id": "s", "created_at": "", "url": str(url)}, status=201)

    async def connect(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "start_session"})
        pending = ""
        segment_id = 0
        async for msg in ws:
            data = json.loads(msg.data)
            self.messages.append(data)
            if data["type"] == "text":
                pending += data["text"]
            elif data["type"] == "flush" and not self.hold:
                for segment in re.split(r"(?<=[.?!;:])\s|\n", pending.strip()):
                    if not segment:
                        continue
                    await ws.send_json({"type": "segment_start", "segment_id": segment_id})
                    await ws.send_bytes(CHUNK)
                    await ws.send_json(
                        {
                            "type": "segment_end",
                            "segment_id": segment_id,
                            "characters": len(segment),
                        }
                    )
                    segment_id += 1
                pending = ""
            elif data["type"] == "cancel":
                pending = ""
                await ws.send_json({"type": "cancelled", "segment_ids": []})
        return ws


@pytest_asyncio.fixture
async def gladia():
    fake = FakeGladia()
    await fake.server.start_server()
    yield fake
    await fake.server.close()


@pytest.mark.asyncio
async def test_reply_ends_once_gladia_said_all_its_text(gladia):
    """The reply's stop frame comes from Gladia's segment counts, not a timeout."""
    tts = GladiaTTSService(
        api_key="test-key",
        url=gladia.url,
        sample_rate=24000,
        settings=GladiaTTSService.Settings(voice="en-2", speed=1.1, style="warm"),
    )

    down_frames, up_frames = await run_test(
        tts,
        frames_to_send=[
            TTSSpeakFrame(text="Sure. Tomorrow will be sunny in Paris,  with a breeze. Bye."),
            SleepFrame(sleep=0.3),
            BotStoppedSpeakingFrame(),
        ],
    )

    frame_types = [type(frame) for frame in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types
    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    audio = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert b"".join(frame.audio for frame in audio) == CHUNK * 3
    assert all(frame.sample_rate == 24000 for frame in audio)

    assert gladia.configs == [
        {
            "model": "seldonia-1",
            "voice": "en-2",
            "sample_rate": 24000,
            "encoding": "pcm_s16le",
            "output": "binary",
            "voice_settings": {"speed": 1.1, "style": "warm"},
        }
    ]
    assert [m["type"] for m in gladia.messages] == ["text", "flush"]


@pytest.mark.asyncio
async def test_interruption_cancels_and_keeps_the_session():
    """A barge-in sends cancel; the next reply goes on the same session."""
    gladia = FakeGladia(hold=True)
    await gladia.server.start_server()
    try:
        tts = GladiaTTSService(api_key="test-key", url=gladia.url, sample_rate=24000)
        await run_test(
            tts,
            frames_to_send=[
                TTSSpeakFrame(text="A long answer."),
                SleepFrame(sleep=0.2),
                InterruptionFrame(),
                SleepFrame(sleep=0.2),
            ],
        )
    finally:
        await gladia.server.close()

    assert "cancel" in [m["type"] for m in gladia.messages]
    assert len(gladia.configs) == 1
    assert not tts.get_audio_contexts()


def test_reply_said_counts_like_gladia():
    """eu1 staging, 2026-11-02: four segments of 5, 71, 14 and 4 characters."""
    tts = GladiaTTSService(api_key="test-key")
    tts._reply_text = (
        "Sure. Tomorrow will be sunny in Paris,  with a light breeze in the afternoon.\n"
        "Anything else? Bye. "
    )
    tts._flushed = True
    for characters in (5, 71, 14):
        tts._said_chars += characters
        tts._said_segments += 1
        assert not tts._reply_said()
    tts._said_chars += 4
    tts._said_segments += 1
    assert tts._reply_said()

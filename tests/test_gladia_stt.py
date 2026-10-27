#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EagerTranscriptionFrame,
    EndFrame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    UserTurnResponseCancelFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.gladia.config import EndOfTurnConfig
from pipecat.services.gladia.stt import GladiaSTTService
from pipecat.turns.user_turn_strategies import EagerUserTurnStrategies, ExternalUserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


class _FakeWebsocket:
    def __init__(self, *, state=State.OPEN, incoming=None):
        self.state = state
        self.sent = []
        self.closed = False
        self._incoming = incoming or []

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def close(self):
        self.closed = True
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._incoming:
            yield json.dumps(message)


def _connected_service(incoming=None, **kwargs):
    """Build a service holding an open fake socket, without touching the network."""
    service = GladiaSTTService(api_key="test-key", **kwargs)
    service._setup = frame_processor_setup(TaskManager())
    websocket = _FakeWebsocket(incoming=incoming)
    service._websocket = websocket
    service._connection_active = True
    return service, websocket


def _message_types(websocket):
    return [message["type"] for message in websocket.sent]


@pytest.mark.asyncio
async def test_stop_sends_stop_recording_while_the_socket_is_open():
    service, websocket = _connected_service()

    await service.stop(EndFrame())

    assert _message_types(websocket) == ["stop_recording"]
    assert websocket.closed


@pytest.mark.asyncio
async def test_stop_disconnects_once():
    service, _ = _connected_service()
    service._disconnect_websocket = AsyncMock(wraps=service._disconnect_websocket)

    await service.stop(EndFrame())

    service._disconnect_websocket.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_does_not_send_stop_recording():
    service, websocket = _connected_service()

    await service.cancel(CancelFrame())

    assert _message_types(websocket) == []
    assert websocket.closed


@pytest.mark.asyncio
async def test_audio_chunk_not_acknowledged_reports_error():
    """A rejected audio_chunk carries an ``error`` and no ``acknowledged: true``.

    The rejection is reported upstream as an error.
    """
    service, _ = _connected_service(
        incoming=[
            {
                "type": "audio_chunk",
                "acknowledged": False,
                "error": {"message": "quota exceeded"},
                "data": None,
            }
        ]
    )
    service.push_error = AsyncMock()

    await service._receive_messages()

    service.push_error.assert_awaited_once()
    assert "quota exceeded" in service.push_error.call_args.kwargs["error_msg"]


@pytest.mark.asyncio
async def test_translation_addon_error_reports_error_instead_of_crashing():
    """A failed translation addon call arrives with ``error`` set and ``data: null``.

    The failure is reported upstream as an error and the receive loop keeps
    running, so the connection is not torn down.
    """
    service, _ = _connected_service(
        incoming=[
            {
                "type": "translation",
                "error": {"message": "translation addon failed"},
                "data": None,
            }
        ]
    )
    service.push_error = AsyncMock()
    service.push_frame = AsyncMock()

    await service._receive_messages()

    service.push_error.assert_awaited_once()
    assert "translation addon failed" in service.push_error.call_args.kwargs["error_msg"]
    service.push_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_translation_without_error_still_pushes_translation_frame():
    """A translation with ``error: null`` pushes a ``TranslationFrame``."""
    service, _ = _connected_service(
        incoming=[
            {
                "type": "translation",
                "error": None,
                "data": {
                    "original_language": "en",
                    "translated_utterance": {"language": "fr", "text": "Bonjour"},
                },
            }
        ]
    )
    service.push_error = AsyncMock()
    service.push_frame = AsyncMock()

    await service._receive_messages()

    service.push_error.assert_not_awaited()
    service.push_frame.assert_awaited_once()
    pushed_frame = service.push_frame.call_args.args[0]
    assert pushed_frame.text == "Bonjour"


def _transcript(text, is_final):
    return {
        "type": "transcript",
        "data": {"is_final": is_final, "utterance": {"text": text, "language": "en"}},
    }


def _turn_service(incoming=None, end_of_turn=None, **kwargs):
    """Build a connected service with end of turn on, recording what it emits."""
    service, websocket = _connected_service(
        incoming=incoming,
        settings=GladiaSTTService.Settings(end_of_turn=end_of_turn or EndOfTurnConfig()),
        **kwargs,
    )
    service.broadcast_frame = AsyncMock()
    service.push_frame = AsyncMock()
    service._call_event_handler = AsyncMock()
    return service, websocket


def _broadcasts(service):
    return [call.args[0] for call in service.broadcast_frame.call_args_list]


def _pushed(service):
    return [call.args[0] for call in service.push_frame.call_args_list]


def _events(service):
    return [call.args[0] for call in service._call_event_handler.call_args_list]


def test_end_of_turn_is_off_by_default():
    service = GladiaSTTService(api_key="test-key")

    assert not service.end_of_turn_enabled
    assert "end_of_turn" not in service._prepare_settings()


def test_end_of_turn_options_are_sent_without_unset_fields():
    service = GladiaSTTService(
        api_key="test-key",
        settings=GladiaSTTService.Settings(
            end_of_turn=EndOfTurnConfig(eot_threshold=0.8, stop_s=1.2)
        ),
    )

    assert service.end_of_turn_enabled
    assert service._prepare_settings()["end_of_turn"] == {"eot_threshold": 0.8, "stop_s": 1.2}


def test_empty_end_of_turn_config_turns_on_the_server_defaults():
    service = GladiaSTTService(
        api_key="test-key", settings=GladiaSTTService.Settings(end_of_turn=EndOfTurnConfig())
    )

    assert service._prepare_settings()["end_of_turn"] == {}


def test_eager_end_of_turn_turns_on_end_of_turn_with_a_default_threshold():
    service = GladiaSTTService(api_key="test-key", enable_eager_end_of_turn=True)

    settings = service._prepare_settings()

    assert settings["end_of_turn"] == {"eager_eot_threshold": 0.5}
    assert settings["messages_config"] == {"receive_partial_transcripts": True}


def test_eager_end_of_turn_keeps_the_configured_thresholds():
    service = GladiaSTTService(
        api_key="test-key",
        enable_eager_end_of_turn=True,
        settings=GladiaSTTService.Settings(
            end_of_turn=EndOfTurnConfig(eot_threshold=0.9, eager_eot_threshold=0.7)
        ),
    )

    assert service._prepare_settings()["end_of_turn"] == {
        "eot_threshold": 0.9,
        "eager_eot_threshold": 0.7,
    }


def test_turn_strategies_follow_the_turn_mode():
    plain = GladiaSTTService(api_key="test-key")
    vad = GladiaSTTService(api_key="test-key", settings=GladiaSTTService.Settings(enable_vad=True))
    end_of_turn = GladiaSTTService(
        api_key="test-key", settings=GladiaSTTService.Settings(end_of_turn=EndOfTurnConfig())
    )
    eager = GladiaSTTService(api_key="test-key", enable_eager_end_of_turn=True)

    assert plain.service_metadata_frame().user_turn_strategies is None
    assert isinstance(vad.service_metadata_frame().user_turn_strategies, ExternalUserTurnStrategies)
    assert isinstance(
        end_of_turn.service_metadata_frame().user_turn_strategies, ExternalUserTurnStrategies
    )
    assert isinstance(eager.service_metadata_frame().user_turn_strategies, EagerUserTurnStrategies)


@pytest.mark.asyncio
async def test_end_of_turn_messages_are_ignored_without_end_of_turn():
    service, _ = _connected_service(
        incoming=[{"type": "speech_start", "data": {}}, {"type": "end_of_turn", "data": {}}]
    )
    service.broadcast_frame = AsyncMock()
    service.push_frame = AsyncMock()

    await service._receive_messages()

    service.broadcast_frame.assert_not_awaited()
    service.push_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_speech_start_proposes_the_start_of_the_turn_once():
    service, _ = _turn_service(
        incoming=[{"type": "speech_start", "data": {}}, {"type": "speech_start", "data": {}}]
    )

    await service._receive_messages()

    assert _broadcasts(service) == [ProposedUserStartedSpeakingFrame]


@pytest.mark.asyncio
async def test_speech_end_does_not_end_the_turn():
    """With end of turn, a pause is not the end of the turn, even with Gladia's VAD on."""
    service, _ = _turn_service(
        incoming=[{"type": "speech_start", "data": {}}, {"type": "speech_end", "data": {}}],
    )
    service._settings.enable_vad = True

    await service._receive_messages()

    assert _broadcasts(service) == [ProposedUserStartedSpeakingFrame]


@pytest.mark.asyncio
async def test_end_of_turn_proposes_the_end_of_the_turn():
    data = {"channel": 0, "probability": 0.97, "reason": "model", "end": 4.18}
    service, _ = _turn_service(
        incoming=[{"type": "speech_start", "data": {}}, {"type": "end_of_turn", "data": data}]
    )

    await service._receive_messages()

    assert _broadcasts(service) == [
        ProposedUserStartedSpeakingFrame,
        ProposedUserStoppedSpeakingFrame,
    ]
    assert _events(service) == ["on_end_of_turn"]
    assert service._call_event_handler.call_args.args[1] == data


@pytest.mark.asyncio
async def test_end_of_turn_without_a_turn_proposes_no_end():
    service, _ = _turn_service(incoming=[{"type": "end_of_turn", "data": {}}])

    await service._receive_messages()

    assert _broadcasts(service) == []
    assert _events(service) == ["on_end_of_turn"]


@pytest.mark.asyncio
async def test_the_final_after_an_end_of_turn_is_finalized(monkeypatch):
    """Gladia sends the final that closes the turn right after its end of turn."""
    service, _ = _turn_service(
        incoming=[
            {"type": "end_of_turn", "data": {"reason": "timeout"}},
            _transcript("Hello there.", True),
            _transcript("Next turn.", True),
        ]
    )
    del service.push_frame
    service.stop_ttfb_metrics = AsyncMock()
    parent_push_frame = AsyncMock()
    monkeypatch.setattr(FrameProcessor, "push_frame", parent_push_frame)

    await service._receive_messages()

    finals = [
        call.args[0]
        for call in parent_push_frame.call_args_list
        if isinstance(call.args[0], TranscriptionFrame)
    ]
    assert [(f.text, f.finalized) for f in finals] == [
        ("Hello there.", True),
        ("Next turn.", False),
    ]
    service.stop_ttfb_metrics.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_new_turn_proposes_its_start_again():
    service, _ = _turn_service(
        incoming=[
            {"type": "speech_start", "data": {}},
            {"type": "end_of_turn", "data": {}},
            {"type": "speech_start", "data": {}},
        ]
    )

    await service._receive_messages()

    assert _broadcasts(service) == [
        ProposedUserStartedSpeakingFrame,
        ProposedUserStoppedSpeakingFrame,
        ProposedUserStartedSpeakingFrame,
    ]


@pytest.mark.asyncio
async def test_eager_end_of_turn_pushes_the_turn_text_so_far():
    service, _ = _turn_service(
        incoming=[
            {"type": "speech_start", "data": {}},
            _transcript("My number is", True),
            _transcript("zero six", False),
            {"type": "eager_end_of_turn", "data": {"probability": 0.72}},
        ],
        enable_eager_end_of_turn=True,
    )

    await service._receive_messages()

    eager = _pushed(service)[-1]
    assert isinstance(eager, EagerTranscriptionFrame)
    assert eager.text == "My number is zero six"
    assert service.eager_end_of_turn_pending
    assert _events(service) == ["on_eager_end_of_turn"]


@pytest.mark.asyncio
async def test_eager_end_of_turn_prefers_the_text_gladia_sends():
    service, _ = _turn_service(
        incoming=[
            _transcript("book a", False),
            {"type": "eager_end_of_turn", "data": {"text": "book a table for two"}},
        ],
        end_of_turn=EndOfTurnConfig(speculation=True),
        enable_eager_end_of_turn=True,
    )

    await service._receive_messages()

    assert _pushed(service)[-1].text == "book a table for two"


@pytest.mark.asyncio
async def test_a_new_turn_starts_with_no_text():
    service, _ = _turn_service(
        incoming=[
            {"type": "speech_start", "data": {}},
            _transcript("First turn.", True),
            {"type": "end_of_turn", "data": {}},
            {"type": "speech_start", "data": {}},
            _transcript("second", False),
            {"type": "eager_end_of_turn", "data": {}},
        ],
        enable_eager_end_of_turn=True,
    )

    await service._receive_messages()

    assert _pushed(service)[-1].text == "second"


@pytest.mark.asyncio
async def test_turn_resumed_withdraws_the_eager_end_of_turn():
    service, _ = _turn_service(
        incoming=[
            _transcript("I want", False),
            {"type": "eager_end_of_turn", "data": {}},
            {"type": "turn_resumed", "data": {"end": 4.62}},
        ],
        enable_eager_end_of_turn=True,
    )

    await service._receive_messages()

    assert isinstance(_pushed(service)[-1], UserTurnResponseCancelFrame)
    assert not service.eager_end_of_turn_pending
    assert _events(service) == ["on_eager_end_of_turn", "on_turn_resumed"]


@pytest.mark.asyncio
async def test_end_of_turn_resolves_the_eager_end_of_turn_without_withdrawing_it():
    service, _ = _turn_service(
        incoming=[
            _transcript("Done", False),
            {"type": "eager_end_of_turn", "data": {}},
            {"type": "end_of_turn", "data": {}},
        ],
        enable_eager_end_of_turn=True,
    )

    await service._receive_messages()

    assert not service.eager_end_of_turn_pending
    assert not any(isinstance(f, UserTurnResponseCancelFrame) for f in _pushed(service))


@pytest.mark.asyncio
async def test_eager_end_of_turn_is_not_answered_unless_enabled():
    service, _ = _turn_service(
        incoming=[{"type": "eager_end_of_turn", "data": {}}],
        end_of_turn=EndOfTurnConfig(eager_eot_threshold=0.6),
    )

    await service._receive_messages()

    assert _pushed(service) == []
    assert not service.eager_end_of_turn_pending


@pytest.mark.asyncio
async def test_hold_is_reported_as_an_event():
    service, _ = _turn_service(incoming=[{"type": "hold", "data": {"kind": "phone_number"}}])

    await service._receive_messages()

    assert _events(service) == ["on_hold"]


@pytest.mark.asyncio
async def test_transcripts_still_reach_the_pipeline_with_end_of_turn():
    service, _ = _turn_service(
        incoming=[_transcript("Hel", False), _transcript("Hello.", True)],
    )

    await service._receive_messages()

    frames = _pushed(service)
    assert [type(f) for f in frames] == [InterimTranscriptionFrame, TranscriptionFrame]
    assert frames[1].text == "Hello."

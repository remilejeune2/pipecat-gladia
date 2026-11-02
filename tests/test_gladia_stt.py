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
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.gladia.config import (
    DiarizationConfig,
    EndOfTurnConfig,
    PreProcessingConfig,
    RealtimeProcessingConfig,
    SpeculationConfig,
)
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


def _skip_parent_process_frame(monkeypatch):
    """Let process_frame run only GladiaSTTService's own handling."""
    parent = next(c for c in GladiaSTTService.__mro__[1:] if "process_frame" in c.__dict__)
    monkeypatch.setattr(parent, "process_frame", AsyncMock())


@pytest.mark.asyncio
async def test_vad_end_of_speech_sends_finalize(monkeypatch):
    _skip_parent_process_frame(monkeypatch)
    service, websocket = _connected_service(enable_finalize=True)

    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert _message_types(websocket) == ["finalize"]


@pytest.mark.asyncio
async def test_finalize_is_off_by_default(monkeypatch):
    _skip_parent_process_frame(monkeypatch)
    service, websocket = _connected_service()

    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert _message_types(websocket) == []


@pytest.mark.asyncio
async def test_finalize_is_not_sent_when_gladia_ends_the_turn(monkeypatch):
    _skip_parent_process_frame(monkeypatch)
    service, websocket = _turn_service(enable_finalize=True)

    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert _message_types(websocket) == []


@pytest.mark.asyncio
async def test_the_final_from_finalize_is_finalized(monkeypatch):
    final = _transcript("Hello there.", True)
    final["data"]["from_finalize"] = True
    service, _ = _connected_service(
        incoming=[
            final,
            {"type": "finalize", "acknowledged": True, "data": {"finalized": True}},
            _transcript("Next turn.", True),
        ],
        enable_finalize=True,
    )
    service.request_finalize()
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
async def test_nothing_to_finalize_reports_the_latency_at_once():
    """Gladia's endpointing had already sent the final before the request."""
    service, _ = _connected_service(
        incoming=[{"type": "finalize", "acknowledged": True, "data": {"finalized": False}}],
        enable_finalize=True,
    )
    service.stop_ttfb_metrics = AsyncMock()

    await service._receive_messages()

    service.stop_ttfb_metrics.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_refused_finalize_waits_for_the_usual_final():
    service, _ = _connected_service(
        incoming=[
            {
                "type": "finalize",
                "acknowledged": False,
                "error": {"message": "finalize sent again within 200 ms"},
                "data": None,
            }
        ],
        enable_finalize=True,
    )
    service.stop_ttfb_metrics = AsyncMock()

    await service._receive_messages()

    service.stop_ttfb_metrics.assert_not_awaited()


def test_enterprise_options_are_sent_only_when_set():
    default = GladiaSTTService(api_key="test-key")._prepare_settings()
    for key in ("diarization", "diarization_config", "numerals", "emails", "addresses", "pii"):
        assert key not in default

    service = GladiaSTTService(
        api_key="test-key",
        url="https://eu1.api.gladia.io/v2/live",
        settings=GladiaSTTService.Settings(
            diarization=True,
            diarization_config=DiarizationConfig(max_speakers=2),
            numerals={"region": "CH"},
            pii={"entities": "pci", "mode": "mask"},
            pre_processing=PreProcessingConfig(speech_threshold="auto"),
            end_of_turn=EndOfTurnConfig(
                vad_threshold=0.6, speculation=SpeculationConfig(threshold=0.9, entities=True)
            ),
        ),
    )
    settings = service._prepare_settings()

    assert settings["diarization"] is True
    assert settings["diarization_config"] == {"max_speakers": 2}
    assert settings["numerals"] == {"region": "CH"}
    assert settings["pii"] == {"entities": "pci", "mode": "mask"}
    assert settings["pre_processing"] == {"speech_threshold": "auto"}
    assert settings["end_of_turn"] == {
        "vad_threshold": 0.6,
        "speculation": {"threshold": 0.9, "entities": True},
    }


def test_cloud_only_fields_are_left_out_on_the_enterprise_api():
    pre_processing = PreProcessingConfig(audio_enhancer=True, speech_threshold=0.4)
    realtime_processing = RealtimeProcessingConfig(words_accurate_timestamps=True)

    def settings_for(url):
        return GladiaSTTService(
            api_key="test-key",
            url=url,
            settings=GladiaSTTService.Settings(
                pre_processing=pre_processing, realtime_processing=realtime_processing
            ),
        )._prepare_settings()

    enterprise = settings_for("https://eu1.staging.api.gladia.io/v2/live")
    assert enterprise["pre_processing"] == {"speech_threshold": 0.4}
    assert "realtime_processing" not in enterprise

    cloud = settings_for("https://api.gladia.io/v2/live")
    assert cloud["pre_processing"] == {"audio_enhancer": True, "speech_threshold": 0.4}
    assert cloud["realtime_processing"] == {"words_accurate_timestamps": True}


@pytest.mark.asyncio
async def test_diarization_puts_the_speaker_in_the_user_id():
    def spoken(text, is_final, speaker):
        message = _transcript(text, is_final)
        message["data"]["utterance"]["speaker"] = speaker
        return message

    incoming = [spoken("Bon", False, 1), spoken("Bonjour.", True, 1)]
    service, _ = _connected_service(
        incoming=incoming, settings=GladiaSTTService.Settings(diarization=True)
    )
    service.push_frame = AsyncMock()
    await service._receive_messages()
    assert [f.user_id for f in _pushed(service)] == ["S1", "S1"]

    plain, _ = _connected_service(incoming=incoming)
    plain.push_frame = AsyncMock()
    await plain._receive_messages()
    assert [f.user_id for f in _pushed(plain)] == [plain._user_id] * 2


EU1 = "https://eu1.api.gladia.io/v2/live"


def _final_until(end):
    message = _transcript("Bonjour.", True)
    message["data"]["utterance"]["end"] = end
    return message


@pytest.mark.asyncio
async def test_enterprise_reconnect_opens_a_new_session(monkeypatch):
    """eu1's session URL carries a single-use token: a reconnect needs a new session."""
    sessions = iter([{"url": "wss://eu1/first", "id": "1"}, {"url": "wss://eu1/second", "id": "2"}])
    service, _ = _connected_service(url=EU1)
    monkeypatch.setattr(service, "_setup_gladia", AsyncMock(side_effect=lambda _: next(sessions)))
    connected = []

    async def connect(url):
        connected.append(url)
        return _FakeWebsocket()

    monkeypatch.setattr(service, "_websocket_connect", connect)
    monkeypatch.setattr(service, "_verify_connection", AsyncMock(return_value=True))
    await service._start_session()

    assert await service._reconnect_websocket(1)
    assert connected == ["wss://eu1/second"]
    assert service._session_id == "2"


@pytest.mark.asyncio
async def test_cloud_reconnect_rejoins_its_session(monkeypatch):
    service, _ = _connected_service()
    service._session_url = "wss://cloud/session"
    setup = AsyncMock()
    monkeypatch.setattr(service, "_setup_gladia", setup)
    connected = []

    async def connect(url):
        connected.append(url)
        return _FakeWebsocket()

    monkeypatch.setattr(service, "_websocket_connect", connect)
    monkeypatch.setattr(service, "_verify_connection", AsyncMock(return_value=True))

    assert await service._reconnect_websocket(1)
    assert connected == ["wss://cloud/session"]
    setup.assert_not_awaited()


@pytest.mark.asyncio
async def test_enterprise_final_drops_the_audio_it_transcribed():
    """The buffer keeps the open utterance only, for a new session to hear again."""
    service, _ = _connected_service(url=EU1, incoming=[_final_until(1.5)])
    service._sample_rate = 16000  # set when the pipeline starts
    service.push_frame = AsyncMock()
    service._audio_buffer = bytearray(b"\x01\x00" * 16000 * 2)  # 2 s sent

    await service._receive_messages()

    assert len(service._audio_buffer) == 16000  # the last 0.5 s


@pytest.mark.asyncio
async def test_enterprise_replay_sends_at_most_the_last_seconds_in_chunks():
    service, websocket = _connected_service(url=EU1)
    service._sample_rate = 16000
    service._audio_buffer = bytearray(b"\x01\x00" * 16000 * 20)  # 20 s without a final

    await service._send_buffered_audio()

    chunks = [m for m in websocket.sent if m["type"] == "audio_chunk"]
    assert len(chunks) == 15
    assert len(service._audio_buffer) == 2 * 16000 * 15

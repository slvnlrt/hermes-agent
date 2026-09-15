"""Feedback for explicit MEDIA directives withheld before post-stream delivery."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _event():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
    )
    return MessageEvent(
        text="send the file",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )


def _runner():
    return SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: None,
        _reply_anchor_for_event=lambda event: None,
    )


def _adapter():
    return SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        send=AsyncMock(),
        send_voice=AsyncMock(),
        send_document=AsyncMock(),
        send_image_file=AsyncMock(),
        send_video=AsyncMock(),
        send_multiple_images=AsyncMock(),
    )


def _background_runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda source: adapter
    runner._thread_metadata_for_source = lambda source, anchor=None: None
    runner._resolve_session_agent_runtime = lambda **kwargs: ("test-model", {"api_key": "test-key"})
    runner._resolve_turn_toolsets = lambda *args: ([], None)
    runner._provider_routing = {}
    runner._resolve_session_reasoning_config = lambda **kwargs: None
    runner._resolve_session_service_tier = lambda **kwargs: None
    runner._resolve_turn_agent_config = lambda *args: {"model": "test-model", "runtime": {}}
    runner._run_in_executor_with_context = _run_inline
    runner._cleanup_agent_resources = lambda agent: None
    runner._session_db = None
    runner._refresh_fallback_model = lambda: None
    return runner


async def _run_inline(fn):
    return fn()


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return media_file.resolve()


@pytest.mark.asyncio
async def test_emphasized_media_preserves_underscores_and_protected_text(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "result__final.pdf")
    protected_path = media_file.with_name("example__protected.pdf")
    response = (
        "Keep **ordinary __text__** unchanged.\n"
        f"`__MEDIA:{protected_path}__`\n"
        f"__MEDIA:{media_file}__"
    )

    media_files, cleaned = BasePlatformAdapter.extract_media(response)

    assert media_files == [(str(media_file), False)]
    assert "Keep **ordinary __text__** unchanged." in cleaned
    assert f"`__MEDIA:{protected_path}__`" in cleaned

    adapter = _adapter()
    await GatewayRunner._deliver_media_from_response(_runner(), response, _event(), adapter)

    adapter.send_document.assert_awaited_once()
    assert adapter.send_document.await_args.kwargs["file_path"] == str(media_file)
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_policy_dropped_media_sends_basename_only_feedback(tmp_path, monkeypatch):
    blocked = tmp_path / "outside" / "report.pdf"
    blocked.parent.mkdir()
    blocked.write_bytes(b"report")
    monkeypatch.setattr(
        "gateway.run_notifications._split_media_by_delivery_policy",
        lambda media_files, local_files: ([], [], [str(blocked)]),
    )
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _runner(), f"MEDIA:{blocked}", _event(), adapter,
    )

    adapter.send_document.assert_not_awaited()
    adapter.send.assert_awaited_once()
    notice = adapter.send.await_args.kwargs["content"]
    assert "report.pdf" in notice
    assert str(tmp_path) not in notice


@pytest.mark.asyncio
async def test_missing_media_sends_feedback(tmp_path, monkeypatch):
    missing = _allowed_media_path(tmp_path, monkeypatch, "placeholder.pdf").with_name("missing.pdf")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _runner(), f"MEDIA:{missing}", _event(), adapter,
    )

    adapter.send_document.assert_not_awaited()
    adapter.send.assert_awaited_once()
    assert "missing.pdf" in adapter.send.await_args.kwargs["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("withheld", "missing", "declined"))
async def test_background_attachment_failure_never_claims_success(tmp_path, monkeypatch, outcome):
    adapter = _adapter()
    if outcome == "withheld":
        media_path = tmp_path / "outside" / "withheld.pdf"
        media_path.parent.mkdir()
        media_path.write_bytes(b"media")
        monkeypatch.setattr(
            "gateway.run_notifications._split_media_by_delivery_policy",
            lambda media_files, local_files: ([], [], [str(media_path)]),
        )
    elif outcome == "missing":
        media_path = _allowed_media_path(tmp_path, monkeypatch, "placeholder.pdf").with_name("missing.pdf")
    else:
        media_path = _allowed_media_path(tmp_path, monkeypatch, "declined.pdf")
        adapter.send_document = AsyncMock(return_value=SendResult(success=False, error="connector declined"))

    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": f"MEDIA:{media_path}", "messages": []}
    with patch("gateway.run._load_gateway_config", return_value={}), patch(
        "run_agent.AIAgent", return_value=agent,
    ):
        await GatewayRunner._run_background_task_inner(
            _background_runner(adapter), "send the file", _event().source, "bg-test",
        )

    delivered = [call.kwargs["content"] for call in adapter.send.await_args_list]
    assert delivered
    assert all(not content.startswith("✅") for content in delivered)
    assert all(str(tmp_path) not in content for content in delivered)
    assert any(media_path.name in content for content in delivered)
    if outcome == "declined":
        adapter.send_document.assert_awaited_once()
    else:
        adapter.send_document.assert_not_awaited()

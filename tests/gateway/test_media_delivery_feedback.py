"""Delivery-failure feedback for MEDIA: markers (no more silent drops).

Two prod-lived failure modes around ``GatewayRunner._deliver_media_from_response``:

1. ``**MEDIA:/x.png**`` — markdown emphasis glued to the marker broke
   ``extract_media``'s extension anchor: no delivery attempted, marker left in
   the streamed text, zero feedback. Fixed by ``_unwrap_media_emphasis``.
2. Marker parsed but the path rejected by ``validate_media_delivery_path``
   (outside allowed roots in strict mode, denied prefix, missing file): the
   filter dropped it with only a server-side log line — clean message, no
   file, no explanation. Fixed by ``_notify_dropped_media``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner, _unwrap_media_emphasis
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


def _fake_runner(thread_meta=None):
    return SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )


def _fake_adapter():
    return SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(return_value=SendResult(success=True, message_id="text")),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="batch")),
    )


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return media_file.resolve()


def _force_strict_mode(monkeypatch):
    """Mirror prod: HERMES_MEDIA_DELIVERY_STRICT=1, no recency tolerance."""
    monkeypatch.setattr("gateway.platforms.base._media_delivery_strict_mode", lambda: True)
    monkeypatch.setattr("gateway.platforms.base._media_delivery_recency_seconds", lambda: 0)


# ---------------------------------------------------------------------------
# _unwrap_media_emphasis (unit)
# ---------------------------------------------------------------------------

class TestUnwrapMediaEmphasis:
    def test_bold_is_unwrapped(self):
        assert _unwrap_media_emphasis("voici : **MEDIA:/tmp/a.png**") == "voici : MEDIA:/tmp/a.png"

    def test_italic_star_is_unwrapped(self):
        assert _unwrap_media_emphasis("*MEDIA:/tmp/a.png*") == "MEDIA:/tmp/a.png"

    def test_double_underscore_is_unwrapped(self):
        assert _unwrap_media_emphasis("__MEDIA:/tmp/a.png__") == "MEDIA:/tmp/a.png"

    def test_bold_italic_is_unwrapped(self):
        assert _unwrap_media_emphasis("***MEDIA:/tmp/a.png***") == "MEDIA:/tmp/a.png"

    def test_underscore_in_filename_survives(self):
        text = "**MEDIA:/tmp/hermes_test_sylvain.mp3**"
        assert _unwrap_media_emphasis(text) == "MEDIA:/tmp/hermes_test_sylvain.mp3"

    def test_single_underscore_wrap_is_left_alone(self):
        # Single ``_`` is ambiguous with snake_case paths — deliberately skipped.
        text = "_MEDIA:/tmp/a.png_"
        assert _unwrap_media_emphasis(text) == text

    def test_backtick_code_span_is_left_alone(self):
        # Code spans are masked out of delivery on purpose (quoted examples);
        # unwrapping them would defeat that protection.
        text = "exemple : `MEDIA:/tmp/a.png`"
        assert _unwrap_media_emphasis(text) == text

    def test_text_without_media_untouched(self):
        text = "du **gras** normal sans marqueur"
        assert _unwrap_media_emphasis(text) == text

    def test_surrounding_text_keeps_its_emphasis(self):
        text = "**important** puis **MEDIA:/tmp/a.png** fin"
        assert _unwrap_media_emphasis(text) == "**important** puis MEDIA:/tmp/a.png fin"


# ---------------------------------------------------------------------------
# _deliver_media_from_response (integration, fake adapter)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bold_wrapped_allowed_media_is_delivered(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.flac")
    adapter = _fake_adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(), f"Voilà : **MEDIA:{media_file}**", _event(), adapter,
    )

    adapter.send_document.assert_awaited_once()
    assert adapter.send_document.await_args.kwargs["file_path"] == str(media_file)
    adapter.send.assert_not_awaited()  # no parasite notice


@pytest.mark.asyncio
async def test_blocked_path_sends_explicit_notice(tmp_path, monkeypatch):
    _force_strict_mode(monkeypatch)
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path / "cache-only",)
    )
    blocked = tmp_path / "outside" / "report.pdf"
    blocked.parent.mkdir(parents=True)
    blocked.write_bytes(b"%PDF fake")
    adapter = _fake_adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(), f"MEDIA:{blocked}", _event(), adapter,
    )

    adapter.send_document.assert_not_awaited()
    adapter.send.assert_awaited_once()
    notice = adapter.send.await_args.kwargs["content"]
    assert "Couldn't deliver" in notice
    assert "report.pdf" in notice
    assert str(tmp_path) not in notice  # host directories never leak


@pytest.mark.asyncio
async def test_bold_wrapped_blocked_path_also_notices(tmp_path, monkeypatch):
    # The two fixes compose: emphasis unwrapped first, THEN the filter drop
    # is surfaced (previously: unparsed marker, no extraction, no feedback).
    _force_strict_mode(monkeypatch)
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path / "cache-only",)
    )
    blocked = tmp_path / "pets" / "homelander.png"
    blocked.parent.mkdir(parents=True)
    blocked.write_bytes(b"\x89PNG fake")
    adapter = _fake_adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(), f"**MEDIA:{blocked}**", _event(), adapter,
    )

    adapter.send_multiple_images.assert_not_awaited()
    notice = adapter.send.await_args.kwargs["content"]
    assert "homelander.png" in notice


@pytest.mark.asyncio
async def test_missing_file_sends_notice(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,)
    )
    adapter = _fake_adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(), f"MEDIA:{tmp_path}/nope.mp3", _event(), adapter,
    )

    adapter.send_voice.assert_not_awaited()
    notice = adapter.send.await_args.kwargs["content"]
    assert "nope.mp3" in notice


@pytest.mark.asyncio
async def test_allowed_plain_path_no_notice(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "clip.mp4")
    adapter = _fake_adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(), f"MEDIA:{media_file}", _event(), adapter,
    )

    adapter.send_video.assert_awaited_once()
    adapter.send.assert_not_awaited()

"""Queued-follow-up lane: the fallback first-response send and the normal completion send share one
"already delivered" source of truth (#81052).

With streaming disabled (the shipped default) there is never a stream consumer, so the queued
lane always sends the first turn's final itself before running the follow-up. When the lane then
hands the FIRST turn's result back (the follow-up's text was refused), the normal completion path
must not send that text — or its attachments — a second time, and must still send them when the
queued lane's own send was refused.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import InjectionReceipt, MessageEvent, MessageType, SessionSource
from gateway.run import GatewayRunner
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner, _run_with_agent

_SESSION_KEY = "agent:main:telegram:group:-1001:17585"


class _DocCaptureAdapter(ProgressCaptureAdapter):
    """Capture document uploads so a double MEDIA delivery is visible."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.documents = []

    async def send_document(self, chat_id, file_path, caption=None, file_name=None,
                            reply_to=None, metadata=None, **kwargs) -> SendResult:
        self.documents.append(file_path)
        return SendResult(success=True, message_id="doc-1")


class _RefusingDocCaptureAdapter(_DocCaptureAdapter):
    """Every send is refused (flood control, retries exhausted): a SendResult, not an exception."""

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
        return SendResult(success=False, error="flood control", retryable=False)


def _agent_returning(text):
    class _Agent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            return {"final_response": text, "messages": [], "api_calls": 1}

    return _Agent


async def _none():
    return None


async def _run_with_refused_followup(monkeypatch, tmp_path, agent_cls, adapter_cls):
    """First turn answers, the queued follow-up's text is refused: the lane hands back the FIRST
    turn's result, which is the object the completion send then acts on."""
    monkeypatch.setattr(
        GatewayRunner, "_expand_inbound_context_references",
        lambda self, source, session_key, message_text: _none(),
    )
    return await _run_with_agent(
        monkeypatch, tmp_path, agent_cls, session_id="sess-refused-followup",
        pending_text="@file:/etc/shadow please", adapter_cls=adapter_cls,
    )


async def _completion_seam(adapter, agent_result, response):
    """Run the result through ``_hmwa_deliver_turn_response`` — the seam that decides whether the
    caller sends the body again. Returns the text the caller would send (None = suppressed)."""
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=adapter.platform, chat_id="-1001", chat_type="group", thread_id="17585",
    )
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="1")

    class _Entry:
        session_id = "s1"

    return await runner._hmwa_deliver_turn_response(
        event, source, _Entry(), _SESSION_KEY, None, agent_result, [], response, None, False,
    )


@pytest.mark.asyncio
async def test_queued_lane_delivers_text_and_media_exactly_once(monkeypatch, tmp_path):
    """Single delivery: the queued lane sends the text and uploads the attachment, and the
    completion seam re-sends neither."""
    doc = tmp_path / "report.pdf"
    doc.write_text("x", encoding="utf-8")
    final = f"answer 1\n\nMEDIA: {doc}"
    adapter, result = await _run_with_refused_followup(
        monkeypatch, tmp_path, _agent_returning(final), _DocCaptureAdapter,
    )

    assert [c["content"] for c in adapter.sent] == ["answer 1"]
    assert adapter.documents == [str(doc)]
    assert result["final_response"] == final

    assert await _completion_seam(adapter, result, final) is None
    assert len(adapter.sent) == 1
    assert adapter.documents == [str(doc)]


@pytest.mark.asyncio
async def test_refused_queued_send_leaves_the_completion_send_as_the_fallback(monkeypatch, tmp_path):
    """A refused queued send (``SendResult.success`` False) must NOT mark the turn delivered: the
    completion seam is the only remaining send, and suppressing it loses the answer entirely."""
    doc = tmp_path / "report.pdf"
    doc.write_text("x", encoding="utf-8")
    final = f"answer 1\n\nMEDIA: {doc}"
    adapter, result = await _run_with_refused_followup(
        monkeypatch, tmp_path, _agent_returning(final), _RefusingDocCaptureAdapter,
    )

    # Attempted (and retried) but never accepted by the platform.
    assert adapter.sent and {c["content"] for c in adapter.sent} == {"answer 1"}
    # The text never landed, so its attachments wait for the completion send too.
    assert adapter.documents == []
    assert not result.get("already_sent")

    assert await _completion_seam(adapter, result, final) == final


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_success", "footer", "expected_outcome"),
    [(False, None, "success"), (True, "turn footer", "failure")],
    ids=["refused-body-successful-fallback", "successful-body-refused-footer"],
)
async def test_refused_followup_receipt_waits_for_outer_delivery(
    initial_success, footer, expected_outcome,
):
    """A blocked follow-up leaves the opening injection responsible for its final obligations."""
    adapter = _DocCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", thread_id="17585",
    )
    outcomes = []
    event = MessageEvent(
        text="plugin wake", source=source, message_id="opening-injection",
        internal=True, allow_gateway_control=False,
    )
    event._injection_receipts.append(InjectionReceipt(outcomes.append))
    pending_event = MessageEvent(
        text="@file:/etc/shadow please", source=source, message_id="blocked-followup",
    )
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=None)
    runner._run_agent = AsyncMock()
    runner._pop_post_delivery_callback = lambda *_args: None
    ctx = SimpleNamespace(
        source=source, session_id="receipt-fallback", session_key=_SESSION_KEY, run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata=None,
        stream_consumer_holder=[None], event_message_id=None,
        inbound_message_id=event.message_id, _queued_receipt_owner=event,
        mute_notification_reply=False, persist_user_display_kind=None,
    )
    result = {"final_response": "opening answer", "messages": []}
    phase = "queued"
    sends = []
    before_completion = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sends.append((phase, content, list(outcomes)))
        success = initial_success if phase == "queued" else footer is None
        return SendResult(success=success, error=None if success else "delivery refused")

    async def handle(opening_event):
        nonlocal phase
        returned = await runner._run_agent_queued_followup(
            ctx, adapter, pending_event.text, pending_event, result, result, None,
        )
        before_completion.append(list(outcomes))
        phase = "completion"
        return await runner._hmwa_deliver_turn_response(
            opening_event, source, SimpleNamespace(session_id=ctx.session_id),
            _SESSION_KEY, 1, returned, [], returned["final_response"], footer, False,
        )

    adapter.send = AsyncMock(side_effect=send)
    adapter.set_message_handler(handle)
    adapter._active_sessions[_SESSION_KEY] = asyncio.Event()

    await adapter._process_message_background(event, _SESSION_KEY)

    runner._prepare_profile_scoped_inbound_message_text.assert_awaited_once()
    runner._run_agent.assert_not_awaited()
    assert before_completion == [[]]
    assert outcomes == [expected_outcome]
    assert all(not receipt_outcomes for _, _, receipt_outcomes in sends)
    assert [content for stage, content, _ in sends if stage == "completion"] == [
        footer if footer is not None else "opening answer",
    ]

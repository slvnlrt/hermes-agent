"""Tests for plugin-triggered turns in existing gateway sessions."""

import asyncio
import concurrent.futures
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    PlatformConfig,
    SendResult,
)
from gateway.platforms.event import InjectionReceipt, MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _entry(*, origin=True) -> SessionEntry:
    source = None
    if origin:
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
            user_id="42",
            user_name="tester",
        )
    now = datetime.now()
    return SessionEntry(
        session_key="agent:main:telegram:dm:42",
        session_id="session-42",
        created_at=now,
        updated_at=now,
        origin=source,
        platform=Platform.TELEGRAM,
    )


def _runner(entry: SessionEntry | None, adapter=None) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, lookup_by_session_key=AsyncMock(return_value=entry)
    )
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner._profile_adapters = {}
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._is_user_authorized = MagicMock(return_value=True)
    return runner


class _RoutingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("network send is not expected")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


@pytest.mark.asyncio
async def test_plugin_context_routes_through_live_gateway_to_existing_session(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({
            "plugins": {"entries": {"notify-plugin": {"allow_gateway_injection": True}}}
        })
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    source = _entry().origin
    entry = store.get_or_create_session(source)
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    pending_user_event = MessageEvent(
        text="human follow-up",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["human.jpg"],
        media_types=["image/jpeg"],
    )
    adapter._pending_messages[entry.session_key] = pending_user_event

    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._queued_events = {}
    runner._is_user_authorized = MagicMock(return_value=True)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="notify-plugin", key="notify-plugin", source="user"),
        manager,
    )

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._install_plugin_message_injector()
        assert (
            context.inject_message(
                "/approve always",
                session_key=entry.session_key,
            )
            is True
        )
        task = next(iter(runner._background_tasks))
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        assert adapter._pending_messages[entry.session_key] is pending_user_event
        queued = runner._queued_events[entry.session_key][0]
        assert pending_user_event.text == "human follow-up"
        assert pending_user_event.media_urls == ["human.jpg"]
        assert pending_user_event.allow_gateway_control is True
        assert queued.text == "/approve always"
        assert queued.allow_gateway_control is False
        assert queued.metadata["gateway_session_id"] == entry.session_id
        adapter._message_handler.assert_not_awaited()

        runner._clear_plugin_message_injector()
        assert manager.has_gateway_message_injector is False


@pytest.mark.asyncio
async def test_dispatch_uses_stored_origin_and_adapter_message_path():
    async def _accept(event):
        event._gateway_accepted = True

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_accept))
    entry = _entry()
    runner = _runner(entry, adapter)

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
        content="check the deployment",
        plugin_id="notify-plugin",
    )

    assert accepted is True
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "check the deployment"
    assert event.internal is True
    assert event.allow_gateway_control is False
    assert event.get_command() is None
    assert event.source == entry.origin
    assert event.source is not entry.origin
    runner._is_user_authorized.assert_called_once_with(
        event.source,
        allow_adapter_delegation=False,
    )
    assert event.metadata == {
        "hermes_plugin_id": "notify-plugin",
        "hermes_plugin_injection": True,
        "gateway_session_key": entry.session_key,
        "gateway_session_id": entry.session_id,
        "gateway_session_strict": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry", "with_adapter"),
    [
        (None, True),
        (_entry(origin=False), True),
        (_entry(), False),
    ],
)
async def test_dispatch_rejects_unroutable_session(entry, with_adapter):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(entry, adapter if with_adapter else None)

    accepted = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_dispatch_rechecks_current_authorization(raises):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)
    if raises:
        runner._is_user_authorized.side_effect = RuntimeError("config unavailable")
    else:
        runner._is_user_authorized.return_value = False

    accepted = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_rejects_stored_role_only_authorization(monkeypatch):
    """A stored adapter role grant must be revalidated against current core auth."""
    for key in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)

    adapter = MagicMock(spec=BasePlatformAdapter)
    adapter.handle_message = AsyncMock()
    entry = _entry()
    entry.session_key = "agent:main:discord:dm:42"
    entry.platform = Platform.DISCORD
    source = entry.origin
    assert source is not None
    source.platform = Platform.DISCORD
    source.role_authorized = True

    runner = _runner(entry)
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = GatewayConfig()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    del runner._is_user_authorized

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_stops_when_gateway_drains_during_lookup():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)
    lookup_started = asyncio.Event()
    release_lookup = asyncio.Event()

    async def _lookup(_session_key):
        lookup_started.set()
        await release_lookup.wait()
        return _entry()

    runner._async_session_store.lookup_by_session_key = _lookup
    dispatch = asyncio.create_task(
        runner._dispatch_plugin_message_injection(
            session_key="agent:main:telegram:dm:42",
            content="wake up",
            plugin_id="notify-plugin",
        )
    )

    await lookup_started.wait()
    runner._draining = True
    release_lookup.set()

    assert await dispatch is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_queues_non_control_plugin_text_for_exact_session():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    source = _entry().origin
    session_key = build_session_key(source)
    adapter._active_sessions[session_key] = asyncio.Event()
    event = MessageEvent(
        text="/approve always",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        allow_gateway_control=False,
        metadata={"gateway_session_key": session_key},
    )

    await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    assert adapter._pending_messages[session_key] is event
    assert adapter._active_sessions[session_key].is_set() is False


@pytest.mark.asyncio
async def test_base_adapter_rejects_derived_session_mismatch():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    event = MessageEvent(
        text="ordinary input",
        source=_entry().origin,
        internal=True,
        allow_gateway_control=False,
        metadata={"gateway_session_key": "agent:main:telegram:dm:other"},
    )

    await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    assert adapter._active_sessions == {}


@pytest.mark.asyncio
async def test_scheduler_submits_dispatch_on_live_gateway_loop():
    runner = _runner(_entry())
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_message_injection = AsyncMock(return_value=True)

    assert (
        runner._schedule_plugin_message_injection(
            session_key="agent:main:telegram:dm:42",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is True
    )

    await asyncio.sleep(0)
    dispatched = runner._dispatch_plugin_message_injection.await_args.kwargs
    assert {
        key: dispatched[key]
        for key in ("session_key", "content", "plugin_id")
    } == {
        "session_key": "agent:main:telegram:dm:42",
        "content": "wake up",
        "plugin_id": "notify-plugin",
    }


@pytest.mark.asyncio
async def test_scheduler_ignores_same_loop_task_cancellation():
    runner = _runner(_entry())
    loop = asyncio.get_running_loop()
    runner._gateway_loop = loop
    callback_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: callback_errors.append(context))

    blocker = asyncio.Event()

    async def _wait_for_cancellation(**_kwargs):
        await blocker.wait()

    runner._dispatch_plugin_message_injection = _wait_for_cancellation

    try:
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )

        task = next(iter(runner._background_tasks))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert callback_errors == []


@pytest.mark.asyncio
async def test_scheduler_logs_async_failure_without_callback_error(caplog):
    runner = _runner(_entry())
    loop = asyncio.get_running_loop()
    runner._gateway_loop = loop
    callback_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: callback_errors.append(context))
    runner._dispatch_plugin_message_injection = AsyncMock(
        side_effect=RuntimeError("adapter failed")
    )

    try:
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )
        task = next(iter(runner._background_tasks))
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert callback_errors == []
    assert "plugin=notify-plugin session=key" in caplog.text


def test_scheduler_uses_threadsafe_bridge_outside_gateway_loop():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    def _submit(coro, target_loop, **_kwargs):
        assert target_loop is loop
        coro.close()
        future = concurrent.futures.Future()
        future.set_result(True)
        return future

    with patch("gateway.run.safe_schedule_threadsafe", side_effect=_submit) as submit:
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )

    submit.assert_called_once()


def test_scheduler_ignores_threadsafe_future_cancellation():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    def _submit(coro, _target_loop, **_kwargs):
        coro.close()
        future = concurrent.futures.Future()
        future.cancel()
        return future

    with (
        patch("gateway.run.safe_schedule_threadsafe", side_effect=_submit),
        patch("gateway.run.logger.warning") as warning,
    ):
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )

    warning.assert_not_called()


def test_scheduler_rejects_stopped_or_closed_gateway():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop
    runner._running = False

    assert (
        runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is False
    )
    loop.call_soon_threadsafe.assert_not_called()

    runner._running = True
    runner._gateway_loop = None
    assert (
        runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is False
    )

    runner._gateway_loop = loop
    loop.is_closed.return_value = True
    assert (
        runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is False
    )
    loop.call_soon_threadsafe.assert_not_called()


def test_scheduler_rejects_submission_failure():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    def _reject(coro, _target_loop, **_kwargs):
        coro.close()
        return None

    with patch("gateway.run.safe_schedule_threadsafe", side_effect=_reject):
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is False
        )


def test_install_and_clear_gateway_injector_preserves_newer_owner():
    runner = _runner(_entry())
    manager = PluginManager()

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._install_plugin_message_injector()
        assert manager.has_gateway_message_injector is True

        runner._clear_plugin_message_injector()
        assert manager.has_gateway_message_injector is False

        runner._install_plugin_message_injector()

        newer_owner = MagicMock()
        newer_injector = MagicMock(return_value=True)
        manager.set_gateway_message_injector(newer_owner, newer_injector)
        runner._clear_plugin_message_injector()

    assert manager.has_gateway_message_injector is True
    assert manager.inject_gateway_message(value="kept") is True
    newer_injector.assert_called_once_with(value="kept")


# ── injection receipt lifecycle tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_fires_on_complete_not_routed_when_session_missing():
    """on_complete fires 'not_routed' when session lookup returns None."""
    runner = _runner(None)  # no session entry
    outcomes = []

    result = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:99",
        content="event",
        plugin_id="example-inbox",
        request_id="req-1",
        expires_at=10**12,
        on_complete=outcomes.append,
    )

    assert result is False
    assert outcomes == ["not_routed"]


@pytest.mark.asyncio
async def test_dispatch_fires_on_complete_not_routed_when_auth_denied():
    """on_complete fires 'not_routed' when authorization is denied."""
    runner = _runner(_entry())
    runner._is_user_authorized = MagicMock(return_value=False)
    outcomes = []

    result = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="event",
        plugin_id="example-inbox",
        request_id="req-2",
        expires_at=10**12,
        on_complete=outcomes.append,
    )

    assert result is False
    assert outcomes == ["not_routed"]


@pytest.mark.asyncio
async def test_dispatch_fires_on_complete_expired_before_dispatch():
    """on_complete fires 'expired' when the injection has expired by dispatch time."""
    runner = _runner(_entry())
    outcomes = []

    result = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="event",
        plugin_id="example-inbox",
        request_id="req-3",
        expires_at=1.0,  # well in the past
        on_complete=outcomes.append,
    )
    assert result is False
    assert outcomes == ["expired"]



def _receipt_event(callback, *, expires_at=None):
    event = MessageEvent(
        text="plugin wake",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="42", chat_type="dm",
            user_id="42", user_name="tester",
        ),
        internal=True,
        allow_gateway_control=False,
    )
    event._injection_receipts.append(InjectionReceipt(callback, expires_at))
    return event


@pytest.mark.asyncio
@pytest.mark.parametrize(("send_success", "outcome"), [(True, "success"), (False, "failure")])
async def test_process_receipt_follows_actual_send_result(send_success, outcome):
    adapter = _RoutingAdapter()
    adapter.send = AsyncMock(return_value=SendResult(success=send_success))
    adapter.set_message_handler(AsyncMock(return_value="delivered reply"))
    outcomes = []
    event = _receipt_event(outcomes.append)
    session_key = "agent:main:telegram:dm:42"
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter._process_message_background(event, session_key)

    assert adapter.send.await_count >= 1
    assert outcomes == [outcome]



@pytest.mark.asyncio
async def test_process_receipt_requires_every_multipart_delivery(monkeypatch):
    adapter = _RoutingAdapter()
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    adapter.send_document = AsyncMock(return_value=SendResult(success=False, error="upload declined"))
    adapter.set_message_handler(AsyncMock(return_value="Delivered text\nMEDIA:/tmp/report.pdf"))
    monkeypatch.setattr(
        BasePlatformAdapter,
        "filter_media_delivery_paths",
        staticmethod(lambda media_files, **_kwargs: media_files),
    )
    outcomes = []
    event = _receipt_event(outcomes.append)
    session_key = "agent:main:telegram:dm:42"
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter._process_message_background(event, session_key)

    adapter.send_document.assert_awaited_once()
    assert outcomes == ["failure"]


@pytest.mark.asyncio
async def test_busy_followup_receipt_follows_its_own_failed_delivery_not_next_success():
    """An in-band injection is closed before the next queued response can succeed."""
    source = _entry().origin
    assert source is not None
    session_key = "agent:main:telegram:dm:42"
    adapter = SimpleNamespace(
        _active_sessions={session_key: asyncio.Event()},
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        send=AsyncMock(side_effect=[SendResult(success=False), SendResult(success=True)]),
    )
    runner = object.__new__(GatewayRunner)
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._is_goal_continuation_event = lambda _event: False
    runner._session_key_for_source = lambda _source: session_key
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="A")
    runner._reply_anchor_for_event = lambda _event: None
    runner._adapter_for_source = lambda _source: None
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._run_agent_stream_confirmed_final_delivery = lambda *_args, **_kwargs: False
    runner._is_intentional_silence = lambda *_args, **_kwargs: False
    runner._pop_post_delivery_callback = lambda *_args, **_kwargs: None
    outcomes = []
    injection_a = _receipt_event(outcomes.append)
    injection_a.source = source
    injection_a.message_id = "A"
    root_ctx = SimpleNamespace(
        source=source, session_id="sid", session_key=session_key, run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata=None, context_prompt=None,
        result_holder=[{"interrupted": True, "messages": []}],
    )

    async def _run_child(**kwargs):
        a_ctx = SimpleNamespace(
            source=source, session_key=session_key, run_generation=1,
            stream_consumer_holder=[None], _status_thread_metadata=None, event_message_id=None,
            inbound_message_id="A", _queued_receipt_owner=kwargs["receipt_owner"],
        )
        await GatewayRunner._run_agent_deliver_first_response(
            runner, a_ctx, adapter, {"final_response": "A response"}, {"messages": []}, None,
        )
        b_ctx = SimpleNamespace(
            source=source, session_key=session_key, run_generation=1,
            stream_consumer_holder=[None], _status_thread_metadata=None, event_message_id=None,
            inbound_message_id="B", _queued_receipt_owner=None,
        )
        await GatewayRunner._run_agent_deliver_first_response(
            runner, b_ctx, adapter, {"final_response": "B response"}, {"messages": []}, None,
        )
        return {"final_response": "B response", "messages": []}

    runner._run_agent = _run_child
    await GatewayRunner._run_agent_queued_followup(
        runner, root_ctx, adapter, pending="A", pending_event=injection_a,
        response={"final_response": "root"}, result={"interrupted": True, "messages": []}, stream_task=None,
    )

    assert outcomes == ["failure"]
    assert [call.args[1] for call in adapter.send.await_args_list] == ["A response", "B response"]

@pytest.mark.asyncio
async def test_waiting_receipt_expires_before_handler_runs():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock(return_value="must not run"))
    outcomes = []
    event = _receipt_event(outcomes.append, expires_at=1.0)
    session_key = "agent:main:telegram:dm:42"
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter._process_message_background(event, session_key)

    assert outcomes == ["expired"]
    adapter._message_handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_reports_missing_handler_as_not_routed():
    adapter = _RoutingAdapter()
    runner = _runner(_entry(), adapter)
    outcomes = []

    accepted = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake",
        plugin_id="notify-plugin",
        on_complete=outcomes.append,
    )

    assert accepted is False
    assert outcomes == ["not_routed"]


def test_injection_receipt_is_once_across_threads():
    outcomes = []
    receipt = InjectionReceipt(outcomes.append)
    threads = [threading.Thread(target=receipt.terminal, args=("cancelled",)) for _ in range(16)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes == ["cancelled"]


@pytest.mark.asyncio
async def test_queue_clear_terminalizes_waiting_receipt():
    adapter = _RoutingAdapter()
    outcomes = []
    session_key = "agent:main:telegram:dm:42"
    adapter._pending_messages[session_key] = _receipt_event(outcomes.append)

    await adapter.cancel_session_processing(session_key, discard_pending=True)

    assert outcomes == ["cancelled"]
    assert session_key not in adapter._pending_messages




@pytest.mark.asyncio
async def test_admitted_callbackless_deadline_expires_while_waiting():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock(return_value="must not run"))
    session_key = "agent:main:telegram:dm:42"
    adapter._active_sessions[session_key] = asyncio.Event()

    async def _queue(event, key):
        adapter._pending_messages[key] = event
        event._gateway_accepted = True
        return True
    adapter.set_busy_session_handler(_queue)

    event = _receipt_event(None, expires_at=time.time() + 0.02)
    receipt = event._injection_receipts[0]
    event._injection_lifecycle_managed = True
    await adapter.handle_message(event)
    await asyncio.sleep(0.05)

    assert receipt.outcome == "expired"
    assert session_key not in adapter._pending_messages
    adapter._message_handler.assert_not_awaited()


def test_injection_receipt_is_excluded_from_dataclass_serialization():
    event = _receipt_event(lambda _outcome: None, expires_at=10**12)

    payload = asdict(event)

    assert "_injection_receipts" not in payload
    json.dumps(payload, default=str)

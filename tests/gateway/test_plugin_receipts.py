"""Native plugin receipts follow real adapter delivery, including silence."""
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.platforms.webhook import WebhookAdapter
from gateway.session import SessionSource
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class Adapter(BasePlatformAdapter):
    def __init__(self, succeeds=True):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.config.typing_indicator = False
        self.sent = []
        self.succeeds = succeeds

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=self.succeeds, message_id="receipt-test" if self.succeeds else None,
                          error=None if self.succeeds else "delivery rejected")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def event(callback, expires_at=None):
    metadata = {"hermes_plugin_injection": True, "hermes_plugin_id": "test-inbox"}
    if expires_at is not None:
        metadata["hermes_plugin_expires_at"] = expires_at
    return MessageEvent(
        text="technical event", source=SessionSource(platform=Platform.TELEGRAM,
            chat_id="42", user_id="42", chat_type="dm"),
        internal=True, allow_gateway_control=False, metadata=metadata,
        processing_callback=callback,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("response,succeeds,outcome", [
    ("normal reply", True, "success"), ("", True, "success"),
    ("normal reply", False, "failure"),
])
async def test_receipt_waits_for_standard_delivery_or_intentional_silence(
    tmp_path, monkeypatch, response, succeeds, outcome,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter(succeeds)
    adapter.set_message_handler(AsyncMock(return_value=response))
    outcomes = []
    await adapter.handle_message(event(outcomes.append))
    assert outcomes == []
    await asyncio.gather(*list(adapter._background_tasks))
    assert outcomes == [outcome]
    if response:
        assert response in adapter.sent
    else:
        assert adapter.sent == []


@pytest.mark.asyncio
async def test_cancelled_processing_does_not_ack_success(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter()
    started = asyncio.Event()
    async def waiting(_event):
        started.set()
        await asyncio.Event().wait()
    adapter.set_message_handler(waiting)
    outcomes = []
    await adapter.handle_message(event(outcomes.append))
    await asyncio.wait_for(started.wait(), 2)
    await adapter.cancel_session_processing(next(iter(adapter._session_tasks)))
    assert outcomes == ["cancelled"]
    assert adapter.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", [False, True])
async def test_expired_injection_never_invokes_agent(tmp_path, monkeypatch, deferred):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter()
    handler = AsyncMock(return_value="must not run")
    adapter.set_message_handler(handler)
    outcomes = []
    expired = event(outcomes.append, time.time() - 1)
    if deferred:
        adapter._start_session_processing(expired, "deferred-test")
        await asyncio.gather(*list(adapter._background_tasks))
    else:
        await adapter.handle_message(expired)
    handler.assert_not_awaited()
    assert outcomes == ["cancelled"]
    assert adapter.sent == []


def test_rejected_or_repeated_completion_is_reported_once(monkeypatch):
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="test-inbox", key="test-inbox", source="user"), manager)
    monkeypatch.setattr(ctx, "_gateway_injection_allowed", lambda: True)
    callbacks = []
    def injector(**kwargs):
        callbacks.append(kwargs["on_complete"])
        return True
    manager.set_gateway_message_injector(object(), injector)
    outcomes = []
    assert ctx.inject_message("event", session_key="existing", on_complete=outcomes.append)
    callbacks[0]("success")
    callbacks[0]("failure")
    assert outcomes == ["success"]
    manager.clear_gateway_message_injector(manager._gateway_message_injector[0])
    assert not ctx.inject_message("event", session_key="existing", on_complete=outcomes.append)
    assert outcomes == ["success", "failure"]


@pytest.mark.asyncio
async def test_native_webhook_extension_route_is_reachable(tmp_path, monkeypatch):
    from aiohttp import ClientSession, web

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="test-route", source="user"), manager)
    async def status(_request):
        return web.json_response({"native_extension": "ready"})
    ctx.register_platform_handler("webhook", lambda app, adapter: app.router.add_get("/extension-status", status))
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0}))
    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        assert await adapter.connect()
        try:
            site = next(iter(adapter._runner.sites))
            port = site._server.sockets[0].getsockname()[1]
            async with ClientSession() as client:
                async with client.get(f"http://127.0.0.1:{port}/extension-status") as response:
                    assert response.status == 200
                    assert await response.json() == {"native_extension": "ready"}
        finally:
            await adapter.disconnect()

"""Parser-only and lightweight routing tests for send_message targets.

These stay separate from ``test_send_message_tool.py`` because that module
skips wholesale when optional Telegram dependencies are not installed.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from tools.send_message_tool import _parse_target_ref, send_message_tool


def _run_async_immediately(coro):
    return asyncio.run(coro)


def test_photon_e164_target_is_explicit() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref("photon", "+15551234567")

    assert chat_id == "+15551234567"
    assert thread_id is None
    assert is_explicit is True


def test_e164_target_still_requires_phone_platform() -> None:
    assert _parse_target_ref("matrix", "+15551234567")[2] is False



def test_teams_one_to_one_conversation_id_is_explicit() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref("teams", "a:1Kw4r6LgXYtKhk7K-v58yMJ5")

    assert chat_id == "a:1Kw4r6LgXYtKhk7K-v58yMJ5"
    assert thread_id is None
    assert is_explicit is True

def test_teams_group_and_channel_conversation_ids_are_explicit() -> None:
    assert _parse_target_ref("teams", "19:072cAbC_de-fg@thread.v2")[2] is True
    assert _parse_target_ref("teams", "19:8a1b2c3d4e5f@thread.tacv2")[2] is True
    assert _parse_target_ref("teams", "19:meeting_ZmFrZQ==@thread.v2")[2] is True
    assert _parse_target_ref("teams", "19:user_bot@unq.gbl.spaces")[2] is True

def test_teams_channel_reply_keeps_its_thread_suffix_verbatim() -> None:
    # ';messageid=' addresses the channel thread; splitting it into thread_id
    # would drop the targeting, since the Teams adapter threads via reply_to.
    target = "19:8a1b2c3d4e5f@thread.tacv2;messageid=1785394370106"
    chat_id, thread_id, is_explicit = _parse_target_ref("teams", target)

    assert chat_id == target
    assert thread_id is None
    assert is_explicit is True

def test_teams_conversation_id_shape_only_matches_teams() -> None:
    assert _parse_target_ref("slack", "a:1Kw4r6LgXYtKhk7K")[2] is False
    assert _parse_target_ref("matrix", "19:072cAbC@thread.v2")[2] is False

def test_teams_friendly_name_still_uses_directory_resolution() -> None:
    assert _parse_target_ref("teams", "WEITEN Nicolas")[2] is False
    assert _parse_target_ref("teams", "IT")[2] is False

def test_send_message_routes_teams_conversation_id_without_home_fallback() -> None:
    teams_cfg = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={Platform("teams"): teams_cfg},
        get_home_channel=lambda _platform: SimpleNamespace(chat_id="a:1HOME"),
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.channel_directory.resolve_channel_name", side_effect=AssertionError("raw conversation id should not resolve via directory")), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "teams:19:072cAbC_de-fg@thread.v2",
                    "message": "hello channel",
                }
            )
        )

    assert result["success"] is True
    assert "note" not in result
    send_mock.assert_awaited_once_with(
        Platform("teams"),
        teams_cfg,
        "19:072cAbC_de-fg@thread.v2",
        "hello channel",
        thread_id=None,
        media_files=[],
        force_document=False,
    )

def test_send_message_keeps_a_resolved_name_that_matches_no_parser() -> None:
    # The directory resolves the name to an opaque platform-native id. Re-parsing
    # it yields nothing; the resolved value must survive rather than fall back to
    # the home channel.
    photon_cfg = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={Platform("photon"): photon_cfg},
        get_home_channel=lambda _platform: SimpleNamespace(chat_id="+15550000000"),
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.channel_directory.resolve_channel_name", return_value="any;-;+15551234567"), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {"action": "send", "target": "photon:family", "message": "hi"}
            )
        )

    assert result["success"] is True
    assert "note" not in result
    assert send_mock.await_args.args[2] == "any;-;+15551234567"

def test_send_message_still_rejects_an_unresolvable_name() -> None:
    teams_cfg = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={Platform("teams"): teams_cfg},
        get_home_channel=lambda _platform: SimpleNamespace(chat_id="a:1HOME"),
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.channel_directory.resolve_channel_name", return_value=None), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock()) as send_mock:
        result = json.loads(
            send_message_tool(
                {"action": "send", "target": "teams:nope", "message": "hi"}
            )
        )

    assert "Could not resolve" in result["error"]
    send_mock.assert_not_awaited()
def test_send_message_routes_whatsapp_group_jid_without_home_fallback() -> None:
    whatsapp_cfg = SimpleNamespace(enabled=True, token=None, extra={"api_url": "http://bridge"})
    config = SimpleNamespace(
        platforms={Platform.WHATSAPP: whatsapp_cfg},
        get_home_channel=lambda _platform: SimpleNamespace(chat_id="15551234567@s.whatsapp.net"),
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.channel_directory.resolve_channel_name", side_effect=AssertionError("raw JID should not resolve via directory")), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "whatsapp:120363408391911677@g.us",
                    "message": "hello group",
                }
            )
        )

    assert result["success"] is True
    assert "note" not in result
    send_mock.assert_awaited_once_with(
        Platform.WHATSAPP,
        whatsapp_cfg,
        "120363408391911677@g.us",
        "hello group",
        thread_id=None,
        media_files=[],
        force_document=False,
    )


def test_resolved_opaque_plugin_target_uses_directory_id() -> None:
    from gateway.platform_registry import PlatformEntry, platform_registry

    platform_name = "opaque-resolved-test"
    entry = PlatformEntry(
        name=platform_name,
        label="Opaque resolved test",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
    )
    platform_registry.register(entry)
    platform = Platform(platform_name)
    pconfig = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={platform: pconfig},
        get_home_channel=lambda _platform: None,
    )
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch(
                 "gateway.channel_directory.resolve_channel_name",
                 return_value="opaque:directory-id",
             ), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch(
                 "tools.send_message_tool._send_to_platform",
                 new=AsyncMock(return_value={"success": True}),
             ) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": f"{platform_name}:Friendly name",
                        "message": "hello",
                    }
                )
            )
    finally:
        platform_registry.unregister(platform_name)

    assert result["success"] is True
    send_mock.assert_awaited_once_with(
        platform,
        pconfig,
        "opaque:directory-id",
        "hello",
        thread_id=None,
        media_files=[],
        force_document=False,
    )


def test_unresolved_plugin_target_requires_explicit_parser() -> None:
    from gateway.platform_registry import PlatformEntry, platform_registry

    platform_name = "opaque-verbatim-test"
    entry = PlatformEntry(
        name=platform_name,
        label="Opaque verbatim test",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
    )
    platform_registry.register(entry)
    platform = Platform(platform_name)
    # Simulate a fresh `hermes send` process: the dynamic Platform member
    # is known from config, but plugin discovery has not registered its
    # adapter entry yet.
    platform_registry.unregister(platform_name)
    pconfig = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={platform: pconfig},
        get_home_channel=lambda _platform: None,
    )
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch(
                 "gateway.channel_directory.resolve_channel_name",
                 return_value=None,
             ), \
             patch(
                 "hermes_cli.plugins.discover_plugins",
                 side_effect=lambda: platform_registry.register(entry),
             ) as discover_mock, \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch(
                 "tools.send_message_tool._send_to_platform",
                 new=AsyncMock(return_value={"success": True}),
             ) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": f"{platform_name}:dm:panyaozhen",
                        "message": "hello",
                    }
                )
            )
    finally:
        platform_registry.unregister(platform_name)

    assert result == {
        "error": f"Could not resolve 'dm:panyaozhen' on {platform_name}. "
        "The plugin parser did not recognize it and no channel-directory entry matched."
    }
    discover_mock.assert_called_once_with()
    send_mock.assert_not_awaited()


def test_unresolved_builtin_target_keeps_directory_error() -> None:
    telegram_cfg = SimpleNamespace(enabled=True, token="***", extra={})
    config = SimpleNamespace(
        platforms={Platform.TELEGRAM: telegram_cfg},
        get_home_channel=lambda _platform: None,
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("gateway.channel_directory.resolve_channel_name", return_value=None), \
         patch(
             "tools.send_message_tool._send_to_platform",
             new=AsyncMock(return_value={"success": True}),
         ) as send_mock:
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "telegram:missing-room",
                    "message": "hello",
                }
            )
        )

    assert result == {
        "error": "Could not resolve 'missing-room' on telegram. "
        "Use send_message(action='list') to see available targets."
    }
    send_mock.assert_not_awaited()



def test_unresolved_builtin_target_passes_through_when_requested() -> None:
    """Cron and react keep the old pass-through behavior for unresolved
    built-in targets: with no model in the loop to react to an error, the
    raw id must reach the adapter, as it did before resolve_send_target
    took over these callers."""
    from tools.send_message_tool import resolve_send_target

    with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
        chat_id, thread_id, error = resolve_send_target(
            "telegram", "ops-room", pass_unresolved_references=True
        )

    assert error is None
    assert chat_id == "ops-room"
    assert thread_id is None


def test_unresolved_builtin_target_still_errors_for_the_model_tool() -> None:
    """The model-facing default stays strict: unresolved targets error with a hint."""
    from tools.send_message_tool import resolve_send_target

    with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
        chat_id, _thread_id, error = resolve_send_target("telegram", "ops-room")

    assert chat_id is None
    assert error is not None


def test_photon_group_guid_passes_through_when_requested() -> None:
    """The reported regression case: a photon group GUID matches no parser
    pattern (only DM GUIDs have an explicit rule) and no directory entry.
    Photon registers as a parser-less plugin platform, so the pass-through
    applies once platforms are prepared."""
    from tools.send_message_tool import (
        prepare_send_message_platforms,
        resolve_send_target,
    )

    prepare_send_message_platforms()
    with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
        chat_id, thread_id, error = resolve_send_target(
            "photon", "iMessage;+;chat527148912345", pass_unresolved_references=True
        )

    assert error is None
    assert chat_id == "iMessage;+;chat527148912345"
    assert thread_id is None


def test_parserless_plugin_target_passes_through_when_requested() -> None:
    """A plugin platform that declares no parser has no explicit syntax at
    all, so passing the raw id through is the only way cron can target it."""
    from gateway.platform_registry import PlatformEntry, platform_registry
    from tools.send_message_tool import resolve_send_target

    platform_name = "opaque-cron-fallback-test"
    entry = PlatformEntry(
        name=platform_name,
        label="Opaque cron fallback test",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
    )
    platform_registry.register(entry)
    try:
        with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
            chat_id, thread_id, error = resolve_send_target(
                platform_name, "dm:panyaozhen", pass_unresolved_references=True
            )
    finally:
        platform_registry.unregister(platform_name)

    assert error is None
    assert chat_id == "dm:panyaozhen"
    assert thread_id is None


def test_plugin_parser_stays_authoritative_despite_fallback() -> None:
    """A plugin that DOES declare a parser stays strict for every caller:
    its parser is the authority on native syntax, so an unrecognized
    target errors even with pass_unresolved_references."""
    from gateway.platform_registry import PlatformEntry, platform_registry
    from tools.send_message_tool import resolve_send_target

    platform_name = "opaque-parser-strict-test"
    entry = PlatformEntry(
        name=platform_name,
        label="Opaque parser strict test",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        parse_target_ref_fn=lambda ref: None,
    )
    platform_registry.register(entry)
    try:
        with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
            chat_id, _thread_id, error = resolve_send_target(
                platform_name, "dm:panyaozhen", pass_unresolved_references=True
            )
    finally:
        platform_registry.unregister(platform_name)

    assert chat_id is None
    assert error is not None

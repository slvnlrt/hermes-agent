"""Tests for per-channel toolset / memory_mode overrides (W3 — config resolved
per conversation).

These extend the upstream ``channel_overrides`` mechanism (model / provider /
system_prompt) with ``toolsets`` and ``memory_mode``, so resolution/precedence
is exercised by ``test_channel_overrides.py``; here we cover the two runner
accessors (``_resolve_channel_toolset_override`` / ``_resolve_channel_memory_mode``)
and end-to-end application at agent-build time.
"""

from types import SimpleNamespace

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.gateway.test_discord_channel_prompts import (
    _CapturingAgent,
    _install_fake_agent,
    _make_runner,
)


def _make_source(
    chat_id: str,
    *,
    parent_id: str | None = None,
    thread_id: str | None = None,
) -> SessionSource:
    """Canonical convention: ``chat_id`` is the conversation's own (most
    specific) id; a thread carries its parent channel in ``parent_chat_id``.
    """
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_type="thread" if (parent_id or thread_id) else "group",
        user_id="user-1",
        thread_id=thread_id,
        parent_chat_id=parent_id,
    )


def _runner_with_overrides(overrides: dict) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, channel_overrides=overrides),
        },
    )
    return runner


class TestResolveChannelToolsetOverride:
    def test_no_config_returns_none(self):
        runner = object.__new__(GatewayRunner)
        runner.config = None
        assert runner._resolve_channel_toolset_override(_make_source("123")) is None

    def test_no_override_returns_none(self):
        runner = _runner_with_overrides({})
        assert runner._resolve_channel_toolset_override(_make_source("123")) is None

    def test_match_by_channel_id(self):
        runner = _runner_with_overrides(
            {"100": ChannelOverride(toolsets=["hermes-channel-safe"])}
        )
        assert runner._resolve_channel_toolset_override(_make_source("100")) == [
            "hermes-channel-safe"
        ]

    def test_inherited_from_parent_channel(self):
        runner = _runner_with_overrides(
            {"200": ChannelOverride(toolsets=["hermes-channel-safe"])}
        )
        source = _make_source("999", parent_id="200")
        assert runner._resolve_channel_toolset_override(source) == ["hermes-channel-safe"]

    def test_exact_channel_beats_parent(self):
        runner = _runner_with_overrides(
            {
                "999": ChannelOverride(toolsets=["hermes-slack"]),
                "200": ChannelOverride(toolsets=["hermes-channel-safe"]),
            }
        )
        source = _make_source("999", parent_id="200")
        assert runner._resolve_channel_toolset_override(source) == ["hermes-slack"]

    def test_override_without_toolsets_returns_none(self):
        # An override that only sets model/prompt must not touch toolsets.
        runner = _runner_with_overrides({"100": ChannelOverride(model="x/y")})
        assert runner._resolve_channel_toolset_override(_make_source("100")) is None

    def test_empty_list_normalized_to_none_at_load(self):
        # from_dict drops an empty/blank toolsets list to None.
        ov = ChannelOverride.from_dict({"toolsets": []})
        assert ov.toolsets is None
        ov2 = ChannelOverride.from_dict({"toolsets": ["  ", "hermes-channel-safe", ""]})
        assert ov2.toolsets == ["hermes-channel-safe"]


class TestResolveChannelMemoryMode:
    def test_no_override_returns_none(self):
        runner = _runner_with_overrides({})
        assert runner._resolve_channel_memory_mode(_make_source("123")) is None

    def test_match_by_channel_id(self):
        runner = _runner_with_overrides({"100": ChannelOverride(memory_mode="off")})
        assert runner._resolve_channel_memory_mode(_make_source("100")) == "off"

    def test_inherited_from_parent_channel(self):
        runner = _runner_with_overrides({"200": ChannelOverride(memory_mode="off")})
        source = _make_source("999", parent_id="200")
        assert runner._resolve_channel_memory_mode(source) == "off"

    def test_exact_channel_beats_parent(self):
        runner = _runner_with_overrides(
            {
                "999": ChannelOverride(memory_mode="full"),
                "200": ChannelOverride(memory_mode="off"),
            }
        )
        source = _make_source("999", parent_id="200")
        assert runner._resolve_channel_memory_mode(source) == "full"

    def test_ambient_is_reserved_and_ignored(self):
        # "ambient" is reserved but unimplemented — treated as absent, not "off".
        runner = _runner_with_overrides({"100": ChannelOverride(memory_mode="ambient")})
        assert runner._resolve_channel_memory_mode(_make_source("100")) is None

    def test_unrecognized_value_is_treated_as_absent(self):
        runner = _runner_with_overrides({"100": ChannelOverride(memory_mode="nonsense")})
        assert runner._resolve_channel_memory_mode(_make_source("100")) is None

    def test_is_case_insensitive(self):
        runner = _runner_with_overrides({"100": ChannelOverride(memory_mode="OFF")})
        assert runner._resolve_channel_memory_mode(_make_source("100")) == "off"


def _patch_agent_build_dependencies(monkeypatch, gateway_run, tmp_path):
    """Common monkeypatches so `_run_agent` can build a (fake) agent without
    real credentials/model resolution — mirrors
    test_discord_channel_prompts.test_run_agent_appends_channel_prompt_to_ephemeral_system_prompt.
    """
    (tmp_path / "config.yaml").write_text("agent:\n  system_prompt: Global prompt\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )


def _config_with_overrides(overrides: dict) -> SimpleNamespace:
    # Wrap a real PlatformConfig (so `.channel_overrides` resolves) in the
    # lightweight namespace `_run_agent` expects (it reads `.streaming`).
    return SimpleNamespace(
        streaming=None,
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, channel_overrides=overrides),
        },
    )


@pytest.mark.asyncio
async def test_channel_toolsets_override_replaces_platform_default(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    _install_fake_agent(monkeypatch)
    runner = _make_runner()
    runner.config = _config_with_overrides(
        {"12345": ChannelOverride(toolsets=["hermes-channel-safe"])}
    )
    _patch_agent_build_dependencies(monkeypatch, gateway_run, tmp_path)

    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    _CapturingAgent.last_init = None
    await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        source=_make_source("12345"),
        session_id="session-1",
        session_key="agent:main:discord:channel:12345",
    )
    assert _CapturingAgent.last_init["enabled_toolsets"] == ["hermes-channel-safe"]

    _CapturingAgent.last_init = None
    await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        source=_make_source("99999"),
        session_id="session-2",
        session_key="agent:main:discord:channel:99999",
    )
    # No override configured for this chat_id — the platform default
    # (`_get_platform_tools` stub) passes through unchanged.
    assert _CapturingAgent.last_init["enabled_toolsets"] == ["core"]

    # Cache-safe: enabled_toolsets is already part of _agent_config_signature,
    # so the two channels land in distinct cache entries automatically — no
    # manual invalidation needed for the channel override to take effect.
    sig_override = runner._agent_cache["agent:main:discord:channel:12345"][1]
    sig_default = runner._agent_cache["agent:main:discord:channel:99999"][1]
    assert sig_override != sig_default


@pytest.mark.asyncio
async def test_channel_toolsets_exact_wins_over_parent(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    _install_fake_agent(monkeypatch)
    runner = _make_runner()
    runner.config = _config_with_overrides(
        {
            "12345": ChannelOverride(toolsets=["hermes-channel-safe"]),
            "67890": ChannelOverride(toolsets=["hermes-slack"]),
        }
    )
    _patch_agent_build_dependencies(monkeypatch, gateway_run, tmp_path)

    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    _CapturingAgent.last_init = None
    await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        # thread "67890" has its own exact entry and its parent "12345" also
        # has one — the exact (chat_id) match wins: platform < parent < exact.
        source=_make_source("67890", parent_id="12345"),
        session_id="session-1",
        session_key="agent:main:discord:thread:67890",
    )
    assert _CapturingAgent.last_init["enabled_toolsets"] == ["hermes-slack"]


@pytest.mark.asyncio
async def test_channel_memory_mode_off_sets_skip_memory(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    _install_fake_agent(monkeypatch)
    runner = _make_runner()
    runner.config = _config_with_overrides({"55555": ChannelOverride(memory_mode="off")})
    _patch_agent_build_dependencies(monkeypatch, gateway_run, tmp_path)

    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    _CapturingAgent.last_init = None
    await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        source=_make_source("55555"),
        session_id="session-1",
        session_key="agent:main:discord:channel:55555",
    )
    assert _CapturingAgent.last_init["skip_memory"] is True

    _CapturingAgent.last_init = None
    await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        source=_make_source("12345"),
        session_id="session-2",
        session_key="agent:main:discord:channel:12345",
    )
    # DM/unconfigured channel keeps today's behavior — memory stays on.
    assert _CapturingAgent.last_init["skip_memory"] is False

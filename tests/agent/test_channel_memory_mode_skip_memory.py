"""Verify that ``skip_memory`` (the mechanism ``channel_memory_modes: off``
resolves to at agent build time — gateway/run.py's
``_resolve_channel_memory_mode`` -> ``AIAgent(skip_memory=...)``) actually
cuts off every memory write path, not just the obvious one.

Three layers are covered, matching the reviewer question this is meant to
answer ("does skip_memory really stop everything, including
on_memory_write?"):

1. Agent build time (agent/agent_init.py): with skip_memory=True, neither
   the built-in MemoryStore nor the external MemoryManager get constructed
   — even when config explicitly enables both. This is the actual
   mechanism; if it ever regresses, channel_memory_modes: off silently
   stops working.
2. The built-in ``memory`` tool: with no store (the skip_memory=True
   state), it returns a graceful "not available" error instead of writing
   — so even if a restricted toolset accidentally exposed the tool, no
   write reaches disk.
3. Turn-end external sync (run_agent.py::_sync_external_memory_for_turn,
   which mirrors built-in writes to providers via on_memory_write/
   notify_memory_tool_write): gated behind the same "is there a manager at
   all" check, so it's a no-op whenever skip_memory left it unset.

Mirrors the bare-agent pattern from
tests/run_agent/test_memory_sync_interrupted.py (layer 3) and the minimal
real-AIAgent construction pattern from
tests/agent/test_non_stream_stale_timeout.py (layer 1).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.memory_provider import MemoryProvider


class _FakeAvailableProvider(MemoryProvider):
    """Minimal provider that resolves and reports available — enough to
    make agent_init keep the MemoryManager it builds (an unresolvable
    provider name makes it reset ``_memory_manager`` back to None even
    with skip_memory=False, so a real "loads and is available" provider
    is needed to prove the contrast).
    """

    @property
    def name(self) -> str:
        return "fake-available"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query, *, session_id=""):
        return ""

    def queue_prefetch(self, query, *, session_id=""):
        pass

    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        pass

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return "{}"


def _make_agent(**overrides):
    from run_agent import AIAgent

    kwargs = dict(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


_MEMORY_ENABLING_CONFIG = {
    "memory": {
        "memory_enabled": True,
        "user_profile_enabled": True,
        # A provider name is enough to make agent_init construct a
        # MemoryManager() — it doesn't need to actually resolve to an
        # installed plugin (load_memory_provider() fails closed to None
        # for an unknown name, agent_init tolerates that gracefully).
        "provider": "not-a-real-provider-for-this-test",
    }
}


class TestSkipMemoryCutsBuiltinStore:
    """Layer 1a: the built-in MEMORY.md/USER.md store."""

    def test_skip_memory_true_leaves_store_unset_even_when_config_enables_it(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: _MEMORY_ENABLING_CONFIG)
        agent = _make_agent(skip_memory=True)
        assert agent._memory_store is None
        assert agent._memory_enabled is False
        assert agent._user_profile_enabled is False

    def test_skip_memory_false_builds_store_when_config_enables_it(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: _MEMORY_ENABLING_CONFIG)
        agent = _make_agent(skip_memory=False)
        # Proves the previous test's None is caused by skip_memory, not by
        # some other missing precondition — the same config DOES take
        # effect once skip_memory stops gating it.
        assert agent._memory_store is not None
        assert agent._memory_enabled is True
        assert agent._user_profile_enabled is True


class TestSkipMemoryCutsExternalManager:
    """Layer 1b: the external MemoryManager (on_memory_write / sync_all /
    queue_prefetch_all / has_tool bridge all live behind ``agent._memory_manager``
    — every call site in the codebase guards with ``if agent._memory_manager:``
    first, so a None manager is a hard cut, not just a soft default).
    """

    def test_skip_memory_true_leaves_manager_unset_even_with_provider_configured(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: _MEMORY_ENABLING_CONFIG)
        agent = _make_agent(skip_memory=True)
        assert agent._memory_manager is None

    def test_skip_memory_false_constructs_manager_when_provider_configured(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: _MEMORY_ENABLING_CONFIG)
        monkeypatch.setattr(
            "plugins.memory.load_memory_provider",
            lambda name: _FakeAvailableProvider(),
        )
        agent = _make_agent(skip_memory=False)
        assert agent._memory_manager is not None
        assert agent._memory_manager.providers


class TestMemoryToolNoOpsWithoutStore:
    """Layer 2: even if a misconfigured restricted toolset exposed the
    ``memory`` tool schema, the handler itself refuses to write without a
    store — the skip_memory=True state (agent._memory_store is None).
    """

    def test_memory_tool_refuses_to_write_without_a_store(self):
        from tools.memory_tool import memory_tool

        result = memory_tool(action="add", target="memory", content="secret", store=None)
        assert '"success": false' in result.lower().replace(" ", "").replace("\n", "") or "false" in result
        assert "not available" in result.lower()


class TestSyncExternalMemoryForTurnRequiresManager:
    """Layer 3: ``_sync_external_memory_for_turn`` (the turn-end call that
    mirrors a completed exchange into external providers via
    ``memory_manager.sync_all`` / ``queue_prefetch_all``) is a hard no-op
    whenever ``_memory_manager`` is unset — the exact state skip_memory=True
    leaves the agent in. Contrasted with the "full" (memory on) case where
    a real manager IS synced, so this isn't just "no crash" but "the
    manager's write path is genuinely reached only when memory is on."
    """

    def _bare_agent(self):
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent.session_id = "test-session"
        return agent

    def test_off_state_never_calls_sync_all(self):
        agent = self._bare_agent()
        agent._memory_manager = None  # the skip_memory=True / channel "off" state
        # Must not raise (would AttributeError on `.sync_all` if the guard
        # were ever removed) and must not reach any provider.
        agent._sync_external_memory_for_turn(
            original_user_message="What's on my calendar?",
            final_response="Nothing scheduled today.",
            interrupted=False,
        )

    def test_full_state_does_call_sync_all(self):
        agent = self._bare_agent()
        agent._memory_manager = MagicMock()  # the channel "full" / default state
        agent._sync_external_memory_for_turn(
            original_user_message="What's on my calendar?",
            final_response="Nothing scheduled today.",
            interrupted=False,
        )
        agent._memory_manager.sync_all.assert_called_once()
        agent._memory_manager.queue_prefetch_all.assert_called_once()

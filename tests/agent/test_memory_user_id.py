"""Identity-scoped memory across local surfaces and messaging channels."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_provider import MemoryProvider


class ScopedMemory(MemoryProvider):
    """In-memory backend enforcing the external provider's identity contract."""

    def __init__(self, records):
        self.records = records
        self.owner = None
        self.writable = False

    @property
    def name(self):
        return "scoped"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.owner = kwargs.get("user_id_alt") or kwargs.get("user_id")
        self.writable = bool(self.owner) and kwargs.get("agent_context") == "primary"

    def system_prompt_block(self):
        return ""

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"ok": False})

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        if self.writable:
            self.records.setdefault(self.owner, []).append(user_content)

    def prefetch(self, query, *, session_id=""):
        return "\n".join(
            text for text in self.records.get(self.owner, []) if query in text
        )


@pytest.fixture
def memory_agents(monkeypatch):
    from run_agent import AIAgent

    records = {}
    managers = []
    config = {
        "memory": {
            "memory_enabled": False,
            "user_profile_enabled": False,
            "provider": "scoped",
            "local_user_id": "owner",
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    monkeypatch.setattr(
        "plugins.memory.load_memory_provider", lambda name: ScopedMemory(records)
    )

    def build(platform, *, source=None, **kwargs):
        monkeypatch.setenv("HERMES_SESSION_SOURCE", source or platform)
        agent = AIAgent(
            model="gpt-5.5",
            provider="openai-codex",
            api_key="sk-dummy",
            base_url="https://chatgpt.com/backend-api/codex",
            quiet_mode=True,
            skip_context_files=True,
            platform=platform,
            enabled_toolsets=["memory"],
            **kwargs,
        )
        manager = agent._memory_manager
        assert manager is not None
        managers.append(manager)
        return manager

    yield build, config
    for manager in managers:
        manager.shutdown_all()


def _remember(manager, text):
    manager.sync_all(text, "Noted.")
    assert manager.flush_pending(timeout=5)


def test_local_memory_survives_new_sessions_and_matches_gateway_owner(memory_agents):
    build, _ = memory_agents
    _remember(build("tui"), "The maintenance marker is cobalt-orbit.")
    for surface in ("cli", "tui", "desktop", "web"):
        assert "cobalt-orbit" in build(surface).prefetch_all("cobalt-orbit")
    telegram = build("telegram", user_id="owner", chat_type="private")
    assert "cobalt-orbit" in telegram.prefetch_all("cobalt-orbit")
    _remember(telegram, "The return marker is copper-moon.")
    assert "copper-moon" in build("tui").prefetch_all("copper-moon")


def test_configured_local_identity_never_overrides_gateway_identity(memory_agents):
    build, _ = memory_agents
    _remember(build("cli"), "private-owner-marker")
    other = build("telegram", user_id="other", chat_type="private")
    assert not other.prefetch_all("private-owner-marker")
    _remember(other, "private-other-marker")
    assert not build("cli").prefetch_all("private-other-marker")
    alternate = build("telegram", user_id="transient", user_id_alt="owner")
    assert "private-owner-marker" in alternate.prefetch_all("private-owner-marker")


@pytest.mark.parametrize("source", ["tool", "cron", "kanban", "subagent"])
def test_automated_local_sessions_do_not_read_or_write_owner_memory(memory_agents, source):
    build, _ = memory_agents
    _remember(build("cli"), "personal-only-marker")
    automation = build("cli", source=source)
    assert not automation.prefetch_all("personal-only-marker")
    _remember(automation, "automation-only-marker")
    assert not build("cli").prefetch_all("automation-only-marker")


def test_unidentified_remote_session_cannot_borrow_local_owner(memory_agents):
    build, _ = memory_agents
    _remember(build("tui"), "owner-only-marker")
    for surface in ("telegram", "webhook", "api_server"):
        unidentified = build(surface)
        assert not unidentified.prefetch_all("owner-only-marker")
        _remember(unidentified, "unidentified-marker")
    assert not build("cli").prefetch_all("unidentified-marker")


@pytest.mark.parametrize("configured", ["", "   ", None, 123])
def test_local_identity_requires_an_explicit_nonempty_string(memory_agents, configured):
    build, config = memory_agents
    _remember(build("telegram", user_id="owner"), "existing-owner-marker")
    config["memory"]["local_user_id"] = configured
    local = build("cli")
    assert not local.prefetch_all("existing-owner-marker")
    _remember(local, "unattributed-marker")
    assert not build("telegram", user_id="owner").prefetch_all("unattributed-marker")

# ---------------------------------------------------------------------------
# Mem0 provider user_id tests
# ---------------------------------------------------------------------------


class TestMem0UserIdScoping:
    """Verify Mem0 plugin uses gateway user_id when provided."""


    def test_no_user_id_falls_back_to_config(self):
        """Without user_id in kwargs, should use config default."""
        from plugins.memory.mem0 import Mem0MemoryProvider

        provider = Mem0MemoryProvider()
        with patch("plugins.memory.mem0._load_config", return_value={
            "api_key": "test-key",
            "user_id": "custom-default",
            "agent_id": "hermes",
            "rerank": True,
        }):
            provider.initialize(session_id="test-sess")

        assert provider._user_id == "custom-default"


    def test_different_users_get_different_ids(self):
        """Two providers initialized with different user_ids should be scoped differently."""
        from plugins.memory.mem0 import Mem0MemoryProvider

        p1 = Mem0MemoryProvider()
        p2 = Mem0MemoryProvider()

        with patch("plugins.memory.mem0._load_config", return_value={
            "api_key": "test-key",
            "user_id": "hermes-user",
            "agent_id": "hermes",
            "rerank": True,
        }):
            p1.initialize(session_id="sess-1", user_id="alice_123")
            p2.initialize(session_id="sess-2", user_id="bob_456")

        assert p1._user_id == "alice_123"
        assert p2._user_id == "bob_456"
        assert p1._user_id != p2._user_id


# ---------------------------------------------------------------------------
# Honcho provider user_id tests
# ---------------------------------------------------------------------------


class TestHonchoUserIdScoping:
    """Verify Honcho plugin keeps runtime user scoping separate from config peer_name."""

    def test_gateway_user_id_is_passed_as_runtime_peer(self):
        """Gateway user_id should scope Honcho sessions without mutating config peer_name."""
        from plugins.memory.honcho import HonchoMemoryProvider

        provider = HonchoMemoryProvider()

        mock_cfg = MagicMock()
        mock_cfg.enabled = True
        mock_cfg.api_key = "test-key"
        mock_cfg.base_url = None
        mock_cfg.peer_name = "static-user"
        mock_cfg.recall_mode = "context"
        mock_cfg.context_tokens = None
        mock_cfg.raw = {}
        mock_cfg.dialectic_depth = 1
        mock_cfg.dialectic_depth_levels = None
        mock_cfg.init_on_session_start = False
        mock_cfg.ai_peer = "hermes"
        mock_cfg.resolve_session_name.return_value = "test-sess"
        mock_cfg.session_strategy = "shared"

        with patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=mock_cfg,
        ), patch(
            "plugins.memory.honcho.client.get_honcho_client",
            return_value=MagicMock(),
        ), patch(
            "plugins.memory.honcho.session.HonchoSessionManager",
        ) as mock_manager_cls:
            mock_manager = MagicMock()
            mock_manager.get_or_create.return_value = MagicMock(messages=[])
            mock_manager_cls.return_value = mock_manager
            provider.initialize(
                session_id="test-sess",
                user_id="discord_user_789",
                platform="discord",
            )

        assert mock_cfg.peer_name == "static-user"
        assert mock_manager_cls.call_args.kwargs["runtime_user_peer_name"] == "discord_user_789"

    def test_session_manager_prefers_runtime_user_id_over_config_peer_name(self):
        """Session manager should isolate gateway users even when config peer_name is static."""
        from plugins.memory.honcho.session import HonchoSessionManager

        mock_cfg = MagicMock()
        mock_cfg.peer_name = "static-user"
        mock_cfg.ai_peer = "hermes"
        mock_cfg.write_frequency = "sync"
        mock_cfg.dialectic_reasoning_level = "low"
        mock_cfg.dialectic_dynamic = True
        mock_cfg.dialectic_max_chars = 600
        mock_cfg.observation_mode = "directional"
        mock_cfg.user_observe_me = True
        mock_cfg.user_observe_others = True
        mock_cfg.ai_observe_me = True
        mock_cfg.ai_observe_others = True

        manager = HonchoSessionManager(
            honcho=MagicMock(),
            config=mock_cfg,
            runtime_user_peer_name="discord_user_789",
        )

        with patch.object(manager, "_get_or_create_peer", return_value=MagicMock()), patch.object(
            manager,
            "_get_or_create_honcho_session",
            return_value=(MagicMock(), []),
        ):
            session = manager.get_or_create("discord:channel-1")

        assert session.user_peer_id == "discord_user_789"

    def test_no_user_id_preserves_config_peer_name(self):
        """Without user_id, the config peer_name should be preserved."""
        from plugins.memory.honcho import HonchoMemoryProvider

        provider = HonchoMemoryProvider()

        mock_cfg = MagicMock()
        mock_cfg.enabled = True
        mock_cfg.api_key = "test-key"
        mock_cfg.base_url = None
        mock_cfg.peer_name = "my-custom-peer"
        mock_cfg.recall_mode = "tools"

        with patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=mock_cfg,
        ):
            provider.initialize(
                session_id="test-sess",
                platform="cli",
            )

        # peer_name should not have been overridden
        assert mock_cfg.peer_name == "my-custom-peer"



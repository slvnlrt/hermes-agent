"""Tests for per-user memory scoping via the configured provider lifecycle.

The recording provider is installed through the real memory-plugin discovery
path under a temporary HERMES_HOME. Tests observe initialize() kwargs after
real AIAgent construction instead of copying private state or mocking loader
results.
"""

import json
from contextlib import suppress

import pytest

from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Concrete test provider that records init kwargs
# ---------------------------------------------------------------------------


class RecordingProvider(MemoryProvider):
    """Minimal provider that records what initialize() receives."""

    def __init__(self, name="recording"):
        self._name = name
        self._init_kwargs = {}
        self._init_session_id = None

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._init_session_id = session_id
        self._init_kwargs = dict(kwargs)

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        pass

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({})

    def shutdown(self):
        pass


# ---------------------------------------------------------------------------
# MemoryManager user_id threading tests
# ---------------------------------------------------------------------------


class TestMemoryManagerUserIdThreading:
    """Verify user_id reaches providers via initialize_all."""

    def test_no_user_id_when_cli(self):
        """CLI sessions should not have user_id in kwargs."""
        mgr = MemoryManager()
        p = RecordingProvider()
        mgr.add_provider(p)

        mgr.initialize_all(
            session_id="sess-456",
            platform="cli",
        )

        assert "user_id" not in p._init_kwargs
        assert p._init_kwargs.get("platform") == "cli"

    def test_multiple_providers_all_receive_user_id(self):
        mgr = MemoryManager()
        p1 = RecordingProvider("builtin")
        p2 = RecordingProvider("external")
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        mgr.initialize_all(
            session_id="sess-multi",
            platform="slack",
            user_id="slack_U12345",
        )

        assert p1._init_kwargs.get("user_id") == "slack_U12345"
        assert p1._init_kwargs.get("platform") == "slack"
        assert p2._init_kwargs.get("user_id") == "slack_U12345"
        assert p2._init_kwargs.get("platform") == "slack"




# ---------------------------------------------------------------------------
# Real AIAgent construction provenance tests
# ---------------------------------------------------------------------------


class TestRealAIAgentProvenance:
    """Real configured-provider initialization, with discoverable plugin loading.

    Factory-specific admission is covered in the TUI/desktop transport tests;
    these rows prove the provider receives only the already-established fact.
    """

    @pytest.fixture(autouse=True)
    def _tmp_hermes_home(self, tmp_path, monkeypatch):
        """Install a discoverable provider and let the real loader initialize it."""
        self.hermes_home = tmp_path / "hermes_home"
        provider_dir = self.hermes_home / "plugins" / "recording"
        provider_dir.mkdir(parents=True)
        (self.hermes_home / "config.yaml").write_text(json.dumps({
            "memory": {"provider": "recording", "local_user_id": "test-owner-42"},
        }))
        (provider_dir / "__init__.py").write_text(
            """
import json
from pathlib import Path
from agent.memory_provider import MemoryProvider

class RecordingProvider(MemoryProvider):
    @property
    def name(self):
        return "recording"
    def is_available(self):
        return True
    def initialize(self, session_id, **kwargs):
        self._init_kwargs = dict(kwargs)
        observed = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, type(None)))}
        Path(kwargs["hermes_home"], "recording-init.json").write_text(json.dumps(observed))
    def system_prompt_block(self):
        return ""
    def prefetch(self, query, *, session_id=""):
        return ""
    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        return None
    def get_tool_schemas(self):
        return []
    def handle_tool_call(self, tool_name, args, **kwargs):
        return "{}"
    def shutdown(self):
        return None

def register(ctx):
    ctx.register_memory_provider(RecordingProvider())
""".strip())
        monkeypatch.setenv("HERMES_HOME", str(self.hermes_home))
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    @staticmethod
    def _provider(agent):
        assert agent._memory_manager is not None
        assert len(agent._memory_manager.providers) == 1
        provider = agent._memory_manager.providers[0]
        assert provider.name == "recording"
        return provider
    def test_configured_provider_observes_established_provenance(self):
        """A real provider receives the local owner only for an established fact."""
        from run_agent import AIAgent
        agent = AIAgent(
            model="test/model", api_key="test-key-not-real",
            platform="cli", _local_owner_provenance=True,
            quiet_mode=True, max_iterations=1,
        )
        try:
            assert agent._local_owner_provenance is True
            assert self._provider(agent)._init_kwargs["user_id"] == "test-owner-42"
        finally:
            with suppress(Exception):
                agent.close()

    def test_bare_cli_no_provenance(self):
        """A bare AIAgent(platform='cli') without provenance MUST NOT get owner."""
        from run_agent import AIAgent
        agent = AIAgent(
            model="test/model", api_key="test-key-not-real",
            platform="cli",  # no _local_owner_provenance
            quiet_mode=True, max_iterations=1,
        )
        try:
            assert agent._local_owner_provenance is False
            assert "user_id" not in self._provider(agent)._init_kwargs
        finally:
            with suppress(Exception):
                agent.close()

    def test_nonprimary_execution_blocks_owner_even_with_established_flag(self):
        from run_agent import AIAgent
        agent = AIAgent(
            model="test/model", api_key="test-key-not-real", platform="cli",
            _local_owner_provenance=True, _execution_context="background",
            quiet_mode=True, max_iterations=1,
        )
        try:
            provider = self._provider(agent)
            assert provider._init_kwargs["agent_context"] == "background"
            assert "user_id" not in provider._init_kwargs
        finally:
            with suppress(Exception):
                agent.close()

    def test_gateway_user_id_wins_in_real_construction(self):
        """Gateway user_id takes precedence even with provenance=True."""
        from run_agent import AIAgent
        agent = AIAgent(
            model="test/model", api_key="test-key-not-real",
            platform="cli", _local_owner_provenance=True,
            user_id="gateway-user-99",
            quiet_mode=True, max_iterations=1,
        )
        try:
            assert self._provider(agent)._init_kwargs["user_id"] == "gateway-user-99"
        finally:
            with suppress(Exception):
                agent.close()

    def test_hygiene_platform_non_primary(self):
        """An agent constructed with platform='gateway_hygiene' gets non-primary context."""
        from run_agent import AIAgent
        agent = AIAgent(
            model="test/model", api_key="test-key-not-real",
            platform="gateway_hygiene",
            quiet_mode=True, max_iterations=1,
        )
        try:
            assert agent._local_owner_provenance is False
            assert self._provider(agent)._init_kwargs["agent_context"] == "gateway_hygiene"
            assert "user_id" not in self._provider(agent)._init_kwargs
        finally:
            with suppress(Exception):
                agent.close()

    def test_skip_memory_true_no_provider_init(self):
        """skip_memory=True skips the external provider entirely."""
        from run_agent import AIAgent
        agent = AIAgent(
            model="test/model", api_key="test-key-not-real",
            platform="cli", _local_owner_provenance=True,
            skip_memory=True,
            quiet_mode=True, max_iterations=1,
        )
        try:
            assert agent._memory_manager is None
        finally:
            with suppress(Exception):
                agent.close()

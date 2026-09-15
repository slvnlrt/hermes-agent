"""Behavioral regressions for per-task delegation routes."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from run_agent import AIAgent
import tools.delegate_tool  # register delegate_task with the real registry
from tools.delegate_tool import _build_child_agent
from tools.delegate_tool_config import _credential_bundle, _resolve_task_credentials
from tools.registry import registry


def _parent():
    return AIAgent(
        api_key="parent-secret",
        base_url="http://127.0.0.1:9/v1",
        provider="custom",
        model="parent-model",
        max_tokens=777,
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=None,
    )


def test_registry_rejects_explicit_null_before_any_child_is_constructed():
    """A present JSON null is invalid, even when an earlier task is valid."""
    parent = MagicMock()
    parent._delegate_depth = 0
    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value=_credential_bundle(None, None, None, None, None, None),
        ),
        patch("tools.delegate_tool._build_child_preserving_parent_tools") as build,
    ):
        result = registry.dispatch(
            "delegate_task",
            {
                "tasks": [
                    {"goal": "Keep this valid task from being constructed"},
                    {"goal": "Reject this explicit null route", "provider": None},
                ]
            },
            parent_agent=parent,
        )
    payload = json.loads(result)
    assert "error" in payload
    assert "provider" in payload["error"]
    assert "NoneType" in payload["error"]
    build.assert_not_called()

def test_registry_rejects_global_provider_missing_endpoint_before_construction():
    """Global provider pins fail in preflight instead of reaching ambient routing."""
    parent = MagicMock()
    parent._delegate_depth = 0
    with (
        patch("tools.delegate_tool._load_config", return_value={"provider": "openrouter"}),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "provider": "openrouter", "model": "target-model", "base_url": "",
                "api_key": "target-key", "api_mode": "chat_completions",
            },
        ),
        patch("tools.delegate_tool._build_child_preserving_parent_tools") as build,
    ):
        result = registry.dispatch(
            "delegate_task", {"goal": "must not enter routed construction"}, parent_agent=parent,
        )
    payload = json.loads(result)
    assert "API key and base URL" in payload["error"]
    build.assert_not_called()



def test_live_manifest_records_each_effective_task_route_without_route_secrets(tmp_path, monkeypatch):
    from tools.delegation_live_log import _write_manifest

    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr("tools.delegation_live_log._manifest_path", lambda _id: manifest_path)
    _write_manifest(
        "delegation-id",
        [{"goal": "First"}, {"goal": "Second"}],
        [str(tmp_path / "first.log"), str(tmp_path / "second.log")],
        task_routes=[
            ({"provider": "openrouter", "model": "first-model", "api_key": "never-write"}, {}),
            ({"provider": "nous", "model": "second-model", "base_url": "never-write"}, {}),
        ],
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["provider"] is None
    assert manifest["model"] is None
    assert [
        (task["provider"], task["model"]) for task in manifest["tasks"]
    ] == [("openrouter", "first-model"), ("nous", "second-model")]
    assert "api_key" not in manifest["tasks"][0]
    assert "base_url" not in manifest["tasks"][1]

def test_model_pin_preserves_identity_without_reselecting_credentials_and_updates_nous_wire():
    """A model-only task keeps the batch account while choosing the model's wire mode."""
    batch = {
        "provider": "nous",
        "model": "hermes-4-70b",
        "base_url": "https://inference.example/v1",
        "api_key": "batch-account-token",
        "api_mode": "chat_completions",
        "request_overrides": {"extra_body": {"keep": True}},
        "command": "agent-command",
        "args": ["--safe"],
    }
    with (
        patch("tools.delegate_tool_config._resolve_delegation_credentials", side_effect=AssertionError),
        patch("hermes_cli.runtime_provider_custom._get_named_custom_provider", return_value=None),
        patch("hermes_cli.providers._nous_anthropic_wire", return_value="native"),
    ):
        routed, _ = _resolve_task_credentials(
            {"goal": "Use the native wire", "model": "anthropic/claude-opus-5"},
            batch,
            {"provider": "nous"},
            parent_agent=None,
        )

    assert routed["model"] == "anthropic/claude-opus-5"
    assert routed["api_mode"] == "anthropic_messages"
    assert {key: routed[key] for key in ("provider", "base_url", "api_key", "command", "args")} == {
        key: batch[key] for key in ("provider", "base_url", "api_key", "command", "args")
    }


def test_model_pin_applies_opencode_model_wire_without_changing_the_resolved_endpoint():
    batch = {
        "provider": "opencode-go",
        "model": "deepseek-v4-flash",
        "base_url": "https://opencode.ai/zen/go/v1",
        "api_key": "batch-account-token",
        "api_mode": "chat_completions",
        "request_overrides": None,
    }
    with patch("hermes_cli.runtime_provider_custom._get_named_custom_provider", return_value=None):
        routed, _ = _resolve_task_credentials(
            {"goal": "Use the model's native endpoint", "model": "minimax-m3"},
            batch,
            {"provider": "opencode-go"},
            parent_agent=None,
        )

    assert routed["api_mode"] == "anthropic_messages"
    assert routed["base_url"] == batch["base_url"]


def test_direct_pinned_http_child_refuses_ambient_routing_before_constructor(monkeypatch):
    """A direct caller cannot make an incomplete HTTP pin inherit process credentials."""
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-openrouter")
    parent = _parent()
    try:
        with patch("run_agent.AIAgent") as construct:
            try:
                _build_child_agent(
                    task_index=0, goal="must not route", context=None, toolsets=[],
                    model="target-model", max_iterations=1, task_count=1, parent_agent=parent,
                    override_provider="unconfigured-http",
                )
            except ValueError as exc:
                assert "API key and base URL" in str(exc)
            else:
                raise AssertionError("incomplete pinned HTTP route constructed a child")
            construct.assert_not_called()
    finally:
        parent.close()


def test_direct_keyless_pin_uses_only_its_declared_transport(monkeypatch):
    """A real child uses OpenCode Free's keyless route despite ambient HTTP credentials."""
    from hermes_cli.models import OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER

    class RoutedClient:
        def __init__(self, *, api_key, base_url, **kwargs):
            self.api_key = api_key
            self.base_url = base_url
            self._custom_headers = kwargs.get("default_headers")

    primary_kwargs = {}

    def primary_client(**kwargs):
        primary_kwargs.update(kwargs)
        return MagicMock()

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-openrouter")
    parent = _parent()
    child = None
    try:
        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch("agent.auxiliary_client.OpenAI", RoutedClient),
            patch("agent.process_bootstrap.OpenAI", side_effect=primary_client),
        ):
            child = _build_child_agent(
                task_index=0, goal="use only the free route", context=None, toolsets=[],
                model="mimo-v2.5-free", max_iterations=1, task_count=1, parent_agent=parent,
                override_provider="opencode-free", task_pinned=True,
            )
        assert child.provider == "opencode-free"
        assert child._client_kwargs["api_key"] == OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER
        assert child._client_kwargs["base_url"] == "https://opencode.ai/zen/v1"
        assert primary_kwargs["api_key"] == OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER
        assert primary_kwargs["base_url"] == "https://opencode.ai/zen/v1"
        assert primary_kwargs["default_headers"]["Authorization"] == ""
    finally:
        if child is not None:
            child.close()
        parent.close()


def _capture_dispatched_child(monkeypatch, task, delegation_cfg=None):
    """Dispatch through the registry and retain the actual constructed child."""
    captured = {}

    def capture(batch, _background):
        captured["child"] = batch.children[0][2]
        return json.dumps({"accepted": True})

    parent = _parent()
    with (
        patch("tools.delegate_tool._load_config", return_value=delegation_cfg or {}),
        patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [None], [])),
        patch("tools.delegate_tool._run_batch", side_effect=capture),
    ):
        payload = json.loads(registry.dispatch("delegate_task", {"tasks": [task]}, parent_agent=parent))
    assert payload == {"accepted": True}
    return parent, captured["child"]


def test_unpinned_effective_route_forwards_parent_cap_to_the_child_transport(monkeypatch):
    """A task with no route override retains the parent's concrete transport and cap."""
    parent, child = _capture_dispatched_child(monkeypatch, {"goal": "Use the parent route"})
    try:
        wire = child._build_api_kwargs([{"role": "user", "content": "hello"}], [])
        assert child.provider == parent.provider
        assert child.base_url == parent.base_url
        assert child.max_tokens == parent.max_tokens
        assert wire["max_tokens"] == parent.max_tokens
    finally:
        child.close()
        parent.close()


def test_pinned_route_leaves_output_cap_to_the_target_transport_profile(monkeypatch):
    parent, child = _capture_dispatched_child(
        monkeypatch,
        {"goal": "Pin this task, not its sibling", "model": "meta-llama/llama-4-scout"},
    )
    try:
        wire = child._build_api_kwargs([{"role": "user", "content": "hello"}], [])
        assert child.max_tokens is None
        assert "max_tokens" not in wire
    finally:
        child.close()
        parent.close()


def test_global_provider_route_uses_its_own_output_cap(monkeypatch):
    """A global delegation route is an override even when the task itself is unpinned."""
    parent, child = _capture_dispatched_child(
        monkeypatch,
        {"goal": "Use the configured child route"},
        {"provider": "opencode-free", "model": "mimo-v2.5-free"},
    )
    try:
        wire = child._build_api_kwargs([{"role": "user", "content": "hello"}], [])
        assert child.provider == "opencode-free"
        assert child.max_tokens is None
        assert "max_tokens" not in wire
    finally:
        child.close()
        parent.close()

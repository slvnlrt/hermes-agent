"""Behavioral regressions for per-task delegation routes."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

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
    assert "error" in payload
    build.assert_not_called()



def test_live_manifest_records_each_effective_task_route_without_route_secrets(tmp_path, monkeypatch):
    from tools.delegation_live_log import _write_manifest

    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr("tools.delegation_live_log._manifest_path", lambda _id, home=None: manifest_path)
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


def test_model_pin_applies_opencode_wire_and_normalizes_its_endpoint_path():
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
    assert routed["base_url"] == "https://opencode.ai/zen/go"
    assert routed["provider"] == batch["provider"]
    assert routed["api_key"] == batch["api_key"]


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
    """A real child keeps LM Studio's no-auth route despite ambient HTTP credentials."""
    from hermes_cli.auth import LMSTUDIO_NOAUTH_PLACEHOLDER

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
    monkeypatch.delenv("LM_API_KEY", raising=False)
    monkeypatch.setenv("LM_BASE_URL", "http://127.0.0.1:12345/v1")
    monkeypatch.setattr("hermes_cli.models_local.ensure_lmstudio_model_loaded", lambda *args, **kwargs: None)
    parent = _parent()
    child = None
    try:
        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch("agent.auxiliary_client.OpenAI", RoutedClient),
            patch("agent.process_bootstrap.OpenAI", side_effect=primary_client),
        ):
            routed, routing_cfg = _resolve_task_credentials(
                {"goal": "Use only the local route", "provider": "lmstudio", "model": "meta-llama/llama-4-scout"},
                _credential_bundle(None, None, None, None, None, None),
                {},
                parent_agent=parent,
            )
            child = _build_child_agent(
                task_index=0, goal="use only the local route", context=None, toolsets=[],
                model=routed["model"], max_iterations=1, task_count=1, parent_agent=parent,
                override_provider=routed["provider"], override_base_url=routed["base_url"],
                override_api_key=routed["api_key"], override_api_mode=routed["api_mode"],
                routing_cfg=routing_cfg, task_pinned=True,
            )
        assert child.provider == "lmstudio"
        assert child.requested_provider == "lmstudio"
        assert child.model == "meta-llama/llama-4-scout"
        assert child._client_kwargs["api_key"] == LMSTUDIO_NOAUTH_PLACEHOLDER
        assert child._client_kwargs["base_url"] == "http://127.0.0.1:12345/v1"
        assert primary_kwargs["api_key"] == LMSTUDIO_NOAUTH_PLACEHOLDER
        assert primary_kwargs["base_url"] == "http://127.0.0.1:12345/v1"
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
    monkeypatch.delenv("LM_API_KEY", raising=False)
    monkeypatch.setenv("LM_BASE_URL", "http://127.0.0.1:12345/v1")
    monkeypatch.setattr("hermes_cli.models_local.ensure_lmstudio_model_loaded", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.models_local.lmstudio_model_reasoning_options", lambda *args, **kwargs: [])
    parent, child = _capture_dispatched_child(
        monkeypatch,
        {"goal": "Use the configured child route"},
        {"provider": "lmstudio", "model": "meta-llama/llama-4-scout"},
    )
    try:
        wire = child._build_api_kwargs([{"role": "user", "content": "hello"}], [])
        assert child.provider == "lmstudio"
        assert child.base_url == "http://127.0.0.1:12345/v1"
        assert child.max_tokens is None
        assert "max_tokens" not in wire
    finally:
        child.close()
        parent.close()


def test_registry_prevalidates_late_provider_failure_before_constructing_siblings():
    parent = MagicMock()
    parent._delegate_depth = 0
    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=ValueError("unknown route")),
        patch("tools.delegate_tool._build_child_preserving_parent_tools") as build,
    ):
        payload = json.loads(registry.dispatch(
            "delegate_task",
            {"tasks": [{"goal": "valid sibling"}, {"goal": "invalid sibling", "provider": "missing-provider"}]},
            parent_agent=parent,
        ))
    assert "Task 1:" in payload["error"]
    assert "unknown route" in payload["error"]
    build.assert_not_called()


def test_direct_endpoint_requires_its_own_key_before_any_child_construction():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.api_key = "parent-secret"
    with (
        patch("tools.delegate_tool._load_config", return_value={"base_url": "https://other.example/v1"}),
        patch("tools.delegate_tool._build_child_preserving_parent_tools") as build,
    ):
        payload = json.loads(registry.dispatch(
            "delegate_task", {"goal": "do not borrow parent credentials"}, parent_agent=parent,
        ))
    assert "API key and base URL" in payload["error"]
    build.assert_not_called()


@pytest.mark.parametrize("provider", [None, "custom"])
def test_changed_endpoint_never_inherits_parent_key_wire_or_acp(provider):
    from types import SimpleNamespace
    from tools.delegate_tool_config import _resolve_child_runtime

    parent = SimpleNamespace(
        model="parent-model", provider="custom", base_url="https://parent.example/v1",
        api_mode="anthropic_messages", acp_command="parent-acp", acp_args=["--parent"],
    )
    overrides = {
        "model": "child-model", "override_provider": provider,
        "override_base_url": "https://child.example/v1", "override_api_key": None,
        "override_api_mode": None, "override_acp_command": None, "override_acp_args": None,
    }
    with pytest.raises(ValueError, match="API key and base URL"):
        _resolve_child_runtime(parent, {}, "parent-secret", **overrides)
    overrides["override_api_key"] = "child-secret"
    runtime = _resolve_child_runtime(parent, {}, "parent-secret", **overrides)
    assert runtime["api_key"] == "child-secret"
    assert runtime["base_url"] == "https://child.example/v1"
    assert runtime["api_mode"] is None
    assert runtime["acp_command"] is None
    assert runtime["acp_args"] == []


def test_model_only_pin_updates_inherited_opencode_wire_without_resolving_credentials():
    from types import SimpleNamespace

    inherited = _credential_bundle(None, None, None, None, None, None)
    parent = SimpleNamespace(provider="opencode-go", requested_provider="opencode-go")
    with (
        patch("tools.delegate_tool_config._resolve_delegation_credentials", side_effect=AssertionError),
        patch("hermes_cli.runtime_provider_custom._get_named_custom_provider", return_value=None),
    ):
        routed, _ = _resolve_task_credentials({"model": "minimax-m3"}, inherited, {}, parent)
    assert routed["api_mode"] == "anthropic_messages"
    assert routed["provider"] is None
    assert routed["base_url"] is None
    assert routed["api_key"] is None


def test_mixed_route_manifest_respects_explicit_profile_home(tmp_path, monkeypatch):
    from pathlib import Path
    from tools.delegation_live_log import create_live_transcripts

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))
    home = tmp_path / "profile"
    delegation_id, writers, paths = create_live_transcripts(
        [{"goal": "first"}, {"goal": "second"}], home=home,
        task_routes=[
            ({"provider": "openrouter", "model": "first-model"}, {}),
            ({"provider": "nous", "model": "second-model"}, {}),
        ],
    )
    assert delegation_id
    assert len(writers) == len(paths) == 2
    root = home / "cache" / "delegation" / "live" / delegation_id
    assert all(Path(path).parent == root for path in paths)
    manifest = json.loads((root / "manifest.json").read_text())
    assert [(task["provider"], task["model"]) for task in manifest["tasks"]] == [
        ("openrouter", "first-model"), ("nous", "second-model"),
    ]
    assert not (tmp_path / "ambient" / "cache" / "delegation" / "live").exists()


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("source_model, source_mode, source_url, target_model, target_mode, target_url", [
    (
        "minimax-m3", "anthropic_messages", "https://opencode.ai/zen/go",
        "deepseek-v4-flash", "chat_completions", "https://opencode.ai/zen/go/v1",
    ),
    (
        "deepseek-v4-flash", "chat_completions", "https://opencode.ai/zen/go/v1",
        "minimax-m3", "anthropic_messages", "https://opencode.ai/zen/go",
    ),
])
def test_opencode_model_pin_normalizes_both_protocol_paths_without_reselecting_credentials(
    inherited, source_model, source_mode, source_url, target_model, target_mode, target_url,
):
    from types import SimpleNamespace
    from tools.delegate_tool_config import _resolve_child_runtime
    from tools.delegate_tool import _credential_overrides

    parent = SimpleNamespace(
        provider="opencode-go", requested_provider="opencode-go", model=source_model,
        base_url=source_url, api_mode=source_mode,
        _client_kwargs={"base_url": source_url, "api_key": "selected-account"},
    )
    batch = (
        _credential_bundle(None, None, None, None, None, None) if inherited else
        _credential_bundle(source_model, "opencode-go", source_url, "selected-account", source_mode, None)
    )
    with (
        patch("tools.delegate_tool_config._resolve_delegation_credentials", side_effect=AssertionError),
        patch("hermes_cli.runtime_provider_custom._get_named_custom_provider", return_value=None),
    ):
        routed, cfg = _resolve_task_credentials({"model": target_model}, batch, {}, parent)
        overrides = _credential_overrides(routed, cfg, task_pinned=True)
        overrides.pop("override_request_overrides")
        runtime = _resolve_child_runtime(parent, {}, "selected-account", model=routed["model"], **overrides)
    assert runtime["api_mode"] == target_mode
    assert runtime["base_url"] == target_url
    assert runtime["api_key"] == "selected-account"
    assert runtime["provider"] == runtime["requested_provider"] == "opencode-go"
    assert parent.base_url == source_url
    assert batch["base_url"] == (None if inherited else source_url)


@pytest.mark.parametrize("inherited", [False, True])
def test_custom_opencode_model_pin_preserves_explicit_wire_and_proxy_endpoint(inherited):
    from types import SimpleNamespace
    from tools.delegate_tool_config import _resolve_child_runtime
    from tools.delegate_tool import _credential_overrides

    name = "opencode-go-bridge"
    url = "https://bridge.example/relay/v1"
    parent = SimpleNamespace(
        provider="custom", requested_provider=name, model="deepseek-v4-flash",
        base_url=url, api_mode="chat_completions",
        _client_kwargs={"base_url": url, "api_key": "selected-account"},
    )
    batch = (
        _credential_bundle(None, None, None, None, None, None) if inherited else
        _credential_bundle("deepseek-v4-flash", name, url, "selected-account", "chat_completions", None)
    )
    with (
        patch("tools.delegate_tool_config._resolve_delegation_credentials", side_effect=AssertionError),
        patch("hermes_cli.runtime_provider_custom._get_named_custom_provider", return_value={
            "name": name, "base_url": url, "api_mode": "chat_completions",
        }),
    ):
        routed, cfg = _resolve_task_credentials({"model": "minimax-m3"}, batch, {}, parent)
        overrides = _credential_overrides(routed, cfg, task_pinned=True)
        overrides.pop("override_request_overrides")
        runtime = _resolve_child_runtime(parent, {}, "selected-account", model=routed["model"], **overrides)
    assert runtime["model"] == "minimax-m3"
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["base_url"] == url
    assert runtime["api_key"] == "selected-account"
    assert runtime["requested_provider"] == name

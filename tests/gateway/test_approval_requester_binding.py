"""Tests for requester-bound approval resolution.

The confused-deputy fix binds each pending approval to the verified requester
who triggered the dangerous command.  A different participant in a shared
session (thread/group) cannot resolve it — they get REQUESTER_MISMATCH instead.

These tests exercise the core resolution logic directly, without importing
gateway adapters or platform transports.
"""

import threading

import pytest

import tools.approval as approval
import tools.approval_context as approval_context


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Clean approval state for every test."""
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_approval_session_key",
                        approval_context._ctx("test_session_key"))
    monkeypatch.setattr(approval_context, "_approval_requester_id",
                        approval_context._ctx("test_requester_id"))
    monkeypatch.setattr(
        approval_context, "_approval_session_owner",
        approval_context.contextvars.ContextVar("test_session_owner", default=False))
    with approval._lock:
        approval._pending.clear()
        approval._session_approved.clear()
        approval._session_yolo.clear()
        approval._gateway_queues.clear()
        approval._gateway_notify_cbs.clear()
        approval._permanent_approved.clear()
        approval._requester_permanent_approved_by_home.clear()
    yield


def _enqueue(
    session_key: str, requester_id: str = "", *, requester_required: bool = True,
    session_owner_required: bool = False,
) -> None:
    """Enqueue one gateway approval with its verified authority policy."""
    from tools.approval_gateway_wait import _ApprovalEntry
    entry = _ApprovalEntry({
        "command": "rm -rf /",
        "pattern_key": "dangerous:rm",
        "description": "test",
        "requester_id": requester_id,
        "requester_required": requester_required,
        "session_owner_required": session_owner_required,
    })
    with approval._lock:
        approval._gateway_queues.setdefault(session_key, []).append(entry)


# ---------------------------------------------------------------------------
# resolve_gateway_approval requester binding
# ---------------------------------------------------------------------------


class TestRequesterMismatch:
    """Approvals bound to a verified requester reject a different clicker."""

    def test_matching_clicker_resolves(self):
        _enqueue("sess-1", requester_id="alice")
        count = approval.resolve_gateway_approval("sess-1", "once", clicker_id="alice")
        assert count == 1

    def test_mismatched_clicker_returns_sentinel(self):
        _enqueue("sess-1", requester_id="alice")
        count = approval.resolve_gateway_approval("sess-1", "once", clicker_id="bob")
        assert count == approval.REQUESTER_MISMATCH
        # Entry still in queue for the real requester.
        assert approval.has_blocking_approval("sess-1")

    def test_missing_requester_is_not_a_gateway_wildcard(self):
        _enqueue("sess-1", requester_id="")
        count = approval.resolve_gateway_approval("sess-1", "once", clicker_id="bob")
        assert count == approval.REQUESTER_MISMATCH

    def test_missing_clicker_cannot_resolve_bound_gateway_entry(self):
        _enqueue("sess-1", requester_id="alice")
        count = approval.resolve_gateway_approval("sess-1", "once", clicker_id=None)
        assert count == approval.REQUESTER_MISMATCH

    def test_resolve_all_skips_mismatched_entries(self):
        _enqueue("sess-1", requester_id="alice")
        _enqueue("sess-1", requester_id="bob")
        count = approval.resolve_gateway_approval("sess-1", "once", resolve_all=True, clicker_id="alice")
        # Only alice's entry resolved; bob's stays.
        assert count == 1
        assert approval.has_blocking_approval("sess-1")

    def test_resolve_all_mismatch_when_none_match(self):
        _enqueue("sess-1", requester_id="alice")
        count = approval.resolve_gateway_approval("sess-1", "once", resolve_all=True, clicker_id="charlie")
        assert count == approval.REQUESTER_MISMATCH

    def test_resolve_by_request_id_respects_binding(self):
        from tools.approval_gateway_wait import _ApprovalEntry
        entry = _ApprovalEntry({
            "command": "rm -rf /",
            "pattern_key": "dangerous:rm",
            "description": "test",
            "requester_id": "alice",
            "request_id": "req-42",
        })
        with approval._lock:
            approval._gateway_queues.setdefault("sess-1", []).append(entry)
        count = approval.resolve_gateway_approval("sess-1", "once", request_id="req-42", clicker_id="bob")
        assert count == approval.REQUESTER_MISMATCH

    def test_config_disable_allows_any_clicker(self, monkeypatch):
        monkeypatch.setattr(approval_context, "_get_require_requester_match", lambda: False)
        _enqueue("sess-1", requester_id="alice")
        count = approval.resolve_gateway_approval("sess-1", "once", clicker_id="bob")
        assert count == 1

    def test_native_entry_requires_verified_session_owner_even_when_requester_matching_is_disabled(
        self, monkeypatch
    ):
        monkeypatch.setattr(approval_context, "_get_require_requester_match", lambda: False)
        _enqueue(
            "native", requester_required=False, session_owner_required=True)
        assert approval.resolve_gateway_approval("native", "once") == approval.REQUESTER_MISMATCH
        assert approval.resolve_gateway_approval(
            "native", "once", session_owner=True) == 1


class TestRequesterBlocksPeek:
    """gateway_approval_requester_blocks is a non-mutating preview of the binding check."""

    def test_empty_queue_returns_false(self):
        assert not approval.gateway_approval_requester_blocks("no-such-session", "bob")

    def test_matching_clicker_not_blocked(self):
        _enqueue("sess-1", requester_id="alice")
        assert not approval.gateway_approval_requester_blocks("sess-1", "alice")

    def test_mismatched_clicker_blocked(self):
        _enqueue("sess-1", requester_id="alice")
        assert approval.gateway_approval_requester_blocks("sess-1", "bob")


# ---------------------------------------------------------------------------
# Session grants scoped by (session_key, requester_id)
# ---------------------------------------------------------------------------


class TestSessionGrantScoping:
    """approve_session / is_approved scope grants per requester."""

    def test_grant_scoped_to_requester(self):
        approval.approve_session("sess-1", "dangerous:rm", requester_id="alice")
        assert approval.is_approved("sess-1", "dangerous:rm", requester_id="alice")
        # bob in the same session does NOT inherit alice's grant.
        assert not approval.is_approved("sess-1", "dangerous:rm", requester_id="bob")

    def test_local_empty_grant_does_not_cover_gateway_requester(self):
        approval.approve_session("sess-1", "dangerous:rm", requester_id="")
        assert approval.is_approved("sess-1", "dangerous:rm", requester_id="")
        assert not approval.is_approved("sess-1", "dangerous:rm", requester_id="alice")

    def test_clear_session_removes_all_requester_grants(self):
        approval.approve_session("sess-1", "dangerous:rm", requester_id="alice")
        approval.approve_session("sess-1", "dangerous:rm", requester_id="bob")
        approval.clear_session("sess-1")
        assert not approval.is_approved("sess-1", "dangerous:rm", requester_id="alice")
        assert not approval.is_approved("sess-1", "dangerous:rm", requester_id="bob")

    def test_permanent_grant_is_platform_and_requester_scoped(self):
        from gateway.session_context import reset_session_vars, set_session_vars

        set_session_vars(platform="telegram", user_id="alice")
        try:
            approval.approve_requester_permanent("dangerous:rm", "alice")
            assert approval.is_approved("sess-1", "dangerous:rm", requester_id="alice")
            set_session_vars(platform="discord", user_id="alice")
            assert not approval.is_approved("sess-1", "dangerous:rm", requester_id="alice")
        finally:
            reset_session_vars()


# ---------------------------------------------------------------------------
# Requester contextvar lifecycle
# ---------------------------------------------------------------------------


class TestRequesterContextVar:
    """set / get / reset helpers on the requester contextvar."""

    def test_default_is_empty(self):
        assert approval_context.get_current_requester_id() == ""

    def test_set_and_get(self):
        token = approval_context.set_current_requester_id("alice")
        assert approval_context.get_current_requester_id() == "alice"
        approval_context.reset_current_requester_id(token)
        assert approval_context.get_current_requester_id() == ""

    def test_none_coerced_to_empty(self):
        token = approval_context.set_current_requester_id(None)
        assert approval_context.get_current_requester_id() == ""
        approval_context.reset_current_requester_id(token)


# ---------------------------------------------------------------------------
# Gateway gate threading (request_tool_approval / MCP elicitation)
# ---------------------------------------------------------------------------


class TestGatewayGateRequesterBinding:
    """The gateway decision path records and enforces the requester."""

    def test_gateway_callback_binds_resolution_to_requester(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
        token = approval_context.set_current_requester_id("requester")
        sk_token = approval_context.set_current_session_key("test-session")
        resolutions = []

        def notify(_data):
            resolutions.append(
                approval.resolve_gateway_approval(
                    "test-session", "once", clicker_id="other-user"
                )
            )
            resolutions.append(
                approval.resolve_gateway_approval(
                    "test-session", "once", clicker_id="requester"
                )
            )

        approval.register_gateway_notify("test-session", notify)
        try:
            result = approval.request_tool_approval(
                "browser_navigate", "external URL", rule_key="ext-nav"
            )
        finally:
            approval.unregister_gateway_notify("test-session")
            approval_context.reset_current_requester_id(token)
            approval_context.reset_current_session_key(sk_token)

        assert resolutions == [approval.REQUESTER_MISMATCH, 1]
        assert result["approved"] is True

    def test_remote_native_owner_does_not_inherit_or_create_local_grants(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.setattr(approval_context, "_get_require_requester_match", lambda: True)
        session_token = approval_context.set_current_session_key("remote-native")
        owner_token = approval_context.set_current_approval_session_owner("user:oidc:alice")
        approval.approve_session("remote-native", "plugin_rule:remote-native")
        local_permanent = approval._permanent_set()
        local_permanent.add("plugin_rule:remote-native")
        seen = []

        def notify(data):
            seen.append(data)
            assert data["allow_session"] is False
            assert data["allow_permanent"] is False
            approval.resolve_gateway_approval(
                "remote-native", "always", session_owner=True)

        approval.register_gateway_notify("remote-native", notify)
        try:
            assert not approval.is_approved(
                "remote-native", "plugin_rule:remote-native")
            assert approval.permanent_patterns_for_requester() == set()
            approval._session_approved.pop(("remote-native", ""), None)
            local_permanent.discard("plugin_rule:remote-native")
            result = approval.request_tool_approval(
                "synthetic_mutation", "Mutate synthetic fixture",
                rule_key="remote-native")
        finally:
            approval.unregister_gateway_notify("remote-native")
            local_permanent.discard("plugin_rule:remote-native")
            approval_context.reset_current_approval_session_owner(owner_token)
            approval_context.reset_current_session_key(session_token)

        assert len(seen) == 1
        assert result["approved"] is True
        assert not approval._session_approved.get(("remote-native", ""))
        assert "plugin_rule:remote-native" not in local_permanent


# ---------------------------------------------------------------------------
# require_human: strict per-operation human confirmation
# ---------------------------------------------------------------------------


    def test_same_command_from_two_requesters_does_not_coalesce(self):
        """A grant-capable wait can coalesce only with the same verified principal."""
        from tools.approval_gateway_wait import _await_gateway_decision

        notified = []
        both_notified = threading.Event()
        results = {}

        def notify(_data):
            notified.append(_data)
            if len(notified) == 2:
                both_notified.set()

        def wait_for(requester):
            token = approval_context.set_current_requester_id(requester)
            try:
                results[requester] = _await_gateway_decision(
                    "shared-session", notify,
                    {"command": "rm -rf /tmp/work", "pattern_key": "recursive delete",
                     "pattern_keys": ["recursive delete"], "allow_session": True,
                     "allow_permanent": True},
                )
            finally:
                approval_context.reset_current_requester_id(token)

        alice = threading.Thread(target=wait_for, args=("alice",))
        bob = threading.Thread(target=wait_for, args=("bob",))
        alice.start()
        bob.start()
        assert both_notified.wait(timeout=2)
        assert approval.resolve_gateway_approval("shared-session", "session", clicker_id="alice") == 1
        assert approval.resolve_gateway_approval("shared-session", "session", clicker_id="bob") == 1
        alice.join(timeout=2)
        bob.join(timeout=2)
        assert not alice.is_alive() and not bob.is_alive()
        assert results["alice"]["choice"] == "session"
        assert results["bob"]["choice"] == "session"

class TestRequireHuman:
    """require_human=True skips every automatic bypass and demands a real human."""

    def test_yolo_does_not_bypass_when_require_human(self, monkeypatch):
        """--yolo auto-approves normal gates but NOT require_human ones."""
        monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
        # With a CLI surface present, require_human still prompts.
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval",
                            lambda *a, **k: "once")
        import tools.approval_prompt as prompt_mod
        monkeypatch.setattr(prompt_mod, "prompt_dangerous_approval",
                            lambda *a, **k: "once")
        result = approval.request_tool_approval(
            "mail_move", "Move 5 msgs INBOX->Archive", rule_key="test-yolo",
            require_human=True,
        )
        # Yolo is on, but require_human forced a real prompt → approved via mock.
        assert result["approved"] is True

    def test_yolo_bypasses_normal_gate(self, monkeypatch):
        """Sanity: without require_human, yolo auto-approves without prompting."""
        monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
        result = approval.request_tool_approval(
            "mail_move", "Move 5 msgs INBOX->Archive", rule_key="test-yolo-normal",
        )
        assert result["approved"] is True

    def test_mode_off_does_not_bypass_when_require_human(self, monkeypatch):
        """approvals.mode=off auto-approves normal gates but NOT require_human."""
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "off")
        # No human surface → blocks.
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        result = approval.request_tool_approval(
            "mail_move", "Move 5 msgs", rule_key="test-off",
            require_human=True,
        )
        assert result["approved"] is False
        assert "human confirmation required" in result["message"].lower() or "blocked" in result["message"].lower()

    def test_session_cache_does_not_bypass_when_require_human(self, monkeypatch):
        """A session-cached approval does NOT skip require_human."""
        sk_token = approval_context.set_current_session_key("cached-sess")
        approval.approve_session("cached-sess", "plugin_rule:test-cache")
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval",
                            lambda *a, **k: "once")
        import tools.approval_prompt as prompt_mod
        monkeypatch.setattr(prompt_mod, "prompt_dangerous_approval",
                            lambda *a, **k: "once")
        try:
            result = approval.request_tool_approval(
                "test_tool", "reason", rule_key="test-cache",
                require_human=True,
            )
            # Should NOT short-circuit from cache; prompted and approved via mock.
            assert result["approved"] is True
        finally:
            approval_context.reset_current_session_key(sk_token)

    def test_no_human_surface_blocks(self, monkeypatch):
        """require_human with no interactive/gateway surface blocks unconditionally."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        result = approval.request_tool_approval(
            "mail_trash", "Trash 10 messages", rule_key="test-no-human",
            require_human=True,
        )
        assert result["approved"] is False
        assert "blocked" in result["message"].lower()

    def test_native_session_owner_without_messaging_id_gets_fresh_once_prompt(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
        monkeypatch.setattr(approval_context, "_get_require_requester_match", lambda: True)
        session_token = approval_context.set_current_session_key("native-session")
        owner_token = approval_context.set_current_approval_session_owner(True)
        attempts = []

        def notify(data):
            assert data["requester_id"] == ""
            assert data["requester_required"] is False
            assert data["session_owner_required"] is True
            assert data["allow_session"] is False
            assert data["allow_permanent"] is False
            attempts.append(approval.resolve_gateway_approval("native-session", "once"))
            attempts.append(approval.resolve_gateway_approval(
                "native-session", "always", session_owner=True))

        approval.register_gateway_notify("native-session", notify)
        try:
            result = approval.request_tool_approval(
                "synthetic_mutation", "Mutate synthetic fixture",
                rule_key="native-owner", require_human=True)
        finally:
            approval.unregister_gateway_notify("native-session")
            approval_context.reset_current_approval_session_owner(owner_token)
            approval_context.reset_current_session_key(session_token)

        assert attempts == [approval.REQUESTER_MISMATCH, 1]
        assert result["approved"] is True
        assert ("native-session", "") not in approval._session_approved

    def test_messaging_gateway_without_verified_actor_denies_before_notification(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
        monkeypatch.setattr(approval_context, "_get_require_requester_match", lambda: True)
        session_token = approval_context.set_current_session_key("messaging-session")
        notified = []
        approval.register_gateway_notify("messaging-session", notified.append)
        try:
            result = approval.request_tool_approval(
                "synthetic_mutation", "Mutate synthetic fixture",
                rule_key="missing-actor", require_human=True)
        finally:
            approval.unregister_gateway_notify("messaging-session")
            approval_context.reset_current_session_key(session_token)

        assert result["approved"] is False
        assert "no verified requester identity" in result["message"]
        assert notified == []

    def test_gateway_resolves_require_human_once_only(self, monkeypatch):
        """Strict gateway approval cannot create a reusable session or permanent grant."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
        sk_token = approval_context.set_current_session_key("rh-gw-sess")
        requester_token = approval_context.set_current_requester_id("alice")

        def notify(data):
            assert data["allow_session"] is False
            assert data["allow_permanent"] is False
            approval.resolve_gateway_approval("rh-gw-sess", "always", clicker_id="alice")

        approval.register_gateway_notify("rh-gw-sess", notify)
        try:
            result = approval.request_tool_approval(
                "mail_label", "Label 3 messages as Important",
                rule_key="test-gw-rh", require_human=True,
            )
        finally:
            approval.unregister_gateway_notify("rh-gw-sess")
            approval_context.reset_current_requester_id(requester_token)
            approval_context.reset_current_session_key(sk_token)

        assert result["approved"] is True
        assert not approval.is_approved("rh-gw-sess", "plugin_rule:test-gw-rh", requester_id="alice")

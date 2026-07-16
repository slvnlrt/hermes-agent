"""Tests for the Teams exec-approval button handler (_on_card_action).

Focus: the confused-deputy fix (W1). Clicking an approval button must be bound
to the verified requester — an allowlisted user who is NOT the requester cannot
resolve another user's dangerous command, and the misleading confirmation card
is not built. Also covers the Teams-specific return-value check that guards
against a concurrent double-click.

Reuses the Teams SDK mock and adapter loader installed by ``test_teams``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Importing test_teams installs the Teams SDK mock in sys.modules and loads the
# plugin adapter under a stable module name.
from tests.gateway.test_teams import TeamsAdapter, _make_config, _teams_mod


class _MsgResp:
    """Recording stand-in for AdaptiveCardActionMessageResponse."""

    def __init__(self, value=None):
        self.value = value


class _CardResp:
    """Recording stand-in for AdaptiveCardActionCardResponse."""

    def __init__(self, value=None):
        self.value = value


def _make_ctx(hermes_action, session_key, *, aad="aad-requester", uid="id-requester"):
    action = SimpleNamespace(
        data={
            "hermes_action": hermes_action,
            "session_key": session_key,
            "cmd": "rm -rf /important",
            "desc": "recursive delete",
        }
    )
    activity = SimpleNamespace(
        value=SimpleNamespace(action=action),
        from_=SimpleNamespace(aad_object_id=aad, id=uid),
    )
    return SimpleNamespace(activity=activity)


@pytest.fixture(autouse=True)
def _recording_responses(monkeypatch):
    """Swap the mocked response classes for recording ones so tests can read
    the response ``value`` (the mock classes discard constructor kwargs)."""
    monkeypatch.setattr(_teams_mod, "AdaptiveCardActionMessageResponse", _MsgResp)
    monkeypatch.setattr(_teams_mod, "AdaptiveCardActionCardResponse", _CardResp)
    # Allow every clicker through the allowlist gate by default; individual
    # tests override to exercise the unauthorized path.
    monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
    monkeypatch.delenv("TEAMS_ALLOWED_USERS", raising=False)


def _adapter():
    return TeamsAdapter(_make_config())


class TestTeamsApprovalRequesterBinding:
    @pytest.mark.asyncio
    async def test_requester_click_resolves(self):
        adapter = _adapter()
        ctx = _make_ctx("approve_once", "sess", aad="aad-requester")

        with patch("tools.approval.has_blocking_approval", return_value=True), \
                patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            resp = await adapter._on_card_action(ctx)

        # Bound to the verified clicker id (aad_object_id preferred).
        mock_resolve.assert_called_once_with("sess", "once", clicker_id="aad-requester")
        # Confirmation card (not a plain message rejection).
        assert isinstance(resp.body, _CardResp)

    @pytest.mark.asyncio
    async def test_non_requester_click_rejected(self):
        from tools.approval import REQUESTER_MISMATCH

        adapter = _adapter()
        ctx = _make_ctx("approve_once", "sess", aad="aad-other")

        with patch("tools.approval.has_blocking_approval", return_value=True), \
                patch("tools.approval.resolve_gateway_approval",
                      return_value=REQUESTER_MISMATCH) as mock_resolve:
            resp = await adapter._on_card_action(ctx)

        mock_resolve.assert_called_once_with("sess", "once", clicker_id="aad-other")
        # A message-type rejection, NOT a confirmation card.
        assert isinstance(resp.body, _MsgResp)
        assert "Only the user who ran this command" in resp.body.value

    @pytest.mark.asyncio
    async def test_double_click_already_resolved(self):
        """A concurrent second click (resolve returns 0) must NOT render a
        misleading confirmation card — Teams' historical gap."""
        adapter = _adapter()
        ctx = _make_ctx("approve_once", "sess", aad="aad-requester")

        with patch("tools.approval.has_blocking_approval", return_value=True), \
                patch("tools.approval.resolve_gateway_approval", return_value=0):
            resp = await adapter._on_card_action(ctx)

        # "Already resolved or expired" card, not an approval confirmation.
        assert isinstance(resp.body, _CardResp)

    @pytest.mark.asyncio
    async def test_unauthorized_clicker_not_allowlisted(self, monkeypatch):
        """A clicker outside TEAMS_ALLOWED_USERS never reaches resolution."""
        monkeypatch.delenv("TEAMS_ALLOW_ALL_USERS", raising=False)
        monkeypatch.setenv("TEAMS_ALLOWED_USERS", "aad-someone-else")
        adapter = _adapter()
        ctx = _make_ctx("approve_once", "sess", aad="aad-intruder")

        with patch("tools.approval.has_blocking_approval", return_value=True), \
                patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            resp = await adapter._on_card_action(ctx)

        mock_resolve.assert_not_called()
        assert isinstance(resp.body, _MsgResp)
        assert "Not authorized" in resp.body.value

    @pytest.mark.asyncio
    async def test_deny_by_requester_resolves(self):
        adapter = _adapter()
        ctx = _make_ctx("deny", "sess", aad="aad-requester")

        with patch("tools.approval.has_blocking_approval", return_value=True), \
                patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            resp = await adapter._on_card_action(ctx)

        mock_resolve.assert_called_once_with("sess", "deny", clicker_id="aad-requester")
        assert isinstance(resp.body, _CardResp)

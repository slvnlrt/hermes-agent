import types

import pytest
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig


class TestMatrixExecApprovalReactions:


    @pytest.mark.asyncio
    async def test_reaction_keeps_pending_for_wrong_actor_then_resolves_for_requester(self, monkeypatch):
        """Only the verified Matrix requester can resolve the live approval."""
        requester = "@requester:example.test"
        other_actor = "@other:example.test"
        session_key = "matrix-session-1"
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", f"{requester},{other_actor}")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt
        from tools import approval
        from tools.approval_gateway_wait import _ApprovalEntry

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.test"},
            )
        )
        adapter._user_id = "@bot:example.test"
        adapter._redact_bot_approval_reactions = AsyncMock()
        adapter._send_invalid_reaction_feedback = AsyncMock()
        prompt = _MatrixApprovalPrompt(
            session_key=session_key,
            chat_id="!room:example.test",
            message_id="$approval:example.test",
            requester_user_id=requester,
        )
        adapter._approval_prompts_by_event[prompt.message_id] = prompt
        adapter._approval_prompt_by_session[session_key] = prompt.message_id
        entry = _ApprovalEntry(
            {
                "command": "rm -rf /tmp/example",
                "description": "test approval",
                "requester_id": requester,
                "requester_required": True,
            }
        )
        with approval._lock:
            approval._gateway_queues.pop(session_key, None)
            approval._gateway_queues[session_key] = [entry]
        try:
            wrong_reaction = types.SimpleNamespace(
                sender=other_actor,
                event_id="$wrong-reaction:example.test",
                room_id=prompt.chat_id,
                content={"m.relates_to": {"event_id": prompt.message_id, "key": "✅"}},
            )
            await adapter._on_reaction(wrong_reaction)

            assert approval.has_blocking_approval(session_key)
            assert not entry.event.is_set()
            assert prompt.message_id in adapter._approval_prompts_by_event

            valid_reaction = types.SimpleNamespace(
                sender=requester,
                event_id="$valid-reaction:example.test",
                room_id=prompt.chat_id,
                content={"m.relates_to": {"event_id": prompt.message_id, "key": "✅"}},
            )
            await adapter._on_reaction(valid_reaction)

            assert entry.event.is_set()
            assert entry.result == "once"
            assert not approval.has_blocking_approval(session_key)
            assert prompt.message_id not in adapter._approval_prompts_by_event
            assert session_key not in adapter._approval_prompt_by_session
        finally:
            with approval._lock:
                approval._gateway_queues.pop(session_key, None)

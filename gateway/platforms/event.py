"""Inbound message event types shared by every gateway platform adapter.

A leaf module: adapters, helpers and the runner import it, so it must not import from
gateway.platforms.*.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from gateway.session import SessionSource

logger = logging.getLogger(__name__)


class InjectionReceipt:
    """Process-local terminal receipt for one accepted plugin injection.

    ``terminal`` is safe to call from the scheduling thread, event loop, or an adapter
    cancellation callback.  It guarantees at-most-once callback invocation within this
    process; a process crash before invocation remains inherently unacknowledged.
    """

    def __init__(self, callback: Callable[[str], None] | None = None,
                 expires_at: float | None = None) -> None:
        self.callback = callback
        self.expires_at = expires_at
        self._lock = threading.Lock()
        self.outcome: str | None = None
        self.started = False

    def is_expired(self, now: float | None = None) -> bool:
        return self.expires_at is not None and self.expires_at <= (time.time() if now is None else now)
    def start(self) -> bool:
        """Claim execution unless already terminal; callers check queued expiry first."""
        with self._lock:
            if self.outcome is not None:
                return False
            self.started = True
            return True

    def terminal(self, outcome: str) -> bool:
        with self._lock:
            if self.outcome is not None:
                return False
            self.outcome = outcome
        if self.callback is not None:
            try:
                self.callback(outcome)
            except Exception:
                logger.warning("Plugin injection completion callback failed", exc_info=True)
        return True

    def __call__(self, outcome: str) -> bool:
        return self.terminal(outcome)


def event_receipts(event: Any) -> list[InjectionReceipt]:
    """The event's process-local injection receipts, never serialized in metadata."""
    return list(getattr(event, "_injection_receipts", ()) or ())


def terminalize_event_receipts(event: Any, outcome: str) -> None:
    for receipt in event_receipts(event):
        receipt.terminal(outcome)


def mark_event_receipts_started(event: Any) -> bool:
    """Claim all receipts currently owned by an event; False when expiry won the race."""
    receipts = event_receipts(event)
    return bool(receipts) and all(receipt.start() for receipt in receipts)


def expire_event_receipts(event: Any, now: float | None = None) -> bool:
    """Terminalize waiting receipts and return whether the receipt-bearing event is empty."""
    receipts = event_receipts(event)
    if not receipts:
        return False
    live = [receipt for receipt in receipts if receipt.started or not receipt.is_expired(now)]
    for receipt in receipts:
        if receipt not in live:
            receipt.terminal("expired")
    event._injection_receipts = live
    return not live
def transfer_event_receipts(source: Any, target: Any) -> None:
    """Move receipt ownership with a coalesced or runner-drained event."""
    receipts = event_receipts(source)
    if not receipts:
        return
    handle = getattr(source, "_injection_deadline_handle", None)
    if handle is not None:
        handle.cancel()
    target._injection_receipts = [*event_receipts(target), *receipts]
    source._injection_receipts = []
    rearm = getattr(source, "_injection_deadline_rearm", None)
    if callable(rearm):
        rearm(target)


class MessageType(Enum):
    """Types of incoming messages."""
    TEXT = "text"
    LOCATION = "location"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    DOCUMENT = "document"
    STICKER = "sticker"
    COMMAND = "command"  # /command style


class ProcessingOutcome(Enum):
    """Result classification for message-processing lifecycle hooks."""
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass
class MessageEvent:
    """Incoming message from a platform — the normalized shape all adapters produce."""
    text: str
    message_type: MessageType = MessageType.TEXT
    # Author, mirrored from ``source`` for per-message prompt builders; None for non-IM sources.
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    # None only in isolated unit tests; production always sets it. Typing it Optional
    # exposes ~60 unguarded ``.source.<attr>`` reads, so that is a separate change.
    source: SessionSource = None
    raw_message: Any = None
    message_id: Optional[str] = None
    # Delivery-ledger identity for the final send, when it differs from ``message_id``. A queued
    # (/queue) chain answers the LAST message of the chain, so its final send has to be ledgered
    # under that message's id. Keyed on the opening event's id instead, two chained turns carrying
    # the same text collide on one obligation id and the earlier turn's row is overwritten (a
    # refused first reply then reads as delivered). Reply routing is unaffected: the reply anchor
    # still comes from this event.
    ledger_message_id: Optional[str] = None
    # Platform update id (Telegram ``update_id``): ``/restart`` records it so the new gateway
    # advances past it even if PTB's shutdown ACK times out.
    platform_update_id: Optional[int] = None
    # Media attachments: local file paths (for vision tool access)
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    # Per-attachment text-inlining contract; None = legacy "text/* already inlined into ``text``".
    media_text_inlined: List[Optional[bool]] = field(default_factory=list)
    reply_to_message_id: Optional[str] = None
    reply_to_text: Optional[str] = None  # Text of the replied-to message (for context injection)
    reply_to_author_id: Optional[str] = None
    reply_to_author_name: Optional[str] = None
    reply_to_is_own_message: bool = False  # True when the user replied to this bot/assistant's message
    # Structured interactive-prompt reply (relay only): {prompt_id, option_id, label?,
    # prompt_message_id?}; routed to the approval/slash-confirm/clarify resolvers BEFORE dispatch.
    prompt_response: Optional[Dict[str, Any]] = None
    # Auto-loaded skill(s) for topic/channel bindings; a single name or ordered list.
    auto_skill: Optional[str | list[str]] = None
    # Per-channel ephemeral system prompt; applied at API call time, never persisted to transcript.
    channel_prompt: Optional[str] = None
    # History-backfilled channel context (missed under require_mention); kept out of ``text`` so
    # run.py's sender-prefix logic sees only the trigger message.
    channel_context: Optional[str] = None
    # Set for synthetic events (e.g. background-process notifications) that must bypass user authorization.
    internal: bool = False
    # Free-form per-event metadata (e.g. ``whatsapp_from_owner=True``); plugins must ``.get()``.
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    # May this event resolve gateway commands / control prompts? Proactive plugin events set False
    # so untrusted payload text stays conversational. Kept last for positional compat.
    allow_gateway_control: bool = True

    # Process-local admission receipt, never routing metadata or execution acknowledgement.
    _gateway_accepted: bool = field(default=False, init=False, repr=False, compare=False)
    def __post_init__(self) -> None:
        # Deliberately an ordinary instance attribute: dataclasses.asdict() and every
        # serializable event path must never traverse callbacks, locks, or deadlines.
        self._injection_receipts: list[InjectionReceipt] = []

    def is_command(self) -> bool:
        """Check if this is a command message (e.g., /new, /reset)."""
        return self.allow_gateway_control and (self.text or "").lstrip().startswith("/")

    def get_command(self) -> Optional[str]:
        """Extract command name if this is a command message."""
        if not self.is_command():
            return None
        raw = (self.text or "").lstrip().split(maxsplit=1)[0][1:].lower().split("@", 1)[0]
        # Reject file paths: valid command names never contain /
        return None if "/" in raw else raw

    def get_command_args(self) -> str:
        """Get the arguments after a command."""
        if not self.is_command():
            return self.text
        parts = (self.text or "").lstrip().split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        # iOS auto-corrects -- to — (em dash) and - to – (en dash)
        return args.replace("\u2014\u2014", "--").replace("\u2014", "--").replace("\u2013", "-")

"""Peer-to-peer agent communication protocol + message bus.

Extends the Phase 2 AgentMessage envelope with enterprise message kinds
and delivery modes (request / reply / broadcast / event). Agents no longer
depend solely on shared GraphState for collaboration.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Message kinds & delivery modes
# ---------------------------------------------------------------------------

MessageKind = Literal[
    # Phase 2 (preserved)
    "goal",
    "task",
    "observation",
    "result",
    "feedback",
    "plan",
    "decision",
    # Phase 3 enterprise P2P
    "task_request",
    "task_result",
    "question",
    "clarification",
    "status_update",
    "approval_request",
    "approval_response",
    "help_request",
    "capability_offer",
    "event",
    # Live A2A extensions
    "command",
    "data_request",
    "context_share",
    "reject",
]

DeliveryMode = Literal["request", "reply", "broadcast", "event", "command"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _msg_id() -> str:
    return uuid4().hex[:12]


class AgentMessage(BaseModel):
    """Envelope for inter-agent communication (hub-mediated + P2P capable)."""

    id: str = Field(default_factory=_msg_id)
    kind: MessageKind
    sender: str
    recipient: str = "coordinator"
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str = ""
    reply_to: str = ""
    delivery: DeliveryMode = "event"
    priority: int = Field(default=5, ge=0, le=9)  # 0=highest, 9=lowest
    ttl_seconds: float = 0.0  # 0 = no expiry
    expires_at: float = 0.0
    rejected: bool = False
    reject_reason: str = ""
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def is_expired(self, now: float | None = None) -> bool:
        if self.expires_at <= 0:
            return False
        return (now if now is not None else time.time()) >= self.expires_at


# ---------------------------------------------------------------------------
# Convenience constructors (Phase 2 + Phase 3)
# ---------------------------------------------------------------------------

def goal_message(sender: str, goal: dict[str, Any], *, correlation_id: str = "") -> AgentMessage:
    return AgentMessage(
        kind="goal",
        sender=sender,
        recipient="planner",
        payload=goal,
        correlation_id=correlation_id,
        delivery="event",
    )


def task_message(sender: str, task: dict[str, Any], *, recipient: str = "coordinator") -> AgentMessage:
    return AgentMessage(kind="task", sender=sender, recipient=recipient, payload=task)


def observation_message(sender: str, observation: dict[str, Any]) -> AgentMessage:
    return AgentMessage(
        kind="observation",
        sender=sender,
        recipient="coordinator",
        payload=observation,
    )


def result_message(sender: str, result: dict[str, Any], *, recipient: str = "coordinator") -> AgentMessage:
    return AgentMessage(kind="result", sender=sender, recipient=recipient, payload=result)


def feedback_message(sender: str, feedback: dict[str, Any], *, recipient: str = "replan") -> AgentMessage:
    return AgentMessage(kind="feedback", sender=sender, recipient=recipient, payload=feedback)


def plan_message(sender: str, plan: dict[str, Any]) -> AgentMessage:
    return AgentMessage(
        kind="plan",
        sender=sender,
        recipient="coordinator",
        payload=plan,
    )


def decision_message(sender: str, decision: dict[str, Any]) -> AgentMessage:
    return AgentMessage(
        kind="decision",
        sender=sender,
        recipient="observability",
        payload=decision,
    )


def task_request_message(
    sender: str,
    request: dict[str, Any],
    *,
    recipient: str,
    correlation_id: str = "",
) -> AgentMessage:
    return AgentMessage(
        kind="task_request",
        sender=sender,
        recipient=recipient,
        payload=request,
        correlation_id=correlation_id or _msg_id(),
        delivery="request",
    )


def task_result_message(
    sender: str,
    result: dict[str, Any],
    *,
    recipient: str,
    reply_to: str = "",
    correlation_id: str = "",
) -> AgentMessage:
    return AgentMessage(
        kind="task_result",
        sender=sender,
        recipient=recipient,
        payload=result,
        reply_to=reply_to,
        correlation_id=correlation_id,
        delivery="reply",
    )


def question_message(sender: str, question: dict[str, Any], *, recipient: str) -> AgentMessage:
    return AgentMessage(
        kind="question",
        sender=sender,
        recipient=recipient,
        payload=question,
        delivery="request",
    )


def clarification_message(
    sender: str, clarification: dict[str, Any], *, recipient: str = "user"
) -> AgentMessage:
    return AgentMessage(
        kind="clarification",
        sender=sender,
        recipient=recipient,
        payload=clarification,
        delivery="request",
    )


def status_update_message(
    sender: str,
    status: dict[str, Any],
    *,
    recipient: str = "observability",
) -> AgentMessage:
    return AgentMessage(
        kind="status_update",
        sender=sender,
        recipient=recipient,
        payload=status,
        delivery="event",
    )


def approval_request_message(
    sender: str, request: dict[str, Any], *, recipient: str = "approval_gate"
) -> AgentMessage:
    return AgentMessage(
        kind="approval_request",
        sender=sender,
        recipient=recipient,
        payload=request,
        delivery="request",
    )


def broadcast_message(
    sender: str,
    kind: MessageKind,
    payload: dict[str, Any],
) -> AgentMessage:
    return AgentMessage(
        kind=kind,
        sender=sender,
        recipient="*",
        payload=payload,
        delivery="broadcast",
    )


def append_message(
    existing: list[dict[str, Any]] | None,
    message: AgentMessage,
    *,
    max_messages: int = 80,
) -> list[dict[str, Any]]:
    """Append a protocol message for GraphState (operator.add friendly return)."""
    entry = message.to_state_dict()
    _ = existing
    _ = max_messages
    return [entry]


# ---------------------------------------------------------------------------
# In-process P2P message bus (complements GraphState agent_messages)
# ---------------------------------------------------------------------------

MessageHandler = Callable[[AgentMessage], None]


def _bus_trace(
    bus_event: str,
    *,
    sender: str = "",
    receiver: str = "",
    msg_type: str = "",
    payload_summary: Any = None,
    direction: str = "sent",
) -> None:
    try:
        from observability.runtime_trace import safe_trace

        safe_trace(
            "message_bus",
            bus_event=bus_event,
            sender=sender,
            receiver=receiver,
            msg_type=msg_type,
            payload_summary=payload_summary,
            direction=direction,
        )
    except Exception:  # noqa: BLE001
        pass


class AgentMessageBus:
    """Thread-safe peer-to-peer / broadcast / request-reply / pub-sub bus.

    Agents publish structured messages here. Delivery is also mirrored into
    GraphState ``agent_messages`` by callers for checkpoint durability.
    Supports heartbeat, agent discovery, message queue, history, priority,
    TTL, reject, and blocking request/reply with timeout.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._inbox: dict[str, list[AgentMessage]] = defaultdict(list)
        self._handlers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._topic_handlers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._history: list[AgentMessage] = []
        self._max_history = 500
        self._agents: dict[str, dict[str, Any]] = {}
        self._queue: list[AgentMessage] = []
        self._pending_replies: dict[str, threading.Event] = {}
        self._reply_boxes: dict[str, AgentMessage] = {}

    def register_agent(
        self,
        agent_id: str,
        *,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._agents[agent_id] = {
                "agent_id": agent_id,
                "capabilities": list(capabilities or []),
                "metadata": dict(metadata or {}),
                "last_heartbeat": time.time(),
                "status": "online",
            }

    def subscribe(self, agent_id: str, handler: MessageHandler) -> None:
        with self._lock:
            self._handlers[agent_id].append(handler)
            if agent_id not in self._agents:
                self.register_agent(agent_id)
        _bus_trace(
            "Subscribe",
            sender="bus",
            receiver=agent_id,
            msg_type="subscribe",
            direction="received",
        )

    def subscribe_topic(self, topic: str, handler: MessageHandler) -> None:
        with self._lock:
            self._topic_handlers[topic].append(handler)
        _bus_trace(
            "Subscribe",
            sender="bus",
            receiver=topic,
            msg_type=f"topic:{topic}",
            direction="received",
        )

    def publish(self, message: AgentMessage) -> AgentMessage:
        # Apply TTL → absolute expiry
        if message.ttl_seconds > 0 and message.expires_at <= 0:
            message.expires_at = time.time() + message.ttl_seconds
        if message.is_expired():
            message.rejected = True
            message.reject_reason = "ttl_expired"
            _bus_trace(
                "Reject",
                sender=message.sender,
                receiver=message.recipient,
                msg_type=message.kind,
                payload_summary={"reason": "ttl_expired"},
                direction="sent",
            )
            return message

        with self._lock:
            self._history.append(message)
            self._queue.append(message)
            # Priority queue ordering (stable: lower priority number first)
            self._queue.sort(key=lambda m: (m.priority, m.created_at.timestamp()))
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]
            if len(self._queue) > self._max_history:
                self._queue = self._queue[-self._max_history :]

            recipients: list[str]
            if message.delivery == "broadcast" or message.recipient == "*":
                recipients = list(self._handlers.keys()) or list(self._agents.keys()) or ["*"]
                bus_event = "Broadcast"
            elif message.delivery == "request":
                recipients = [message.recipient]
                bus_event = "Request"
            elif message.delivery == "reply":
                recipients = [message.recipient]
                bus_event = "Reply"
            elif message.delivery == "command":
                recipients = [message.recipient]
                bus_event = "Command"
            else:
                recipients = [message.recipient]
                bus_event = "Publish"

            # Complete pending request/reply waiters
            if message.delivery == "reply" and message.correlation_id:
                self._reply_boxes[message.correlation_id] = message
                ev = self._pending_replies.get(message.correlation_id)
                if ev is not None:
                    ev.set()

            for rid in recipients:
                self._inbox[rid].append(message)
                for handler in self._handlers.get(rid, []):
                    try:
                        handler(message)
                    except Exception as exc:  # noqa: BLE001
                        logger = __import__(
                            "utils.logging_config", fromlist=["get_logger"]
                        ).get_logger(__name__)
                        logger.debug("Message handler error recipient=%s: %s", rid, exc)

            # Topic / pub-sub by message kind
            for handler in self._topic_handlers.get(message.kind, []):
                try:
                    handler(message)
                except Exception:  # noqa: BLE001
                    pass

        payload_keys = list((message.payload or {}).keys())[:12]
        _bus_trace(
            bus_event if not (message.payload or {}).get("heartbeat") else "Heartbeat",
            sender=message.sender,
            receiver=message.recipient,
            msg_type=message.kind,
            payload_summary={
                "keys": payload_keys,
                "delivery": message.delivery,
                "priority": message.priority,
                "correlation_id": message.correlation_id,
            },
            direction="sent",
        )
        _bus_trace(
            "Message Queue",
            sender=message.sender,
            receiver=message.recipient,
            msg_type=message.kind,
            payload_summary={"queue_depth": len(self._queue)},
            direction="sent",
        )
        return message

    def heartbeat(self, agent_id: str, **metadata: Any) -> AgentMessage:
        with self._lock:
            info = self._agents.get(agent_id) or {
                "agent_id": agent_id,
                "capabilities": [],
                "metadata": {},
            }
            info["last_heartbeat"] = time.time()
            info["status"] = "online"
            info["metadata"].update(metadata)
            self._agents[agent_id] = info
        return self.publish(
            AgentMessage(
                kind="status_update",
                sender=agent_id,
                recipient="observability",
                payload={"heartbeat": True, "status": "online", **metadata},
                delivery="event",
            )
        )

    def discover(self, *, capability: str | None = None) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            agents = list(self._agents.values())
        out: list[dict[str, Any]] = []
        for a in agents:
            age = now - float(a.get("last_heartbeat") or 0)
            status = "online" if age < 60 else ("stale" if age < 300 else "offline")
            entry = {**a, "status": status, "age_seconds": round(age, 2)}
            if capability and capability not in (a.get("capabilities") or []):
                continue
            out.append(entry)
        return out

    def enqueue(self, message: AgentMessage) -> AgentMessage:
        """Alias for publish — durable in-process message queue."""
        return self.publish(message)

    def request(
        self,
        sender: str,
        recipient: str,
        kind: MessageKind,
        payload: dict[str, Any],
        *,
        correlation_id: str | None = None,
        priority: int = 5,
        ttl_seconds: float = 0.0,
    ) -> AgentMessage:
        msg = AgentMessage(
            kind=kind,
            sender=sender,
            recipient=recipient,
            payload=payload,
            correlation_id=correlation_id or _msg_id(),
            delivery="request",
            priority=priority,
            ttl_seconds=ttl_seconds,
        )
        return self.publish(msg)

    def request_reply(
        self,
        sender: str,
        recipient: str,
        kind: MessageKind,
        payload: dict[str, Any],
        *,
        timeout_seconds: float = 10.0,
        priority: int = 5,
        ttl_seconds: float = 0.0,
        correlation_id: str | None = None,
    ) -> AgentMessage | None:
        """Blocking request/reply with timeout. Returns reply or None on timeout."""
        cid = correlation_id or _msg_id()
        event = threading.Event()
        with self._lock:
            self._pending_replies[cid] = event
        self.request(
            sender,
            recipient,
            kind,
            payload,
            correlation_id=cid,
            priority=priority,
            ttl_seconds=ttl_seconds or timeout_seconds,
        )
        ok = event.wait(timeout=timeout_seconds)
        with self._lock:
            self._pending_replies.pop(cid, None)
            reply_msg = self._reply_boxes.pop(cid, None)
        if not ok or reply_msg is None:
            _bus_trace(
                "Timeout",
                sender=sender,
                receiver=recipient,
                msg_type=kind,
                payload_summary={"correlation_id": cid, "timeout": timeout_seconds},
                direction="sent",
            )
            return None
        return reply_msg

    def reply(
        self,
        original: AgentMessage,
        sender: str,
        payload: dict[str, Any],
        *,
        kind: MessageKind = "task_result",
    ) -> AgentMessage:
        msg = AgentMessage(
            kind=kind,
            sender=sender,
            recipient=original.sender,
            payload=payload,
            reply_to=original.id,
            correlation_id=original.correlation_id,
            delivery="reply",
            priority=original.priority,
        )
        return self.publish(msg)

    def reject(
        self,
        original: AgentMessage,
        sender: str,
        reason: str,
    ) -> AgentMessage:
        msg = AgentMessage(
            kind="reject",
            sender=sender,
            recipient=original.sender,
            payload={"reason": reason, "original_id": original.id},
            reply_to=original.id,
            correlation_id=original.correlation_id,
            delivery="reply",
            rejected=True,
            reject_reason=reason,
            priority=original.priority,
        )
        return self.publish(msg)

    def command(
        self,
        sender: str,
        recipient: str,
        command: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: int = 2,
        ttl_seconds: float = 30.0,
    ) -> AgentMessage:
        return self.publish(
            AgentMessage(
                kind="command",
                sender=sender,
                recipient=recipient,
                payload={"command": command, **(payload or {})},
                delivery="command",
                priority=priority,
                ttl_seconds=ttl_seconds,
                correlation_id=_msg_id(),
            )
        )

    def share_context(
        self,
        sender: str,
        recipient: str,
        context: dict[str, Any],
        *,
        priority: int = 5,
    ) -> AgentMessage:
        return self.publish(
            AgentMessage(
                kind="context_share",
                sender=sender,
                recipient=recipient,
                payload=context,
                delivery="event",
                priority=priority,
                correlation_id=_msg_id(),
            )
        )

    def broadcast_event(
        self,
        sender: str,
        payload: dict[str, Any],
        *,
        kind: MessageKind = "event",
        priority: int = 5,
    ) -> AgentMessage:
        return self.publish(
            AgentMessage(
                kind=kind,
                sender=sender,
                recipient="*",
                payload=payload,
                delivery="broadcast",
                priority=priority,
                correlation_id=_msg_id(),
            )
        )

    def drain(self, agent_id: str) -> list[AgentMessage]:
        with self._lock:
            messages = list(self._inbox.get(agent_id, []))
            self._inbox[agent_id] = []
            return messages

    def history(
        self,
        *,
        agent_id: str | None = None,
        kind: MessageKind | None = None,
        limit: int = 50,
    ) -> list[AgentMessage]:
        with self._lock:
            items = list(self._history)
        if agent_id:
            items = [
                m
                for m in items
                if m.sender == agent_id or m.recipient in {agent_id, "*"}
            ]
        if kind:
            items = [m for m in items if m.kind == kind]
        return items[-limit:]

    def queue_snapshot(self, *, limit: int = 50) -> list[AgentMessage]:
        with self._lock:
            return list(self._queue)[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._inbox.clear()
            self._history.clear()
            self._queue.clear()
            self._agents.clear()


# Process-wide default bus (orchestrator may also hold an instance)
_DEFAULT_BUS: AgentMessageBus | None = None
_BUS_LOCK = threading.Lock()


def get_message_bus() -> AgentMessageBus:
    global _DEFAULT_BUS
    with _BUS_LOCK:
        if _DEFAULT_BUS is None:
            _DEFAULT_BUS = AgentMessageBus()
        return _DEFAULT_BUS

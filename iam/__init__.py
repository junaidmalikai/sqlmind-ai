"""Enterprise IAM — authentication, RBAC, ABAC, tenants, API keys, sessions, audit."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from utils.helpers import ensure_dirs, utc_now_iso
from utils.logging_config import get_logger

logger = get_logger(__name__)

RoleName = Literal[
    "viewer",
    "analyst",
    "approver",
    "admin",
    "service",
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Principal(BaseModel):
    """Authenticated identity."""

    principal_id: str
    username: str
    display_name: str = ""
    roles: list[RoleName] = Field(default_factory=lambda: ["viewer"])
    tenant_id: str = "default"
    workspace_ids: list[str] = Field(default_factory=lambda: ["default"])
    attributes: dict[str, Any] = Field(default_factory=dict)  # ABAC attrs
    disabled: bool = False


class Tenant(BaseModel):
    tenant_id: str
    name: str
    isolation_level: Literal["soft", "hard"] = "hard"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Workspace(BaseModel):
    workspace_id: str
    tenant_id: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Permission(BaseModel):
    """RBAC permission: role → resource → actions."""

    role: RoleName
    resource: str  # e.g. agent.*, tool.execute, sql.read, export.*, approval.decide
    actions: list[str] = Field(default_factory=lambda: ["allow"])


class SessionToken(BaseModel):
    session_id: str
    principal_id: str
    tenant_id: str
    workspace_id: str = "default"
    roles: list[RoleName] = Field(default_factory=list)
    issued_at: float
    expires_at: float
    api_key_id: str = ""


class ApiKeyRecord(BaseModel):
    key_id: str
    principal_id: str
    tenant_id: str
    name: str
    prefix: str
    hash: str
    scopes: list[str] = Field(default_factory=list)
    created_at: str
    revoked: bool = False


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    timestamp: str = Field(default_factory=utc_now_iso)
    actor: str
    tenant_id: str
    workspace_id: str = "default"
    action: str
    resource: str = ""
    decision: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = ""


# Default RBAC matrix
DEFAULT_PERMISSIONS: list[Permission] = [
    Permission(role="viewer", resource="sql.read", actions=["allow"]),
    Permission(role="viewer", resource="agent.schema", actions=["allow"]),
    Permission(role="viewer", resource="agent.insight", actions=["allow"]),
    Permission(role="viewer", resource="memory.read", actions=["allow"]),
    Permission(role="analyst", resource="sql.read", actions=["allow"]),
    Permission(role="analyst", resource="sql.execute", actions=["allow"]),
    Permission(role="analyst", resource="agent.*", actions=["allow"]),
    Permission(role="analyst", resource="tool.*", actions=["allow"]),
    Permission(role="analyst", resource="export.*", actions=["allow"]),
    Permission(role="analyst", resource="memory.*", actions=["allow"]),
    Permission(role="analyst", resource="plugin.execute", actions=["allow"]),
    Permission(role="approver", resource="sql.read", actions=["allow"]),
    Permission(role="approver", resource="sql.execute", actions=["allow"]),
    Permission(role="approver", resource="agent.*", actions=["allow"]),
    Permission(role="approver", resource="approval.decide", actions=["allow"]),
    Permission(role="approver", resource="export.*", actions=["allow"]),
    Permission(role="approver", resource="memory.*", actions=["allow"]),
    Permission(role="admin", resource="*", actions=["allow"]),
    Permission(role="service", resource="agent.*", actions=["allow"]),
    Permission(role="service", resource="tool.*", actions=["allow"]),
    Permission(role="service", resource="sql.read", actions=["allow"]),
    Permission(role="service", resource="sql.execute", actions=["allow"]),
    Permission(role="service", resource="memory.*", actions=["allow"]),
    Permission(role="service", resource="plugin.execute", actions=["allow"]),
]


def _hash_secret(secret: str, *, salt: str = "") -> str:
    material = f"{salt}:{secret}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _match_resource(pattern: str, resource: str) -> bool:
    if pattern == "*":
        return True
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return resource == prefix or resource.startswith(prefix + ".")
    return pattern == resource


class IAMService:
    """Production-grade enterprise identity & access for SQLMind agents."""

    def __init__(self, db_path: str, *, permissions: list[Permission] | None = None) -> None:
        self.db_path = db_path
        ensure_dirs(Path(db_path).parent)
        self._lock = threading.RLock()
        self._permissions = list(permissions or DEFAULT_PERMISSIONS)
        self._sessions: dict[str, SessionToken] = {}
        self._init()
        self._ensure_defaults()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    isolation_level TEXT,
                    metadata_json TEXT
                );
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    metadata_json TEXT
                );
                CREATE TABLE IF NOT EXISTS principals (
                    principal_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    display_name TEXT,
                    password_hash TEXT,
                    roles_json TEXT,
                    tenant_id TEXT,
                    workspace_ids_json TEXT,
                    attributes_json TEXT,
                    disabled INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    principal_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    name TEXT,
                    prefix TEXT,
                    hash TEXT NOT NULL,
                    scopes_json TEXT,
                    created_at TEXT,
                    revoked INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    actor TEXT,
                    tenant_id TEXT,
                    workspace_id TEXT,
                    action TEXT,
                    resource TEXT,
                    decision TEXT,
                    detail_json TEXT,
                    trace_id TEXT
                );
                """
            )

    def _ensure_defaults(self) -> None:
        if self.get_tenant("default") is None:
            self.create_tenant(Tenant(tenant_id="default", name="Default Tenant"))
        if self.get_workspace("default") is None:
            self.create_workspace(
                Workspace(workspace_id="default", tenant_id="default", name="Default")
            )
        if self.get_principal_by_username("local-user") is None:
            self.create_principal(
                Principal(
                    principal_id="prin-local",
                    username="local-user",
                    display_name="Local User",
                    roles=["admin"],
                    tenant_id="default",
                ),
                password="local-dev",
            )

    # -- Tenants / Workspaces -------------------------------------------------

    def create_tenant(self, tenant: Tenant) -> Tenant:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tenants VALUES (?, ?, ?, ?)",
                (
                    tenant.tenant_id,
                    tenant.name,
                    tenant.isolation_level,
                    json.dumps(tenant.metadata),
                ),
            )
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
        if not row:
            return None
        return Tenant(
            tenant_id=row["tenant_id"],
            name=row["name"],
            isolation_level=row["isolation_level"] or "hard",
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def create_workspace(self, workspace: Workspace) -> Workspace:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO workspaces VALUES (?, ?, ?, ?)",
                (
                    workspace.workspace_id,
                    workspace.tenant_id,
                    workspace.name,
                    json.dumps(workspace.metadata),
                ),
            )
        return workspace

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
            ).fetchone()
        if not row:
            return None
        return Workspace(
            workspace_id=row["workspace_id"],
            tenant_id=row["tenant_id"],
            name=row["name"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    # -- Principals / AuthN ---------------------------------------------------

    def create_principal(
        self, principal: Principal, *, password: str | None = None
    ) -> Principal:
        pw_hash = _hash_secret(password, salt=principal.principal_id) if password else ""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO principals
                (principal_id, username, display_name, password_hash, roles_json,
                 tenant_id, workspace_ids_json, attributes_json, disabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    principal.principal_id,
                    principal.username,
                    principal.display_name,
                    pw_hash,
                    json.dumps(principal.roles),
                    principal.tenant_id,
                    json.dumps(principal.workspace_ids),
                    json.dumps(principal.attributes),
                    1 if principal.disabled else 0,
                ),
            )
        return principal

    def get_principal(self, principal_id: str) -> Principal | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM principals WHERE principal_id = ?", (principal_id,)
            ).fetchone()
        return self._row_to_principal(row) if row else None

    def get_principal_by_username(self, username: str) -> Principal | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM principals WHERE username = ?", (username,)
            ).fetchone()
        return self._row_to_principal(row) if row else None

    def _row_to_principal(self, row: sqlite3.Row) -> Principal:
        return Principal(
            principal_id=row["principal_id"],
            username=row["username"],
            display_name=row["display_name"] or "",
            roles=json.loads(row["roles_json"] or "[]"),
            tenant_id=row["tenant_id"] or "default",
            workspace_ids=json.loads(row["workspace_ids_json"] or '["default"]'),
            attributes=json.loads(row["attributes_json"] or "{}"),
            disabled=bool(row["disabled"]),
        )

    def authenticate(self, username: str, password: str) -> SessionToken | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM principals WHERE username = ?", (username,)
            ).fetchone()
        if not row or row["disabled"]:
            self.audit("anonymous", "default", "auth.login", decision="denied", resource=username)
            return None
        expected = row["password_hash"] or ""
        actual = _hash_secret(password, salt=row["principal_id"])
        if not expected or not hmac.compare_digest(expected, actual):
            self.audit(username, row["tenant_id"], "auth.login", decision="denied")
            return None
        principal = self._row_to_principal(row)
        token = self.create_session(principal)
        self.audit(
            principal.username,
            principal.tenant_id,
            "auth.login",
            decision="allow",
            detail={"session_id": token.session_id},
        )
        return token

    def create_session(
        self,
        principal: Principal,
        *,
        workspace_id: str | None = None,
        ttl_seconds: int = 86400,
        api_key_id: str = "",
    ) -> SessionToken:
        now = time.time()
        token = SessionToken(
            session_id=f"sess-{uuid4().hex}",
            principal_id=principal.principal_id,
            tenant_id=principal.tenant_id,
            workspace_id=workspace_id or (principal.workspace_ids[0] if principal.workspace_ids else "default"),
            roles=list(principal.roles),
            issued_at=now,
            expires_at=now + ttl_seconds,
            api_key_id=api_key_id,
        )
        with self._lock:
            self._sessions[token.session_id] = token
        return token

    def get_session(self, session_id: str) -> SessionToken | None:
        with self._lock:
            token = self._sessions.get(session_id)
        if token is None:
            return None
        if token.expires_at < time.time():
            with self._lock:
                self._sessions.pop(session_id, None)
            return None
        return token

    def revoke_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    # -- API Keys -------------------------------------------------------------

    def create_api_key(
        self,
        principal: Principal,
        *,
        name: str = "default",
        scopes: list[str] | None = None,
    ) -> tuple[str, ApiKeyRecord]:
        raw = f"sk_sqlmind_{secrets.token_urlsafe(32)}"
        prefix = raw[:16]
        key_id = f"key-{uuid4().hex[:12]}"
        record = ApiKeyRecord(
            key_id=key_id,
            principal_id=principal.principal_id,
            tenant_id=principal.tenant_id,
            name=name,
            prefix=prefix,
            hash=_hash_secret(raw, salt=key_id),
            scopes=scopes or ["agent.*", "sql.read"],
            created_at=utc_now_iso(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_keys
                (key_id, principal_id, tenant_id, name, prefix, hash, scopes_json, created_at, revoked)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    record.key_id,
                    record.principal_id,
                    record.tenant_id,
                    record.name,
                    record.prefix,
                    record.hash,
                    json.dumps(record.scopes),
                    record.created_at,
                ),
            )
        return raw, record

    def authenticate_api_key(self, raw_key: str) -> SessionToken | None:
        prefix = raw_key[:16]
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM api_keys WHERE prefix = ? AND revoked = 0", (prefix,)
            ).fetchall()
        for row in rows:
            if hmac.compare_digest(row["hash"], _hash_secret(raw_key, salt=row["key_id"])):
                principal = self.get_principal(row["principal_id"])
                if principal is None or principal.disabled:
                    return None
                token = self.create_session(principal, api_key_id=row["key_id"])
                self.audit(
                    principal.username,
                    principal.tenant_id,
                    "auth.api_key",
                    decision="allow",
                    detail={"key_id": row["key_id"]},
                )
                return token
        self.audit("anonymous", "default", "auth.api_key", decision="denied")
        return None

    # -- RBAC / ABAC ----------------------------------------------------------

    def check_permission(
        self,
        principal: Principal | SessionToken,
        resource: str,
        *,
        action: str = "allow",
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        if isinstance(principal, SessionToken):
            roles = principal.roles
            tenant_id = principal.tenant_id
            attrs = attributes or {}
            actor = principal.principal_id
        else:
            if principal.disabled:
                return False
            roles = principal.roles
            tenant_id = principal.tenant_id
            attrs = {**principal.attributes, **(attributes or {})}
            actor = principal.username

        # ABAC: tenant isolation — resource may encode tenant
        required_tenant = attrs.get("resource_tenant_id")
        if required_tenant and required_tenant != tenant_id and "admin" not in roles:
            self.audit(actor, tenant_id, "authz.check", resource=resource, decision="deny",
                       detail={"reason": "tenant_isolation", "action": action})
            return False

        # ABAC: max_row_limit attribute
        if attrs.get("requested_rows") and attrs.get("max_rows"):
            if int(attrs["requested_rows"]) > int(attrs["max_rows"]) and "admin" not in roles:
                self.audit(actor, tenant_id, "authz.check", resource=resource, decision="deny",
                           detail={"reason": "row_limit"})
                return False

        allowed = False
        for role in roles:
            for perm in self._permissions:
                if perm.role != role:
                    continue
                if _match_resource(perm.resource, resource) and action in perm.actions:
                    allowed = True
                    break
            if allowed:
                break

        self.audit(
            actor,
            tenant_id,
            "authz.check",
            resource=resource,
            decision="allow" if allowed else "deny",
            detail={"action": action, "roles": list(roles)},
        )
        return allowed

    def assert_permission(
        self,
        principal: Principal | SessionToken,
        resource: str,
        *,
        action: str = "allow",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if not self.check_permission(principal, resource, action=action, attributes=attributes):
            raise PermissionError(f"Denied: {resource} action={action}")

    def agent_allowed(self, session: SessionToken, graph_node: str) -> bool:
        """Every agent executes inside user permissions."""
        return self.check_permission(session, f"agent.{graph_node.replace('_agent', '').replace('_node', '')}")

    # -- Audit ----------------------------------------------------------------

    def audit(
        self,
        actor: str,
        tenant_id: str,
        action: str,
        *,
        resource: str = "",
        decision: str = "",
        detail: dict[str, Any] | None = None,
        workspace_id: str = "default",
        trace_id: str = "",
    ) -> AuditEvent:
        event = AuditEvent(
            actor=actor,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            action=action,
            resource=resource,
            decision=decision,
            detail=detail or {},
            trace_id=trace_id,
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_log
                    (event_id, timestamp, actor, tenant_id, workspace_id, action,
                     resource, decision, detail_json, trace_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.timestamp,
                        event.actor,
                        event.tenant_id,
                        event.workspace_id,
                        event.action,
                        event.resource,
                        event.decision,
                        json.dumps(event.detail, default=str),
                        event.trace_id,
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audit write failed: %s", exc)
        return event

    def list_audit(
        self, *, tenant_id: str | None = None, limit: int = 100
    ) -> list[AuditEvent]:
        with self._connect() as conn:
            if tenant_id:
                rows = conn.execute(
                    """
                    SELECT * FROM audit_log WHERE tenant_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (tenant_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            AuditEvent(
                event_id=r["event_id"],
                timestamp=r["timestamp"],
                actor=r["actor"],
                tenant_id=r["tenant_id"],
                workspace_id=r["workspace_id"] or "default",
                action=r["action"],
                resource=r["resource"] or "",
                decision=r["decision"] or "",
                detail=json.loads(r["detail_json"] or "{}"),
                trace_id=r["trace_id"] or "",
            )
            for r in rows
        ]

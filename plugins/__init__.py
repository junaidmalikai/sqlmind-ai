"""Plugin Marketplace — dynamic discovery of agents, tools, skills, exporters, memories.

Planner discovers new capabilities via Plugin Manifest + Auto Registration
without changing core code. Supports versioning, hot reload, validation, health,
signing, sandboxing, install/update, and dependency resolution.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import importlib.util
import json
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from utils.logging_config import get_logger

logger = get_logger(__name__)

PluginKind = Literal["agent", "tool", "skill", "exporter", "memory", "bundle"]


def _plugin_trace(kind: str, plugin_id: str = "", status: str = "ok", **kwargs: Any) -> None:
    try:
        from observability.runtime_trace import safe_trace

        safe_trace(
            "plugin_event",
            kind=kind,
            plugin_id=plugin_id,
            status=status,
            output=kwargs.get("output"),
            detail=kwargs.get("detail"),
        )
    except Exception:  # noqa: BLE001
        pass
PluginHealthStatus = Literal["healthy", "degraded", "unhealthy", "unknown"]


class PluginCapability(BaseModel):
    """One capability exported by a plugin."""

    id: str
    kind: PluginKind
    name: str
    description: str = ""
    skills: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)
    graph_node: str | None = None
    entrypoint: str = ""  # module:attr
    risk_class: str = "medium"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PluginManifest(BaseModel):
    """Declarative plugin package descriptor (plugin.json / plugin.yaml fields)."""

    id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    min_sqlmind_version: str = "1.0.0"
    capabilities: list[PluginCapability] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    enabled: bool = True
    hot_reload: bool = True
    checksum: str = ""
    signature: str = ""
    signed_by: str = ""
    sandbox: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not v or "/" in v or "\\" in v:
            raise ValueError("plugin id must be a non-empty dotted/slug identifier")
        return v


def _parse_semver(version: str) -> tuple[int, int, int]:
    parts = (version or "0.0.0").split("+")[0].split("-")[0].split(".")
    nums = []
    for i in range(3):
        try:
            nums.append(int(parts[i]) if i < len(parts) else 0)
        except ValueError:
            nums.append(0)
    return nums[0], nums[1], nums[2]


def version_gte(current: str, required: str) -> bool:
    return _parse_semver(current) >= _parse_semver(required)


def compute_plugin_checksum(plugin_dir: Path) -> str:
    """SHA-256 over sorted relative file paths + contents (excludes .pyc)."""
    h = hashlib.sha256()
    files = sorted(
        p
        for p in plugin_dir.rglob("*")
        if p.is_file() and p.suffix not in {".pyc", ".pyo"} and "__pycache__" not in p.parts
    )
    for path in files:
        rel = path.relative_to(plugin_dir).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


class PluginSigner:
    """HMAC-SHA256 plugin signing / verification."""

    def __init__(self, secret: str = "sqlmind-plugin-dev-secret") -> None:
        self.secret = (secret or "sqlmind-plugin-dev-secret").encode("utf-8")

    def sign_checksum(self, checksum: str) -> str:
        return hmac.new(self.secret, checksum.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify(self, checksum: str, signature: str) -> bool:
        if not checksum or not signature:
            return False
        expected = self.sign_checksum(checksum)
        return hmac.compare_digest(expected, signature)


class PluginSandbox:
    """Lightweight sandbox for plugin callables.

    Restricts execution to a timeout + exception boundary. Does not grant
    filesystem writes outside the plugin directory. Production deployments
    should additionally use OS-level isolation.
    """

    FORBIDDEN_BUILTINS = frozenset(
        {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint"}
    )

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def invoke(self, handler: Any, *args: Any, **kwargs: Any) -> Any:
        if handler is None:
            raise ValueError("sandbox: empty handler")
        if not callable(handler):
            raise TypeError("sandbox: handler is not callable")
        result: dict[str, Any] = {"value": None, "error": None, "done": False}

        def _target() -> None:
            try:
                result["value"] = handler(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                result["error"] = exc
            finally:
                result["done"] = True

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(self.timeout_seconds)
        if not result["done"]:
            raise TimeoutError(
                f"Plugin sandbox timeout after {self.timeout_seconds}s"
            )
        if result["error"] is not None:
            raise result["error"]
        return result["value"]


class PluginHealth(BaseModel):
    plugin_id: str
    status: PluginHealthStatus = "unknown"
    last_check: float = 0.0
    message: str = ""
    load_errors: list[str] = Field(default_factory=list)
    capability_count: int = 0


class PluginValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_manifest(manifest: PluginManifest) -> PluginValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not manifest.capabilities:
        warnings.append("Plugin declares no capabilities")
    if not manifest.version:
        errors.append("Plugin version is required")
    seen: set[str] = set()
    for cap in manifest.capabilities:
        if cap.id in seen:
            errors.append(f"Duplicate capability id: {cap.id}")
        seen.add(cap.id)
        if not cap.entrypoint and cap.kind in {"agent", "tool", "exporter", "memory"}:
            warnings.append(f"Capability {cap.id} has empty entrypoint")
    for dep in manifest.dependencies:
        if not dep or " " in dep.strip() and "==" not in dep and ">=" not in dep:
            warnings.append(f"Unusual dependency declaration: {dep}")
    result = PluginValidationResult(valid=not errors, errors=errors, warnings=warnings)
    _plugin_trace(
        "Plugin Validation",
        plugin_id=manifest.id,
        status="ok" if result.valid else "invalid",
        detail={"errors": errors, "warnings": warnings},
    )
    return result


def _load_attr(entrypoint: str) -> Any:
    """Load ``module.path:attribute`` from an entrypoint string."""
    if ":" not in entrypoint:
        raise ValueError(f"Invalid entrypoint (expected module:attr): {entrypoint}")
    module_name, attr = entrypoint.split(":", 1)
    mod = importlib.import_module(module_name)
    obj: Any = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def load_manifest_file(path: Path) -> PluginManifest:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except ImportError:
            raise RuntimeError("PyYAML required for .yaml plugin manifests") from None
    else:
        data = json.loads(text)
    return PluginManifest.model_validate(data)


class PluginRecord(BaseModel):
    manifest: PluginManifest
    path: str = ""
    loaded_at: float = 0.0
    health: PluginHealth = Field(default_factory=lambda: PluginHealth(plugin_id=""))
    registered_capability_ids: list[str] = Field(default_factory=list)


RegisterHook = Callable[[PluginManifest, PluginCapability, Any], None]


class PluginMarketplace:
    """Discover, validate, register, hot-reload, install, sign, and sandbox plugins."""

    def __init__(
        self,
        plugin_dirs: list[str | Path] | None = None,
        *,
        on_register: RegisterHook | None = None,
        signing_secret: str = "sqlmind-plugin-dev-secret",
        require_signature: bool = False,
        sandbox_timeout: float = 10.0,
    ) -> None:
        self.plugin_dirs = [Path(p) for p in (plugin_dirs or [])]
        self.on_register = on_register
        self.require_signature = require_signature
        self.signer = PluginSigner(signing_secret)
        self.sandbox = PluginSandbox(timeout_seconds=sandbox_timeout)
        self._lock = threading.RLock()
        self._plugins: dict[str, PluginRecord] = {}
        self._mtimes: dict[str, float] = {}
        self._handlers: dict[str, Any] = {}  # capability_id → callable

    @property
    def plugins(self) -> dict[str, PluginRecord]:
        with self._lock:
            return dict(self._plugins)

    def discover(self) -> list[PluginManifest]:
        """Scan plugin directories for plugin.json / plugin.yaml manifests."""
        found: list[PluginManifest] = []
        for root in self.plugin_dirs:
            if not root.exists():
                continue
            for manifest_path in list(root.glob("*/plugin.json")) + list(
                root.glob("*/plugin.yaml")
            ):
                try:
                    manifest = load_manifest_file(manifest_path)
                    found.append(manifest)
                    self._mtimes[str(manifest_path)] = manifest_path.stat().st_mtime
                    # Ensure package importable
                    parent = str(manifest_path.parent.parent.resolve())
                    if parent not in sys.path:
                        sys.path.insert(0, parent)
                    pkg_parent = str(manifest_path.parent.resolve())
                    if pkg_parent not in sys.path:
                        sys.path.insert(0, pkg_parent)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to load plugin manifest %s: %s", manifest_path, exc)
        _plugin_trace(
            "Plugin Discovery",
            status="ok",
            detail={"count": len(found), "ids": [m.id for m in found]},
        )
        return found

    def _verify_signing(self, manifest: PluginManifest, plugin_dir: Path | None) -> None:
        checksum = manifest.checksum
        computed = ""
        if plugin_dir is not None and plugin_dir.exists():
            computed = compute_plugin_checksum(plugin_dir)
            if checksum and checksum != computed:
                msg = (
                    f"Plugin {manifest.id} checksum mismatch "
                    f"(declared={checksum[:12]}… computed={computed[:12]}…)"
                )
                if self.require_signature:
                    raise ValueError(msg)
                logger.warning("%s — continuing (require_signature=False)", msg)
            if not checksum:
                checksum = computed
                manifest.checksum = computed
        if self.require_signature:
            if not manifest.signature:
                raise ValueError(f"Plugin {manifest.id} missing required signature")
            if not self.signer.verify(checksum or computed, manifest.signature):
                raise ValueError(f"Plugin {manifest.id} signature verification failed")
        elif manifest.signature and (checksum or computed):
            if not self.signer.verify(checksum or computed, manifest.signature):
                # Stale signature after content change — warn, don't block
                logger.warning(
                    "Plugin %s signature stale — re-sign recommended", manifest.id
                )
    def resolve_dependencies(self, manifest: PluginManifest) -> list[str]:
        """Return missing dependency plugin ids (id or id>=version)."""
        missing: list[str] = []
        with self._lock:
            installed = {pid: rec.manifest.version for pid, rec in self._plugins.items()}
        for dep in manifest.dependencies:
            dep = dep.strip()
            if not dep:
                continue
            if ">=" in dep:
                pid, ver = dep.split(">=", 1)
                pid, ver = pid.strip(), ver.strip()
                cur = installed.get(pid)
                if cur is None or not version_gte(cur, ver):
                    missing.append(dep)
            elif "==" in dep:
                pid, ver = dep.split("==", 1)
                pid, ver = pid.strip(), ver.strip()
                if installed.get(pid) != ver:
                    missing.append(dep)
            else:
                if dep not in installed:
                    missing.append(dep)
        return missing

    def register_manifest(
        self,
        manifest: PluginManifest,
        *,
        path: str = "",
        load_handlers: bool = True,
    ) -> PluginRecord:
        validation = validate_manifest(manifest)
        if not validation.valid:
            raise ValueError(
                f"Invalid plugin {manifest.id}: {'; '.join(validation.errors)}"
            )
        for w in validation.warnings:
            logger.debug("Plugin %s warning: %s", manifest.id, w)

        plugin_dir = Path(path).parent if path else None
        self._verify_signing(manifest, plugin_dir)

        missing = self.resolve_dependencies(manifest)
        if missing:
            raise ValueError(
                f"Plugin {manifest.id} missing dependencies: {', '.join(missing)}"
            )

        registered: list[str] = []
        load_errors: list[str] = []
        if load_handlers and manifest.enabled:
            for cap in manifest.capabilities:
                try:
                    handler = None
                    if cap.entrypoint:
                        handler = _load_attr(cap.entrypoint)
                        if manifest.sandbox and handler is not None:
                            original = handler

                            def _sandboxed(*a: Any, _h=original, **kw: Any) -> Any:
                                return self.sandbox.invoke(_h, *a, **kw)

                            handler = _sandboxed
                    if self.on_register is not None:
                        self.on_register(manifest, cap, handler)
                    if handler is not None:
                        self._handlers[cap.id] = handler
                    registered.append(cap.id)
                except Exception as exc:  # noqa: BLE001
                    load_errors.append(f"{cap.id}: {exc}")
                    logger.warning(
                        "Plugin capability load failed id=%s: %s", cap.id, exc
                    )

        health = PluginHealth(
            plugin_id=manifest.id,
            status="healthy" if not load_errors else "degraded",
            last_check=time.time(),
            message="ok" if not load_errors else "; ".join(load_errors[:3]),
            load_errors=load_errors,
            capability_count=len(registered),
        )
        record = PluginRecord(
            manifest=manifest,
            path=path,
            loaded_at=time.time(),
            health=health,
            registered_capability_ids=registered,
        )
        with self._lock:
            existing = self._plugins.get(manifest.id)
            if existing is not None:
                # Versioning: keep newer
                if version_gte(existing.manifest.version, manifest.version) and existing.manifest.version != manifest.version:
                    logger.info(
                        "Skipping older plugin %s v%s (have v%s)",
                        manifest.id,
                        manifest.version,
                        existing.manifest.version,
                    )
                    return existing
            self._plugins[manifest.id] = record
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_plugin("register", manifest.id)
        except Exception:  # noqa: BLE001
            pass
        logger.info(
            "Registered plugin %s v%s caps=%s sandbox=%s",
            manifest.id,
            manifest.version,
            len(registered),
            manifest.sandbox,
        )
        return record

    def load_all(self) -> list[PluginRecord]:
        records: list[PluginRecord] = []
        for root in self.plugin_dirs:
            if not root.exists():
                continue
            for manifest_path in list(root.glob("*/plugin.json")) + list(
                root.glob("*/plugin.yaml")
            ):
                try:
                    manifest = load_manifest_file(manifest_path)
                    rec = self.register_manifest(
                        manifest, path=str(manifest_path), load_handlers=True
                    )
                    records.append(rec)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Plugin register failed %s: %s", manifest_path, exc)
        _plugin_trace(
            "Plugin Loading",
            status="ok",
            detail={"loaded": [r.manifest.id for r in records]},
        )
        return records

    def hot_reload(self) -> list[str]:
        """Reload plugins whose manifest mtime changed."""
        reloaded: list[str] = []
        for root in self.plugin_dirs:
            if not root.exists():
                continue
            for manifest_path in list(root.glob("*/plugin.json")) + list(
                root.glob("*/plugin.yaml")
            ):
                key = str(manifest_path)
                mtime = manifest_path.stat().st_mtime
                prev = self._mtimes.get(key)
                if prev is not None and mtime <= prev:
                    continue
                try:
                    manifest = load_manifest_file(manifest_path)
                    if not manifest.hot_reload:
                        continue
                    self.register_manifest(
                        manifest, path=key, load_handlers=True
                    )
                    self._mtimes[key] = mtime
                    reloaded.append(manifest.id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Hot reload failed %s: %s", key, exc)
        return reloaded

    def health_check(self, plugin_id: str | None = None) -> list[PluginHealth]:
        with self._lock:
            items = (
                [self._plugins[plugin_id]]
                if plugin_id and plugin_id in self._plugins
                else list(self._plugins.values())
            )
        results: list[PluginHealth] = []
        for rec in items:
            status: PluginHealthStatus = "healthy"
            msg = "ok"
            if rec.health.load_errors:
                status = "degraded"
                msg = "; ".join(rec.health.load_errors[:3])
            if not rec.manifest.enabled:
                status = "unhealthy"
                msg = "disabled"
            health = PluginHealth(
                plugin_id=rec.manifest.id,
                status=status,
                last_check=time.time(),
                message=msg,
                load_errors=list(rec.health.load_errors),
                capability_count=len(rec.registered_capability_ids),
            )
            rec.health = health
            results.append(health)
        for h in results:
            _plugin_trace(
                "Plugin Health",
                plugin_id=h.plugin_id,
                status=h.status,
                detail={"message": h.message},
            )
        return results

    def catalog(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "id": r.manifest.id,
                    "name": r.manifest.name,
                    "version": r.manifest.version,
                    "description": r.manifest.description,
                    "author": r.manifest.author,
                    "dependencies": list(r.manifest.dependencies),
                    "checksum": r.manifest.checksum,
                    "signed": bool(r.manifest.signature),
                    "sandbox": r.manifest.sandbox,
                    "enabled": r.manifest.enabled,
                    "capabilities": [c.model_dump() for c in r.manifest.capabilities],
                    "health": r.health.model_dump(),
                }
                for r in self._plugins.values()
            ]

    def get_handler(self, capability_id: str) -> Any | None:
        with self._lock:
            return self._handlers.get(capability_id)

    def execute_capability(
        self,
        capability_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        handler = self.get_handler(capability_id)
        if handler is None:
            _plugin_trace(
                "Plugin Failure",
                plugin_id=capability_id,
                status="error",
                detail="unknown capability",
            )
            raise KeyError(f"Unknown plugin capability: {capability_id}")
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_plugin("execute", capability_id)
        except Exception:  # noqa: BLE001
            pass
        try:
            result = handler(*args, **kwargs)
            _plugin_trace(
                "Plugin Execution",
                plugin_id=capability_id,
                status="ok",
                output=str(result)[:300] if result is not None else None,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            _plugin_trace(
                "Plugin Failure",
                plugin_id=capability_id,
                status="error",
                detail=str(exc)[:300],
            )
            raise

    def install_plugin(
        self,
        source: str | Path,
        *,
        target_dir: str | Path | None = None,
        sign: bool = False,
    ) -> PluginRecord:
        """Install a plugin from a directory or zip archive into the marketplace."""
        source_path = Path(source)
        if not self.plugin_dirs and target_dir is None:
            raise ValueError("No plugin_dirs configured and no target_dir provided")
        dest_root = Path(target_dir) if target_dir else self.plugin_dirs[0]
        dest_root.mkdir(parents=True, exist_ok=True)

        if source_path.is_file() and source_path.suffix.lower() == ".zip":
            tmp_name = f"plugin-{uuid4().hex[:8]}"
            extract_to = dest_root / tmp_name
            extract_to.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(source_path, "r") as zf:
                zf.extractall(extract_to)
            # Prefer nested folder containing plugin.json
            manifests = list(extract_to.glob("**/plugin.json")) + list(
                extract_to.glob("**/plugin.yaml")
            )
            if not manifests:
                raise ValueError("Zip archive contains no plugin.json/yaml")
            plugin_dir = manifests[0].parent
            final_dir = dest_root / plugin_dir.name
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.move(str(plugin_dir), str(final_dir))
            if extract_to.exists() and extract_to != final_dir:
                shutil.rmtree(extract_to, ignore_errors=True)
        elif source_path.is_dir():
            manifest_file = source_path / "plugin.json"
            if not manifest_file.exists():
                manifest_file = source_path / "plugin.yaml"
            if not manifest_file.exists():
                raise ValueError(f"No plugin manifest in {source_path}")
            final_dir = dest_root / source_path.name
            if final_dir.resolve() != source_path.resolve():
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                shutil.copytree(source_path, final_dir)
            else:
                final_dir = source_path
        else:
            raise ValueError(f"Unsupported plugin source: {source}")

        manifest_path = final_dir / "plugin.json"
        if not manifest_path.exists():
            manifest_path = final_dir / "plugin.yaml"
        manifest = load_manifest_file(manifest_path)
        checksum = compute_plugin_checksum(final_dir)
        manifest.checksum = checksum
        if sign:
            manifest.signature = self.signer.sign_checksum(checksum)
            manifest.signed_by = "sqlmind-marketplace"
            # Persist signature back to plugin.json when JSON
            if manifest_path.suffix.lower() == ".json":
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                data["checksum"] = checksum
                data["signature"] = manifest.signature
                data["signed_by"] = manifest.signed_by
                manifest_path.write_text(
                    json.dumps(data, indent=2) + "\n", encoding="utf-8"
                )
        rec = self.register_manifest(
            manifest, path=str(manifest_path), load_handlers=True
        )
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_plugin("install", manifest.id)
        except Exception:  # noqa: BLE001
            pass
        return rec

    def update_plugin(self, plugin_id: str, source: str | Path) -> PluginRecord:
        """Replace an installed plugin with a newer package from source."""
        with self._lock:
            existing = self._plugins.get(plugin_id)
        if existing is None:
            raise KeyError(f"Plugin not installed: {plugin_id}")
        old_version = existing.manifest.version
        rec = self.install_plugin(source, target_dir=Path(existing.path).parent.parent if existing.path else None)
        if not version_gte(rec.manifest.version, old_version):
            logger.warning(
                "Updated plugin %s to v%s (previous v%s)",
                plugin_id,
                rec.manifest.version,
                old_version,
            )
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_plugin("update", plugin_id)
        except Exception:  # noqa: BLE001
            pass
        return rec

    def sign_installed(self, plugin_id: str) -> PluginRecord:
        with self._lock:
            rec = self._plugins.get(plugin_id)
        if rec is None:
            raise KeyError(plugin_id)
        plugin_dir = Path(rec.path).parent if rec.path else None
        if plugin_dir is None or not plugin_dir.exists():
            raise FileNotFoundError(f"Plugin path missing for {plugin_id}")
        checksum = compute_plugin_checksum(plugin_dir)
        sig = self.signer.sign_checksum(checksum)
        rec.manifest.checksum = checksum
        rec.manifest.signature = sig
        rec.manifest.signed_by = "sqlmind-marketplace"
        if rec.path and Path(rec.path).suffix.lower() == ".json":
            data = json.loads(Path(rec.path).read_text(encoding="utf-8"))
            data["checksum"] = checksum
            data["signature"] = sig
            data["signed_by"] = "sqlmind-marketplace"
            Path(rec.path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return rec

    def discover_for_planner(self) -> list[dict[str, Any]]:
        """Dynamic capability catalog for the Planner — no hardcoded plugins."""
        self.hot_reload()
        caps: list[dict[str, Any]] = []
        for entry in self.catalog():
            if not entry.get("enabled", True):
                continue
            for cap in entry.get("capabilities") or []:
                caps.append(
                    {
                        "plugin_id": entry["id"],
                        "plugin_version": entry["version"],
                        "capability": cap,
                        "health": entry.get("health"),
                    }
                )
        return caps

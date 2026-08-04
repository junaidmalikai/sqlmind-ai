"""Autonomy kernel exceptions."""

from __future__ import annotations

from core.exceptions import SQLMindError


class KernelError(SQLMindError):
    """Base error for the autonomy kernel."""


class CapabilityNotFoundError(KernelError):
    """Requested capability id or route tool is unknown."""


class CapabilityConflictError(KernelError):
    """Registration conflict (duplicate id or route tool)."""


class CapabilityNotRoutableError(KernelError):
    """Capability exists but is not AI-routable (e.g. security gate)."""


class ContainerError(KernelError):
    """Dependency injection container error."""

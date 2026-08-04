"""SQLMind navigation + token reference.

Colors, spacing, and radii live in ``ui/styles/variables.css`` (single source of truth).
Streamlit theme mirrors those tokens in ``.streamlit/config.toml``.
"""

from __future__ import annotations

# Navigation — grouped for sidebar (glyph + label + help)
NAV_GROUPS: tuple[tuple[str, tuple[tuple[str, str, str, str], ...]], ...] = (
    (
        "Workspace",
        (
            ("Home", "⌂", "Home", "Overview and quick actions"),
            ("Chat", "◇", "Chat", "Natural-language SQL analytics"),
        ),
    ),
    (
        "Data",
        (
            ("Schema", "▦", "Schema", "Tables and relationships"),
            ("History", "◷", "History", "Past analyses"),
            ("Bookmarks", "★", "Bookmarks", "Saved queries"),
            ("Exports", "⇩", "Exports", "Download results"),
        ),
    ),
    (
        "Platform",
        (
            ("Runtime", "◎", "Runtime", "Operations console"),
            ("Settings", "⚙", "Settings", "LLM and database"),
            ("About", "ℹ", "About", "Architecture"),
        ),
    ),
)

NAV_ITEMS: tuple[tuple[str, str, str, str], ...] = tuple(
    item for _, items in NAV_GROUPS for item in items
)

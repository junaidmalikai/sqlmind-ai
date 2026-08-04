"""Pages package."""

from ui.pages.about import render_about
from ui.pages.bookmarks import render_bookmarks
from ui.pages.chat import render_chat, _handle_user_question
from ui.pages.exports import render_exports
from ui.pages.history import render_history
from ui.pages.home import render_home
from ui.pages.runtime_ops import render_runtime_ops
from ui.pages.schema import render_schema
from ui.pages.settings import render_settings

__all__ = [
    "render_about",
    "render_bookmarks",
    "render_chat",
    "render_exports",
    "render_history",
    "render_home",
    "render_runtime_ops",
    "render_schema",
    "render_settings",
    "_handle_user_question",
]

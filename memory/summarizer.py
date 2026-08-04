"""AI conversation summarization for long-context compression."""

from __future__ import annotations

from typing import Any

from config.settings import Settings
from models.structured import ConversationSummaryModel
from prompts.templates import memory_prompt
from services.llm_service import LLMService
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ConversationSummarizer:
    """Compress long chat histories into durable memory summaries."""

    def __init__(self, llm: LLMService, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    def maybe_summarize(
        self,
        history: list[dict[str, str]],
        prior_summary: str = "",
    ) -> str:
        threshold = self.settings.memory_summarize_after
        if len(history) < threshold:
            return prior_summary

        recent = history[-(threshold):]
        turns = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in recent)
        try:
            result: ConversationSummaryModel = self.llm.invoke_structured(
                memory_prompt(),
                ConversationSummaryModel,
                {
                    "prior_summary": prior_summary or "(none)",
                    "recent_turns": turns[:6000],
                },
            )
            return result.summary
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory compression failed: %s", exc)
            return prior_summary or turns[:1500]

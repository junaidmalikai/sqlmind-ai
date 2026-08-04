"""LLM provider factory with structured output and LangChain runnables."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from config.settings import Settings, get_settings
from core.exceptions import LLMProviderError
from observability import configure_observability
from utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """Factory and structured-output helper around LangChain chat models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        configure_observability(self.settings)
        self._model: BaseChatModel | None = None

    @property
    def model(self) -> BaseChatModel:
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def _build_model(self) -> BaseChatModel:
        provider = self.settings.llm_provider
        temperature = self.settings.llm_temperature
        model_name = self.settings.resolve_model()
        logger.info("Initializing LLM provider=%s model=%s", provider, model_name)

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=model_name,
                base_url=self.settings.ollama_base_url,
                temperature=temperature,
            )

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model_name,
                api_key=self.settings.openai_api_key or None,
                temperature=temperature,
                max_tokens=self.settings.llm_max_tokens,
                timeout=self.settings.llm_timeout,
            )

        if provider == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=model_name,
                api_key=self.settings.groq_api_key or None,
                temperature=temperature,
                max_tokens=self.settings.llm_max_tokens,
                timeout=self.settings.llm_timeout,
            )

        if provider == "claude":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model_name,
                api_key=self.settings.anthropic_api_key or None,
                temperature=temperature,
                max_tokens=self.settings.llm_max_tokens,
                timeout=self.settings.llm_timeout,
            )

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.settings.gemini_api_key or None,
                temperature=temperature,
                max_output_tokens=self.settings.llm_max_tokens,
            )

        raise ValueError(f"Unsupported LLM provider: {provider}")

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Runnable:
        """Bind LangChain tools so the LLM emits structured tool_calls at runtime."""
        if not tools:
            raise ValueError("bind_tools requires at least one tool")
        try:
            return self.model.bind_tools(tools, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(
                f"bind_tools failed for provider={self.settings.llm_provider}: {exc}",
                context={"provider": self.settings.llm_provider, "n_tools": len(tools)},
            ) from exc

    def structured(self, schema: type[T]) -> Runnable:
        """Return a model bound to a Pydantic schema via with_structured_output."""
        try:
            return self.model.with_structured_output(schema)
        except Exception:  # noqa: BLE001
            # Some providers (older Ollama) need json mode fallback
            logger.warning(
                "with_structured_output primary path failed; retrying with method=json_mode"
            )
            return self.model.with_structured_output(schema, method="json_mode")

    def chain(self, prompt: ChatPromptTemplate, schema: type[T] | None = None) -> Runnable:
        """Build prompt | model [| structured] runnable sequence."""
        if schema is None:
            return prompt | self.model
        return prompt | self.structured(schema)

    def invoke_structured(
        self,
        prompt: ChatPromptTemplate,
        schema: type[T],
        variables: dict[str, Any],
    ) -> T:
        """Invoke a ChatPromptTemplate and parse into a Pydantic model."""
        try:
            runnable = self.chain(prompt, schema)
            result = runnable.invoke(variables)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
            content = getattr(result, "content", result)
            if isinstance(content, str):
                return schema.model_validate_json(content)
            return schema.model_validate(content)
        except LLMProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(
                f"Structured LLM invoke failed for {schema.__name__}: {exc}",
                context={"schema": schema.__name__},
            ) from exc

    async def ainvoke_structured(
        self,
        prompt: ChatPromptTemplate,
        schema: type[T],
        variables: dict[str, Any],
    ) -> T:
        """Async structured invoke (runs sync path in a worker when provider lacks native async)."""
        return await asyncio.to_thread(self.invoke_structured, prompt, schema, variables)

    def invoke_text(
        self,
        prompt: ChatPromptTemplate,
        variables: dict[str, Any],
    ) -> str:
        try:
            runnable = self.chain(prompt)
            response = runnable.invoke(variables)
            return self._content_to_str(response)
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"Text LLM invoke failed: {exc}") from exc

    async def ainvoke_text(
        self,
        prompt: ChatPromptTemplate,
        variables: dict[str, Any],
    ) -> str:
        return await asyncio.to_thread(self.invoke_text, prompt, variables)

    def stream_text(
        self,
        prompt: ChatPromptTemplate,
        variables: dict[str, Any],
    ) -> Iterator[str]:
        """True token streaming for free-text chains."""
        runnable = prompt | self.model
        for chunk in runnable.stream(variables):
            text = self._content_to_str(chunk)
            if text:
                yield text

    async def astream_text(
        self,
        prompt: ChatPromptTemplate,
        variables: dict[str, Any],
    ) -> AsyncIterator[str]:
        runnable = prompt | self.model
        async for chunk in runnable.astream(variables):
            text = self._content_to_str(chunk)
            if text:
                yield text

    def history_messages(
        self,
        history: list[dict[str, str]] | None,
        *,
        limit: int = 12,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        for item in (history or [])[-limit:]:
            role = item.get("role", "user")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        return messages

    @staticmethod
    def _content_to_str(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "".join(parts)
        return str(content)

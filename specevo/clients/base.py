"""Minimal client abstractions for text-generation backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

ClientInput = str | list[dict[str, Any]]


@dataclass(slots=True)
class ClientResult:
    """Normalized text-generation result.

    ``prompt_tokens`` / ``completion_tokens`` mirror the provider's
    ``usage`` block (input and output tokens for this call). They default
    to 0 for backends that report no usage — CLI-driven clients, or a
    provider that omits the field — so a caller can always sum them.
    """

    text: str
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BaseClient(ABC):
    """Base class for all generation backends used by SpecEvo."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def acompletion(self, prompt: ClientInput, **kwargs: Any) -> ClientResult:
        raise NotImplementedError

    async def close(self) -> None:
        """Release any backend resources held by the client."""
        return None


ClientSpec = str | BaseClient


def client_name(spec: ClientSpec) -> str:
    """Return the model identifier associated with a client spec."""
    if isinstance(spec, BaseClient):
        return spec.model
    return spec


def short_client_name(spec: ClientSpec) -> str:
    """Return the trailing segment of a model identifier for logs."""
    return client_name(spec).split("/")[-1]

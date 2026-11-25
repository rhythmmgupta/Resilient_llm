from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: str
    keys_configured: int
    cache_enabled: bool


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1)
    system_prompt: Optional[str] = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed or len(value) > 128:
            raise ValueError("session_id must be 1 to 128 non-whitespace characters")
        return trimmed

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("message must not be empty after trimming whitespace")
        if len(trimmed) > 4000:
            raise ValueError("message must be at most 4000 characters after trimming whitespace")
        return trimmed

    @field_validator("system_prompt")
    @classmethod
    def normalize_system_prompt(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed if trimmed else None


class ChatMetadata(BaseModel):
    """Per-request metadata returned with each assistant reply."""

    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


class ChatResponse(BaseModel):
    """Response model for POST /chat."""

    session_id: str
    reply: str
    metadata: ChatMetadata


class SessionMessage(BaseModel):
    """Stored message item returned by GET /sessions/{session_id}/messages."""

    role: Literal["user", "assistant"]
    content: str
    metadata: Optional[Dict[str, Any]] = None


class SessionMessagesResponse(BaseModel):
    """Response model for GET /sessions/{session_id}/messages."""

    session_id: str
    messages: List[SessionMessage]
